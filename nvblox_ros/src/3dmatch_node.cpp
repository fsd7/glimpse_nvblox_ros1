/**
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <glog/logging.h>

#include <nvblox/datasets/image_loader.h>
#include <nvblox/datasets/parse_3dmatch.h>
#include <nvblox/nvblox.h>
#include <nvblox/utils/timing.h>

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <memory>
#include <string>
#include <vector>

#include "nvblox_ros/conversions.hpp"

namespace nvblox
{

class Nvblox3DMatchNode
{
public:
  Nvblox3DMatchNode(ros::NodeHandle& nodeHandle);

  /// Get the next frame in the dataset.
  void timerCallback(const ros::TimerEvent& event);

  /// Integrate a particular frame number.
  bool integrateFrame(const int frame_number);

private:
  /// Sets up publishing and subscribing. Should only be called from
  /// constructor.
  void setupRos();

  /// Publish markers for visualization.
  //ros::Publisher<nvblox_msgs::msg::Mesh>::SharedPtr mesh_publisher_;
  ros::Publisher mesh_publisher_;

  ros::NodeHandle nodeHandle3D_;

  /// Timers.
  //ros::TimerBase::SharedPtr update_timer_;
  ros::Timer update_timer_;
  /// Converer.
  RosConverter converter_;

  /// Frame IDs
  std::string map_frame_id_ = "map";
  std::string camera_frame_id_ = "camera";

  /// Dataset settings.
  std::string base_path_;
  float voxel_size_ = 0.2;
  int sequence_num_ = 1;
  int frame_number_ = 0;
  bool rotate_optical_frame_ = false;
  bool rotate_world_frame_ = true;

  /// NVblox layers.
  std::shared_ptr<nvblox::TsdfLayer> tsdf_layer_;
  std::shared_ptr<nvblox::EsdfLayer> esdf_layer_;
  std::shared_ptr<nvblox::MeshLayer> mesh_layer_;

  // Integrators.
  ProjectiveTsdfIntegrator tsdf_integrator_;
  MeshIntegrator mesh_integrator_;
};

Nvblox3DMatchNode::Nvblox3DMatchNode(ros::NodeHandle& nodeHandle): nodeHandle3D_(nodeHandle){
  setupRos();
}

void Nvblox3DMatchNode::setupRos()
{
  double time_between_frames = 0.5;  // seconds
  //time_between_frames =declare_parameter<float>("time_between_frames", time_between_frames);

  nodeHandle3D_.param<double>(nodeHandle3D_.getNamespace() +"/time_between_frames", time_between_frames, 0.5);

  //base_path_ = declare_parameter<std::string>("path", base_path_);

  nodeHandle3D_.param<std::string>(nodeHandle3D_.getNamespace() +"/path", base_path_, "");

  if (base_path_.empty()) {
    ROS_ERROR("No base path specified!");
    return;
  }

  // Mesh publishing
  //mesh_publisher_ = this->create_publisher<nvblox_msgs::Mesh>("mesh", 1);
  mesh_publisher_ = nodeHandle3D_.advertise<nvblox_msgs::Mesh>("mesh",1,false);

  // Image settings
  //rotate_optical_frame_ =     declare_parameter<bool>("rotate_optical_frame", rotate_optical_frame_);
  //rotate_world_frame_ =     declare_parameter<bool>("rotate_world_frame", rotate_world_frame_);

  nodeHandle3D_.param<bool>(nodeHandle3D_.getNamespace() +"/rotate_optical_frame", rotate_optical_frame_, false);
  nodeHandle3D_.param<bool>(nodeHandle3D_.getNamespace() +"/rotate_world_frame", rotate_world_frame_, true);

  // Create the layers.
  //voxel_size_ = declare_parameter<float>("voxel_size", voxel_size_);

  nodeHandle3D_.param<float>(nodeHandle3D_.getNamespace() +"/voxel_size", voxel_size_, 0.2);

  // Initialize the layers.
  const float block_size = voxel_size_ * VoxelBlock<bool>::kVoxelsPerSide;

  tsdf_layer_.reset(new TsdfLayer(voxel_size_, MemoryType::kDevice));
  esdf_layer_.reset(new EsdfLayer(voxel_size_, MemoryType::kUnified));
  mesh_layer_.reset(new MeshLayer(block_size, MemoryType::kUnified));

  mesh_integrator_.min_weight() = 2.0f;

  update_timer_ = nodeHandle3D_.createTimer(ros::Duration(time_between_frames), &Nvblox3DMatchNode::timerCallback, this);
  /*
  // Create a timer to load a new frame every n seconds.
  update_timer_ =
    create_wall_timer(
    std::chrono::duration<float>(time_between_frames),
    std::bind(&Nvblox3DMatchNode::timerCallback, this)); */
}

void Nvblox3DMatchNode::timerCallback(const ros::TimerEvent& /*event*/)
{
  if (integrateFrame(frame_number_)) {
    ROS_INFO("Outputting frame numer %d", frame_number_);
    frame_number_++;
  } else {
    // Kill timer.
    ROS_INFO("Finished dataset.");
    //update_timer_->cancel();
    update_timer_.stop();
  }
}

bool Nvblox3DMatchNode::integrateFrame(const int frame_number)
{
  if (!tsdf_layer_) {
    ROS_ERROR(
      "No layer created. Please create a layer first.");
    return false;
  }

  timing::Timer timer_file("file_loading");

  // Get the camera for this frame.
  Eigen::Matrix3f camera_intrinsics;
  if (!datasets::threedmatch::parseCameraFromFile(
      datasets::threedmatch::getPathForCameraIntrinsics(base_path_),
      &camera_intrinsics))
  {
    return false;
  }

  // Load the image into a Depth Frame.
  DepthImage depth_image;
  if (!datasets::load16BitDepthImage(
      datasets::threedmatch::getPathForDepthImage(
        base_path_, sequence_num_,
        frame_number),
      &depth_image))
  {
    return false;
  }
  // Get the transform.
  Transform T_L_C;
  if (!datasets::threedmatch::parsePoseFromFile(
      datasets::threedmatch::getPathForFramePose(
        base_path_, sequence_num_,
        frame_number),
      &T_L_C))
  {
    return false;
  }

  // Create a camera object.
  int image_width = depth_image.cols();
  int image_height = depth_image.rows();
  float fu = camera_intrinsics(0, 0);
  float fv = camera_intrinsics(1, 1);
  float cu = camera_intrinsics(0, 2);
  float cv = camera_intrinsics(1, 2);

  Camera camera(fu, fv, cu, cv, image_width, image_height);

  timer_file.Stop();

  // Rotate the camera frame to be an optical frame.
  if (rotate_optical_frame_) {
    ROS_INFO("Rotating optical.");
    Eigen::Matrix3f rotation;
    rotation << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    T_L_C = T_L_C.rotate(rotation);
  }

  // Rotate the world frame since Y is up in the normal 3D match dasets.
  if (rotate_world_frame_) {
    ROS_INFO("Rotating world.");
    Eigen::Quaternionf q_L_O = Eigen::Quaternionf::FromTwoVectors(
      Vector3f(0, 1, 0), Vector3f(0, 0, 1));
    T_L_C = q_L_O * T_L_C;
  }

  // Publish the TF pose.
  /*
  geometry_msgs::TransformStamped tf_stamped;
  tf_stamped = tf2::eigenToTransform(T_L_C.cast<double>());
  tf_stamped.header.stamp = ros::Time::now();
  tf_stamped.header.frame_id = map_frame_id_;
  tf_stamped.child_frame_id = camera_frame_id_;
  tf_broadcaster_.sendTransform(tf_stamped); */

  // Finally, processing.
  timing::Timer timer_integrate("integrate");

  // Call the integrator.
  std::vector<Index3D> updated_blocks;

  tsdf_integrator_.integrateFrame(
    depth_image, T_L_C, camera, tsdf_layer_.get(),
    &updated_blocks);
  // Mesh integrator
  mesh_integrator_.integrateBlocksGPU(
    *tsdf_layer_, updated_blocks,
    mesh_layer_.get());

  timer_integrate.Stop();

  // Publish the mesh updates.
  nvblox_msgs::Mesh mesh_msg;
  converter_.meshMessageFromMeshBlocks(*mesh_layer_, updated_blocks, &mesh_msg);

  mesh_msg.header.frame_id = map_frame_id_;
  mesh_msg.header.stamp = ros::Time::now();//get_clock()->now();
  mesh_publisher_.publish(mesh_msg);
  ROS_INFO("Published a message.");
  return true;
}

}  // namespace nvblox

int main(int argc, char ** argv)
{
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  //ros::init(argc, argv);

  ros::init(argc, argv, "nvblox_3dmatch_node");
  ros::NodeHandle nh("~");
  nvblox::Nvblox3DMatchNode Nvblox3DMatchNode(nh);
  ros::spin();
  //ros::spin(std::make_shared<nvblox::Nvblox3DMatchNode>());
  std::cout << "Timings: " << nvblox::timing::Timing::Print();

  return EXIT_SUCCESS;
}
