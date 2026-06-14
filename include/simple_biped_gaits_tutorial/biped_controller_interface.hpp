#pragma once

#include <pinocchio/fwd.hpp> // should be included before any other pinocchio headers

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <simple_biped_gaits_tutorial/biped_controller_core.hpp>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

class BipedControllerInterface {
public:
  BipedControllerInterface(ros::NodeHandle &nh, ros::NodeHandle &pnh);
  ~BipedControllerInterface() = default;

  void update();

  double getTimeStep() const { return core_->getConfig().time_step; }

protected:
  static BipedControllerCore::Config loadCoreConfig(ros::NodeHandle &nh,
                                                    ros::NodeHandle &pnh);

  void publish(int pub_idx);
  void publishRootPose(int pub_idx);
  void publishJointStates(int pub_idx);
  void publishContactWrenches(int pub_idx);
  void publishFootPoses(int pub_idx);
  void publishSolverStatistics();
  void publishFootTrajectory();

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  std::shared_ptr<BipedControllerCore> core_;

  ros::Publisher joint_states_pub_;
  ros::Publisher contact_wrench_pub_;
  ros::Publisher lfoot_traj_pub_;
  ros::Publisher rfoot_traj_pub_;
  ros::Publisher lfoot_pose_pub_;
  ros::Publisher rfoot_pose_pub_;
  ros::Publisher solver_statistics_pub_;
  tf2_ros::TransformBroadcaster robot_base_broadcaster_;
  tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_;
  std::string tf_prefix_;

  bool first_run_ = true;
  int pub_idx_ = 0;
};
