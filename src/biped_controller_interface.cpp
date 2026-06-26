#include "simple_biped_gaits_tutorial/biped_controller_interface.hpp"

#include <set>

using Scalar = BipedControllerCore::Scalar;

BipedControllerInterface::BipedControllerInterface(ros::NodeHandle &nh,
                                                   ros::NodeHandle &pnh)
    : nh_(nh), pnh_(pnh),
      core_(std::make_shared<BipedControllerCore>(loadCoreConfig(nh, pnh))) {
  tf_prefix_ = nh_.getNamespace();
  if (!tf_prefix_.empty() && tf_prefix_[0] == '/')
    tf_prefix_ = tf_prefix_.substr(1);
  std::cout << "tf prefix: " << tf_prefix_ << std::endl;

  joint_states_pub_ =
      nh_.advertise<sensor_msgs::JointState>("joint_states", 10);
  contact_wrench_pub_ =
      nh_.advertise<geometry_msgs::WrenchStamped>("contact_wrenches", 10);
  lfoot_traj_pub_ = nh_.advertise<nav_msgs::Path>("lfoot_traj", 10);
  rfoot_traj_pub_ = nh_.advertise<nav_msgs::Path>("rfoot_traj", 10);
  lfoot_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("lfoot_pose", 10);
  rfoot_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("rfoot_pose", 10);
  solver_statistics_pub_ =
      nh_.advertise<std_msgs::Float64MultiArray>("solver_statistics", 10);

  geometry_msgs::TransformStamped static_transform;
  static_transform.header.stamp = ros::Time::now();
  static_transform.transform.rotation.w = 1.0;
  for (const auto &foot : {core_->getConfig().lleg, core_->getConfig().rleg}) {
    static_transform.header.frame_id = tf::resolve(tf_prefix_, foot);
    static_transform.child_frame_id =
        tf::resolve(tf_prefix_, foot + "_contact");
    static_tf_broadcaster_.sendTransform(static_transform);
  }
}

BipedControllerCore::Config
BipedControllerInterface::loadCoreConfig(ros::NodeHandle &nh,
                                         ros::NodeHandle &pnh) {
  BipedControllerCore::Config config;
  while (!nh.getParam("robot_description", config.robot_description))
    ROS_ERROR_STREAM_THROTTLE(
        1.0, "Failed to get robot_description from ros parameter server");

  pnh.param<double>("step_length", config.step_length, config.step_length);
  pnh.param<double>("step_height", config.step_height, config.step_height);
  pnh.param<double>("time_step", config.time_step, config.time_step);
  pnh.param<int>("step_knots", config.step_knots, config.step_knots);
  pnh.param<int>("support_knots", config.support_knots, config.support_knots);
  pnh.param<int>("max_iterations", config.max_iter, config.max_iter);
  pnh.param<int>("num_steps", config.num_steps, config.num_steps);
  pnh.param<bool>("fwddyn", config.fwddyn, config.fwddyn);
  pnh.param<int>("num_threads", config.num_threads, config.num_threads);
  pnh.param<std::string>("lleg", config.lleg, config.lleg);
  pnh.param<std::string>("rleg", config.rleg, config.rleg);
  pnh.param<std::string>("root", config.root_link_name, config.root_link_name);
  std::cout << "lleg: " << config.lleg << ", rleg: " << config.rleg
            << ", root link: " << config.root_link_name << std::endl;

  int solver_type = (int)config.solver_type;
  pnh.param<int>("solver_type", solver_type, solver_type);
  config.solver_type =
      static_cast<BipedControllerCore::SolverType>(solver_type);

  pnh.getParam("initial_configuration", config.initial_configuration);

  std::vector<double> x_weights;
  pnh.getParam("x_weights", x_weights);
  if (x_weights.size() > 0)
    config.cost_weight.x_weights =
        Eigen::Map<Eigen::VectorXd>(x_weights.data(), x_weights.size());
  pnh.param<double>("control_weight", config.cost_weight.control_weight,
                    config.cost_weight.control_weight);
  pnh.param<double>("com_track_weight", config.cost_weight.com_track_weight,
                    config.cost_weight.com_track_weight);
  pnh.param<double>("centroidal_momentum_weight",
                    config.cost_weight.centroidal_momentum_weight,
                    config.cost_weight.centroidal_momentum_weight);
  pnh.param<double>("foot_track_weight", config.cost_weight.foot_track_weight,
                    config.cost_weight.foot_track_weight);
  pnh.param<double>("contact_wrench_weight",
                    config.cost_weight.contact_wrench_weight,
                    config.cost_weight.contact_wrench_weight);

  return config;
}

void BipedControllerInterface::update() {
  if (first_run_) {
    core_->createGait();
    core_->createPlanningProblem();
    core_->createSolver();
    core_->solvePlanningProblem();
    first_run_ = false;
    publishFootTrajectory();
  }

  publish(pub_idx_);
  pub_idx_++;
  if (pub_idx_ >= (int)core_->getWalkingProblem()->get_T())
    pub_idx_ = 0;
}

void BipedControllerInterface::publish(int pub_idx) {
  publishContactWrenches(pub_idx);
  publishFootPoses(pub_idx);
  publishSolverStatistics();
  publishRootPose(pub_idx);
  publishJointStates(pub_idx);
}

void BipedControllerInterface::publishContactWrenches(int pub_idx) {
  auto contacts = core_->getContactModelAndData(pub_idx);
  std::shared_ptr<crocoddyl::ContactModelMultipleTpl<Scalar>> contacts_model =
      contacts.first;
  std::shared_ptr<crocoddyl::ContactDataMultipleTpl<Scalar>> contacts_data =
      contacts.second;
  const auto &config = core_->getConfig();
  const auto &us = core_->getUs();
  const auto &model = core_->getModel();

  std::map<std::string, std::shared_ptr<crocoddyl::ContactItemTpl<Scalar>>>
      contact_container = contacts_model->get_contacts();
  std::map<std::string,
           std::shared_ptr<crocoddyl::ContactDataAbstractTpl<Scalar>>>
      contact_data_container = contacts_data->contacts;

  std::set<std::string> active_contacts_set = contacts_model->get_active_set();
  int contact_start_index = model->nv;
  for (const auto &active_contact : active_contacts_set) {
    std::shared_ptr<crocoddyl::ContactModelAbstractTpl<Scalar>> contact_model =
        contact_container[active_contact]->contact;
    std::shared_ptr<crocoddyl::ContactDataAbstractTpl<Scalar>> contact_data =
        contact_data_container[active_contact];

    Eigen::VectorXd f_contact = Eigen::VectorXd::Zero(6);

    if (auto m6d =
            std::dynamic_pointer_cast<crocoddyl::ContactModel6DTpl<Scalar>>(
                contact_model)) {
      auto d6d = std::dynamic_pointer_cast<crocoddyl::ContactData6DTpl<Scalar>>(
          contact_data);
      if (config.fwddyn) {
        switch (m6d->get_type()) {
        case pinocchio::ReferenceFrame::LOCAL:
          f_contact = d6d->f.toVector().cast<double>();
          break;
        case pinocchio::ReferenceFrame::WORLD:
        case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
          f_contact = d6d->f_local.toVector().cast<double>();
          break;
        }
      } else {
        pinocchio::ForceTpl<Scalar> f;
        f.linear() = us.at(pub_idx).segment(contact_start_index, 3);
        f.angular() = us.at(pub_idx).segment(contact_start_index + 3, 3);
        switch (m6d->get_type()) {
        case pinocchio::ReferenceFrame::LOCAL:
          f_contact = f.toVector().cast<double>();
          break;
        case pinocchio::ReferenceFrame::WORLD:
        case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
          f_contact = d6d->lwaMl.actInv(f).toVector().cast<double>();
          break;
        }
        contact_start_index += 6;
      }
    } else if (auto m3d = std::dynamic_pointer_cast<
                   crocoddyl::ContactModel3DTpl<Scalar>>(contact_model)) {
      auto d3d = std::dynamic_pointer_cast<crocoddyl::ContactData3DTpl<Scalar>>(
          contact_data);
      if (config.fwddyn) {
        f_contact.head<3>() = d3d->f.linear().cast<double>();
      } else {
        f_contact.head<3>() =
            us.at(pub_idx).segment(contact_start_index, 3).cast<double>();
        contact_start_index += 3;
      }
    }

    geometry_msgs::WrenchStamped contact_wrench_msg;
    contact_wrench_msg.header.stamp = ros::Time::now();
    contact_wrench_msg.header.frame_id =
        tf::resolve(tf_prefix_, active_contact);
    contact_wrench_msg.wrench.force.x = f_contact(0);
    contact_wrench_msg.wrench.force.y = f_contact(1);
    contact_wrench_msg.wrench.force.z = f_contact(2);
    contact_wrench_msg.wrench.torque.x = f_contact(3);
    contact_wrench_msg.wrench.torque.y = f_contact(4);
    contact_wrench_msg.wrench.torque.z = f_contact(5);
    contact_wrench_pub_.publish(contact_wrench_msg);
  }
}

void BipedControllerInterface::publishRootPose(int pub_idx) {
  const auto &model = core_->getModel();
  const auto &xs = core_->getXs();
  Eigen::VectorXd q = xs.at(pub_idx).head(model->nq).cast<double>();
  geometry_msgs::TransformStamped robot_base_transform;
  robot_base_transform.header.stamp = ros::Time::now();
  robot_base_transform.header.frame_id = "world";
  robot_base_transform.child_frame_id =
      tf::resolve(tf_prefix_, core_->getConfig().root_link_name);
  robot_base_transform.transform.translation.x = q(0);
  robot_base_transform.transform.translation.y = q(1);
  robot_base_transform.transform.translation.z = q(2);
  robot_base_transform.transform.rotation.x = q(3);
  robot_base_transform.transform.rotation.y = q(4);
  robot_base_transform.transform.rotation.z = q(5);
  robot_base_transform.transform.rotation.w = q(6);
  robot_base_broadcaster_.sendTransform(robot_base_transform);
}

void BipedControllerInterface::publishJointStates(int pub_idx) {
  std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<Scalar>> actuation_data =
      core_->getActuationData(pub_idx);
  const auto &config = core_->getConfig();
  const auto &model = core_->getModel();
  const auto &xs = core_->getXs();
  const auto &us = core_->getUs();

  Eigen::VectorXd q = xs.at(pub_idx).head(model->nq).cast<double>();
  Eigen::VectorXd v = xs.at(pub_idx).tail(model->nv).cast<double>();
  sensor_msgs::JointState joint_state;
  joint_state.header.stamp = ros::Time::now();
  for (pinocchio::JointIndex i = 2; i < model->njoints; i++) {
    std::string joint_name = model->names[i];
    int q_index = model->joints[model->getJointId(joint_name)].idx_q();
    int v_index = model->joints[model->getJointId(joint_name)].idx_v();
    joint_state.name.push_back(joint_name);
    joint_state.position.push_back(q(q_index));
    joint_state.velocity.push_back(v(v_index));
    joint_state.effort.push_back(config.fwddyn
                                     ? us.at(pub_idx)(v_index - 6)
                                     : actuation_data->u(v_index - 6));
  }
  joint_states_pub_.publish(joint_state);
}

void BipedControllerInterface::publishFootPoses(int pub_idx) {
  const auto &model = core_->getModel();
  const auto &data = core_->getData();
  const auto &xs = core_->getXs();
  const auto &config = core_->getConfig();
  Eigen::VectorXd q = xs.at(pub_idx).head(model->nq).cast<double>();
  pinocchio::forwardKinematics(*model, *data, q);
  pinocchio::updateFramePlacements(*model, *data);

  geometry_msgs::PoseStamped lfoot_pose_msg;
  lfoot_pose_msg.header.stamp = ros::Time::now();
  lfoot_pose_msg.header.frame_id = "world";
  lfoot_pose_msg.pose.position.x =
      data->oMf[model->getFrameId(config.lleg)].translation().x();
  lfoot_pose_msg.pose.position.y =
      data->oMf[model->getFrameId(config.lleg)].translation().y();
  lfoot_pose_msg.pose.position.z =
      data->oMf[model->getFrameId(config.lleg)].translation().z();
  Eigen::Quaterniond lfoot_quat(
      data->oMf[model->getFrameId(config.lleg)].rotation());
  lfoot_pose_msg.pose.orientation.x = lfoot_quat.x();
  lfoot_pose_msg.pose.orientation.y = lfoot_quat.y();
  lfoot_pose_msg.pose.orientation.z = lfoot_quat.z();
  lfoot_pose_msg.pose.orientation.w = lfoot_quat.w();
  lfoot_pose_pub_.publish(lfoot_pose_msg);

  geometry_msgs::PoseStamped rfoot_pose_msg;
  rfoot_pose_msg.header.stamp = ros::Time::now();
  rfoot_pose_msg.header.frame_id = "world";
  rfoot_pose_msg.pose.position.x =
      data->oMf[model->getFrameId(config.rleg)].translation().x();
  rfoot_pose_msg.pose.position.y =
      data->oMf[model->getFrameId(config.rleg)].translation().y();
  rfoot_pose_msg.pose.position.z =
      data->oMf[model->getFrameId(config.rleg)].translation().z();
  Eigen::Quaterniond rfoot_quat(
      data->oMf[model->getFrameId(config.rleg)].rotation());
  rfoot_pose_msg.pose.orientation.x = rfoot_quat.x();
  rfoot_pose_msg.pose.orientation.y = rfoot_quat.y();
  rfoot_pose_msg.pose.orientation.z = rfoot_quat.z();
  rfoot_pose_msg.pose.orientation.w = rfoot_quat.w();
  rfoot_pose_pub_.publish(rfoot_pose_msg);
}

void BipedControllerInterface::publishSolverStatistics() {
  const auto &solver = core_->getSolver();
  std_msgs::Float64MultiArray solver_stats_msg;
  solver_stats_msg.data.push_back(core_->getLastSolveTime());
  solver_stats_msg.data.push_back((double)solver->get_iter());
  solver_stats_msg.data.push_back(core_->getLastSolveTime() /
                                  solver->get_iter());
  solver_stats_msg.data.push_back(solver->get_cost());
  solver_stats_msg.data.push_back(solver->get_stop());
  solver_statistics_pub_.publish(solver_stats_msg);
}

void BipedControllerInterface::publishFootTrajectory() {
  const auto &model = core_->getModel();
  const auto &data = core_->getData();
  const auto &xs = core_->getXs();
  const auto &config = core_->getConfig();

  nav_msgs::Path lfoot_path;
  nav_msgs::Path rfoot_path;
  lfoot_path.header.stamp = ros::Time::now();
  lfoot_path.header.frame_id = "world";
  rfoot_path.header.stamp = ros::Time::now();
  rfoot_path.header.frame_id = "world";

  pinocchio::FrameIndex lfoot_id = model->getFrameId(config.lleg);
  pinocchio::FrameIndex rfoot_id = model->getFrameId(config.rleg);
  for (int i = 0; i < (int)xs.size(); i++) {
    Eigen::VectorXd q_i = xs.at(i).head(model->nq).cast<double>();
    pinocchio::forwardKinematics(*model, *data, q_i);
    pinocchio::updateFramePlacements(*model, *data);
    {
      geometry_msgs::PoseStamped lfoot_point;
      lfoot_point.pose.position.x = data->oMf[lfoot_id].translation().x();
      lfoot_point.pose.position.y = data->oMf[lfoot_id].translation().y();
      lfoot_point.pose.position.z = data->oMf[lfoot_id].translation().z();
      lfoot_point.pose.orientation.w = 1.0;
      lfoot_path.poses.push_back(lfoot_point);
    }
    {
      geometry_msgs::PoseStamped rfoot_point;
      rfoot_point.pose.position.x = data->oMf[rfoot_id].translation().x();
      rfoot_point.pose.position.y = data->oMf[rfoot_id].translation().y();
      rfoot_point.pose.position.z = data->oMf[rfoot_id].translation().z();
      rfoot_point.pose.orientation.w = 1.0;
      rfoot_path.poses.push_back(rfoot_point);
    }
  }
  lfoot_traj_pub_.publish(lfoot_path);
  rfoot_traj_pub_.publish(rfoot_path);
}
