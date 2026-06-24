// Copyright (c) 2019 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE

#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <franka_bimanual_controllers/bimanual_cartesian_impedance_controller_with_learned_controller.h>

#include <cmath>
#include <functional>
#include <memory>

#include <controller_interface/controller_base.h>
#include <eigen_conversions/eigen_msg.h>
#include <franka/robot_state.h>
#include <franka_bimanual_controllers/pseudo_inversion.h>
#include <franka_bimanual_controllers/franka_model.h>
#include <franka_hw/trigger_rate.h>
#include <geometry_msgs/PoseStamped.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <ros/transport_hints.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include "sensor_msgs/JointState.h"


namespace franka_bimanual_controllers {

void BiManualCartesianImpedanceControlWithLearnedController::loadModel() {
  std::cout << "LOADING THE MODEL" << std::endl;
  std::string package_path = ros::package::getPath("franka_bimanual_controllers");
  urdf_path_left = package_path + "/urdf/panda_calibrated_left.urdf";
  urdf_path_right = package_path + "/urdf/panda_calibrated_right.urdf";
  std::cout << "URDF Path Left: " << urdf_path_left << std::endl;
  std::cout << "URDF Path Right: " << urdf_path_right << std::endl;
  ros::param::get("frame_name_left", frame_name_left_);
  ros::param::get("frame_name_right", frame_name_right_);
  std::cout << "Frame Name Right: " << frame_name_right_ << std::endl;
  std::cout << "Frame Name Left: " << frame_name_left_ << std::endl;

  std::cout << "Loading urdf into pinocchio as we are using the calibrated urdf model" << std::endl;
  pinocchio::urdf::buildModel(urdf_path_left, model_pin_left_);
  data_pin_left_ = new pinocchio::Data(model_pin_left_);
  pinocchio::urdf::buildModel(urdf_path_right, model_pin_right_);
  data_pin_right_ = new pinocchio::Data(model_pin_right_);
  std::cout << "Succesfully loaded the model and created the data pointer for both the robots." << std::endl;
}
double* BiManualCartesianImpedanceControlWithLearnedController::get_fk(franka::RobotState robot_state, pinocchio::Model& model_pin, pinocchio::Data* data_pin, const std::string& frame_name)
{
  // cout << "Getting the forward kinematics" << endl;
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::VectorXd q_vector = Eigen::VectorXd::Map(q.data(), q.size());

  pinocchio::forwardKinematics(model_pin, *data_pin, q_vector);
  pinocchio::updateFramePlacement(model_pin, *data_pin, model_pin.getFrameId(frame_name));
  const auto& transformation = data_pin->oMf[model_pin.getFrameId(frame_name)];  // Get the transformation of the frame

  // Allocate memory for the result
  double* result = new double[16];
  std::memcpy(result, transformation.toHomogeneousMatrix().data(), 16 * sizeof(double));
  // std::cout << "Forward Kinematics: " << transformation.toHomogeneousMatrix() << std::endl;
  return result; // Caller is responsible for deleting the allocated memory
}

std::array<double, 42> BiManualCartesianImpedanceControlWithLearnedController::get_jacobian(franka::RobotState robot_state, pinocchio::Model& model_pin, pinocchio::Data* data_pin, const std::string& frame_name)
{
  // cout << "Getting the jacobian" << endl;
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::VectorXd q_vector = Eigen::VectorXd::Map(q.data(), q.size());
  Eigen::MatrixXd jacobian(6, model_pin.nv);  // 6xnv matrix for spatial Jacobian
  jacobian.fill(0);  // Initialize to zero

  pinocchio::forwardKinematics(model_pin, *data_pin, q_vector);
  pinocchio::computeJointJacobians(model_pin, *data_pin, q_vector);
  pinocchio::getFrameJacobian(model_pin, *data_pin, model_pin.getFrameId(frame_name), pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
  std::array<double, 42> result;
  std::memcpy(result.data(), jacobian.data(), 42 * sizeof(double));
  // std::cout << "Jacobian: " << jacobian << std::endl;
  return result;
}

bool BiManualCartesianImpedanceControlWithLearnedController::initArm(
    hardware_interface::RobotHW* robot_hw,
    const std::string& arm_id,
    const std::vector<std::string>& joint_names) {
  FrankaDataContainer arm_data;
  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControlWithLearnedController: Error getting model interface from hardware");
    return false;
  }
  try {
    arm_data.model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControlWithLearnedController: Exception getting model handle from "
        "interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControlWithLearnedController: Error getting state interface from hardware");
    return false;
  }
  try {
    arm_data.state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControlWithLearnedController: Exception getting state handle from "
        "interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControlWithLearnedController: Error getting effort joint interface from "
        "hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      arm_data.joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "BiManualCartesianImpedanceControlWithLearnedController: Exception getting joint handles: "
          << ex.what());
      return false;
    }
  }

  arm_data.position_d_.setZero();
  arm_data.orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  arm_data.cartesian_stiffness_.setZero();
  arm_data.cartesian_damping_.setZero();
  arm_data.force_torque.setZero();

  arms_data_.emplace(std::make_pair(arm_id, std::move(arm_data)));

  return true;
}

bool BiManualCartesianImpedanceControlWithLearnedController::init(hardware_interface::RobotHW* robot_hw,
                                                      ros::NodeHandle& node_handle) {
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;

  this->loadModel();

  // Read parameters for left arm
  double left_orientation_w, left_orientation_x, left_orientation_y, left_orientation_z;
  double left_position_x, left_position_y, left_position_z;

  if (!node_handle.getParam("/left/quaternion/w", left_orientation_w) ||
      !node_handle.getParam("/left/quaternion/x", left_orientation_x) ||
      !node_handle.getParam("/left/quaternion/y", left_orientation_y) ||
      !node_handle.getParam("/left/quaternion/z", left_orientation_z) ||
      !node_handle.getParam("/left/position/x", left_position_x) ||
      !node_handle.getParam("/left/position/y", left_position_y) ||
      !node_handle.getParam("/left/position/z", left_position_z)) {
    ROS_ERROR("Failed to get left arm parameters");
    return false;
  }

  // Create transformation matrix for left arm
  R_left.setIdentity();
  R_left = Eigen::Quaterniond(left_orientation_w, left_orientation_x, left_orientation_y, left_orientation_z).toRotationMatrix();

  t_left.setZero();
  t_left << left_position_x, left_position_y, left_position_z;
  // Read parameters for right arm
  double right_orientation_w, right_orientation_x, right_orientation_y, right_orientation_z;
  double right_position_x, right_position_y, right_position_z;

  if (!node_handle.getParam("/right/quaternion/w", right_orientation_w) ||
      !node_handle.getParam("/right/quaternion/x", right_orientation_x) ||
      !node_handle.getParam("/right/quaternion/y", right_orientation_y) ||
      !node_handle.getParam("/right/quaternion/z", right_orientation_z) ||
      !node_handle.getParam("/right/position/x", right_position_x) ||
      !node_handle.getParam("/right/position/y", right_position_y) ||
      !node_handle.getParam("/right/position/z", right_position_z)) {
    ROS_ERROR("Failed to get right arm parameters");
    return false;
  }

  // Create transformation matrix for right arm
  R_right.setIdentity();
  R_right = Eigen::Quaterniond(right_orientation_w, right_orientation_x, right_orientation_y, right_orientation_z).toRotationMatrix();
  t_right.setZero();
  t_right << right_position_x, right_position_y, right_position_z;

  // Print the transformation matrices
  std::cout << "Transformation matrix for left arm: " << std::endl;
  std::cout << "Rotation matrix: " << std::endl;
  std::cout << R_left << std::endl;
  std::cout << "Translation vector: " << std::endl;
  std::cout << t_left << std::endl;
  std::cout << "Transformation matrix for right arm: " << std::endl;
  std::cout << "Rotation matrix: " << std::endl;
  std::cout << R_right << std::endl;
  std::cout << "Translation vector: " << std::endl;
  std::cout << t_right << std::endl;
  if (!node_handle.getParam("left/arm_id", left_arm_id_)) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControlWithLearnedController: Could not read parameter left_arm_id_");
    return false;
  }

  // Create transformation matrix for left arm

  transform_base_to_left.linear() = R_left;
  transform_base_to_left.translation() = t_left;
  transform_base_to_right.linear() = R_right;
  transform_base_to_right.translation() = t_right;
  transform_left_to_base = transform_base_to_left.inverse();
  transform_right_to_base = transform_base_to_right.inverse();

  std::vector<std::string> left_joint_names;
  if (!node_handle.getParam("left/joint_names", left_joint_names) || left_joint_names.size() != 7) {
    ROS_ERROR(
        "BiManualCartesianImpedanceControlWithLearnedController: Invalid or no left_joint_names parameters "
        "provided, "
        "aborting controller init!");
    return false;
  }

  if (!node_handle.getParam("right/arm_id", right_arm_id_)) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControlWithLearnedController: Could not read parameter right_arm_id_");
    return false;
  }

  std::vector<std::string> right_joint_names;
  if (!node_handle.getParam("right/joint_names", right_joint_names) ||
      right_joint_names.size() != 7) {
    ROS_ERROR(
        "BiManualCartesianImpedanceControlWithLearnedController: Invalid or no right_joint_names parameters "
        "provided, "
        "aborting controller init!");
    return false;
  }

  bool left_success = initArm(robot_hw, left_arm_id_, left_joint_names);
  bool right_success = initArm(robot_hw, right_arm_id_, right_joint_names);

  sub_equilibrium_pose_right_ = node_handle.subscribe(
      "panda_right_equilibrium_pose", 20, &BiManualCartesianImpedanceControlWithLearnedController::equilibriumPoseCallback_right, this,
      ros::TransportHints().reliable().tcpNoDelay());

  sub_equilibrium_pose_right_global_frame_ = node_handle.subscribe(
      "panda_right_equilibrium_pose_global_frame", 20, &BiManualCartesianImpedanceControlWithLearnedController::equilibriumPoseCallback_right_global, this,
      ros::TransportHints().reliable().tcpNoDelay());

  sub_equilibrium_pose_left_ = node_handle.subscribe(
      "panda_left_equilibrium_pose", 20, &BiManualCartesianImpedanceControlWithLearnedController::equilibriumPoseCallback_left, this,
      ros::TransportHints().reliable().tcpNoDelay());

  sub_equilibrium_pose_left_global_frame_ = node_handle.subscribe(
      "panda_left_equilibrium_pose_global_frame", 20, &BiManualCartesianImpedanceControlWithLearnedController::equilibriumPoseCallback_left_global, this,
      ros::TransportHints().reliable().tcpNoDelay());



  sub_nullspace_right_ = node_handle.subscribe(
    "panda_right_nullspace", 20, &BiManualCartesianImpedanceControlWithLearnedController::equilibriumConfigurationCallback_right, this,
    ros::TransportHints().reliable().tcpNoDelay());

  sub_nullspace_left_ = node_handle.subscribe(
    "panda_left_nullspace", 20, &BiManualCartesianImpedanceControlWithLearnedController::equilibriumConfigurationCallback_left, this,
    ros::TransportHints().reliable().tcpNoDelay());

  // sub_equilibrium_distance_ = node_handle.subscribe(
  //       "equilibrium_distance", 20, &BiManualCartesianImpedanceControlWithLearnedController::equilibriumPoseCallback_relative, this,
  //       ros::TransportHints().reliable().tcpNoDelay());

  // Learned DLO controller force injection
  learned_force_left_.setZero();
  learned_force_right_.setZero();
  last_learned_force_time_ = ros::Time(0);
  sub_learned_controller_ = node_handle.subscribe(
      "/learned_dlo_controller/dlo_control_commands", 1, &BiManualCartesianImpedanceControlWithLearnedController::learnedControllerCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  node_handle.param<double>("learned_force_max", learned_force_max_, 20.0);
  node_handle.param<int>("learned_controller_dim", learned_controller_dim_, 36);
  ROS_INFO("Learned controller force norm limit: %.1f N", learned_force_max_);

  node_handle.param<double>("learned_force_timeout", learned_force_timeout_, 0.5);
  ROS_INFO("Learned controller force timeout: %.3f s", learned_force_timeout_);

  ROS_INFO("Learned controller command dimension: %d", learned_controller_dim_);

  // Parse learned force reference frame: "base"/"robot_base", "end_effector", or "world"
  std::string learned_force_frame_str;
  node_handle.param<std::string>("learned_force_frame", learned_force_frame_str, "base");
  if (learned_force_frame_str == "end_effector") {
    learned_force_frame_ = LearnedForceFrame::END_EFFECTOR;
    ROS_INFO("Learned controller forces defined in End Effector frame");
  } else if (learned_force_frame_str == "world") {
    learned_force_frame_ = LearnedForceFrame::WORLD;
    ROS_INFO("Learned controller forces defined in World frame");
    // Look up static rotation from world to each robot base frame
    R_O_W_left_.setIdentity();
    R_O_W_right_.setIdentity();
    try {
      tf_listener_.waitForTransform(left_arm_id_ + "_link0", "world", ros::Time(0), ros::Duration(5.0));
      tf::StampedTransform tf_world_to_left;
      tf_listener_.lookupTransform(left_arm_id_ + "_link0", "world", ros::Time(0), tf_world_to_left);
      Eigen::Affine3d eigen_tf;
      tf::transformTFToEigen(tf_world_to_left, eigen_tf);
      R_O_W_left_ = eigen_tf.rotation();
    } catch (tf::TransformException& ex) {
      ROS_WARN("Could not get world->%s_link0 transform: %s. Using identity.", left_arm_id_.c_str(), ex.what());
    }
    try {
      tf_listener_.waitForTransform(right_arm_id_ + "_link0", "world", ros::Time(0), ros::Duration(5.0));
      tf::StampedTransform tf_world_to_right;
      tf_listener_.lookupTransform(right_arm_id_ + "_link0", "world", ros::Time(0), tf_world_to_right);
      Eigen::Affine3d eigen_tf;
      tf::transformTFToEigen(tf_world_to_right, eigen_tf);
      R_O_W_right_ = eigen_tf.rotation();
    } catch (tf::TransformException& ex) {
      ROS_WARN("Could not get world->%s_link0 transform: %s. Using identity.", right_arm_id_.c_str(), ex.what());
    }
  } else {
    learned_force_frame_ = LearnedForceFrame::ROBOT_BASE;
    ROS_INFO("Learned controller forces defined in Robot Base frame");
  }


  pub_right = node_handle.advertise<geometry_msgs::PoseStamped>("panda_right_cartesian_pose", 1);

  pub_left = node_handle.advertise<geometry_msgs::PoseStamped>("panda_left_cartesian_pose", 1);

  pub_right_global_frame = node_handle.advertise<geometry_msgs::PoseStamped>("panda_right_cartesian_pose_global_frame", 1);
  pub_left_global_frame = node_handle.advertise<geometry_msgs::PoseStamped>("panda_left_cartesian_pose_global_frame", 1);

  pub_force_torque_right= node_handle.advertise<geometry_msgs::WrenchStamped>("/force_torque_right_ext",1);
  pub_force_torque_left= node_handle.advertise<geometry_msgs::WrenchStamped>("/force_torque_left_ext",1);

  pub_cartesian_wrench_task_right_ = node_handle.advertise<geometry_msgs::WrenchStamped>("/cartesian_wrench_task_right", 1);
  pub_cartesian_wrench_task_left_ = node_handle.advertise<geometry_msgs::WrenchStamped>("/cartesian_wrench_task_left", 1);
  cartesian_right_publish_decimation_counter_ = 0;
  cartesian_left_publish_decimation_counter_ = 0;
  node_handle.param<int>("cartesian_pub_decimation_factor", decimation_factor_, 10);


  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle("dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_ = std::make_unique<dynamic_reconfigure::Server<
      franka_combined_bimanual_controllers::dual_arm_compliance_paramConfig>>(
      dynamic_reconfigure_compliance_param_node_);

  dynamic_server_compliance_param_->setCallback(boost::bind(
      &BiManualCartesianImpedanceControlWithLearnedController::complianceParamCallback, this, _1, _2));

        // Define variables to store parameter values
  const std::string limit_types[2] = {"lower", "upper"};

  // Read parameters from the parameter server
  for (int i = 0; i < 7; ++i) {
      for (const std::string& limit_type : limit_types) {
          std::string param_name = "/joint" + std::to_string(i + 1) + "/limit/" + limit_type;
          if (!node_handle.getParam(param_name, joint_limits[i][limit_type == "lower" ? 0 : 1])) {
              ROS_ERROR("Failed to retrieve parameter: %s", param_name.c_str());
              return 1;
          }
      }
  }

    // Stream the parameter values
    ROS_INFO("Joint limits:");
    for (int i = 0; i < 7; ++i) {
        ROS_INFO("Joint %d: lower=%.4f, upper=%.4f", i + 1, joint_limits[i][0], joint_limits[i][1]);
    }

   return left_success && right_success;
}

void BiManualCartesianImpedanceControlWithLearnedController::starting(const ros::Time& /*time*/) {
startingArmLeft();
startingArmRight();
}

void BiManualCartesianImpedanceControlWithLearnedController::update(const ros::Time& /*time*/,
                                                        const ros::Duration& /*period*/) {
  if (last_learned_force_time_.isZero() ||
      (ros::Time::now() - last_learned_force_time_).toSec() > learned_force_timeout_) {
    if (!learned_force_left_.isZero() || !learned_force_right_.isZero()) {
      ROS_WARN_THROTTLE(1.0, "Learned forces stale (>%.3fs since last msg), zeroing.", learned_force_timeout_);
    }
    learned_force_left_.setZero();
    learned_force_right_.setZero();
  }
  updateArmLeft();
  updateArmRight();
}

void BiManualCartesianImpedanceControlWithLearnedController::startingArmLeft() {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  auto& left_arm_data = arms_data_.at(left_arm_id_);

  franka::RobotState initial_state = left_arm_data.state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array = this->get_jacobian(initial_state, model_pin_left_,  data_pin_left_, frame_name_left_);
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq_initial(initial_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  double* O_T_EE = this->get_fk(initial_state, model_pin_left_, data_pin_left_, frame_name_left_);
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(O_T_EE));

  // set target point to current state
  left_arm_data.position_d_ = initial_transform.translation();
  left_arm_data.orientation_d_ = Eigen::Quaterniond(initial_transform.linear());
  left_arm_data.position_d_ = initial_transform.translation();
  left_arm_data.orientation_d_ = Eigen::Quaterniond(initial_transform.linear());

  // set nullspace target configuration to initial q
  left_arm_data.q_d_nullspace_ = q_initial;

}

void BiManualCartesianImpedanceControlWithLearnedController::startingArmRight() {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  franka::RobotState initial_state = right_arm_data.state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array = this->get_jacobian(initial_state, model_pin_right_,data_pin_right_, frame_name_right_);
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq_initial(initial_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  double* O_T_EE = this->get_fk(initial_state,  model_pin_right_,  data_pin_right_, frame_name_right_);
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(O_T_EE));

  // set target point to current state
  right_arm_data.position_d_ = initial_transform.translation();
  right_arm_data.orientation_d_ = Eigen::Quaterniond(initial_transform.linear());
  right_arm_data.position_d_ = initial_transform.translation();
  right_arm_data.orientation_d_ = Eigen::Quaterniond(initial_transform.linear());


  // set nullspace target configuration to initial q
  right_arm_data.q_d_nullspace_ = q_initial;
}

void BiManualCartesianImpedanceControlWithLearnedController::learnedControllerCallback(
    const std_msgs::Float64MultiArray::ConstPtr& msg) {
  if (static_cast<int>(msg->data.size()) >= learned_controller_dim_) {
    // First 3 elements: left endpoint forces [Fx1, Fy1, Fz1]
    learned_force_left_ << msg->data[0], msg->data[1], msg->data[2];
    // Last 3 elements: right endpoint forces [Fx2, Fy2, Fz2]
    learned_force_right_ << msg->data[learned_controller_dim_ - 3],
                            msg->data[learned_controller_dim_ - 2],
                            msg->data[learned_controller_dim_ - 1];
    last_learned_force_time_ = ros::Time::now();

    // print some info
      ROS_INFO_THROTTLE(0.3, "Received learned forces - Left: [%.2f, %.2f, %.2f] N, Right: [%.2f, %.2f, %.2f] N",
          learned_force_left_.x(), learned_force_left_.y(), learned_force_left_.z(),
          learned_force_right_.x(), learned_force_right_.y(), learned_force_right_.z());
    // debug return early for now
    return;

    // Clamp force norms to learned_force_max_, rescaling if exceeded
    double norm_left = learned_force_left_.norm();
    if (norm_left > learned_force_max_) {
      ROS_WARN_THROTTLE(1.0, "Learned left force norm (%.2f N) exceeds limit (%.1f N), rescaling.", norm_left, learned_force_max_);
      learned_force_left_ *= learned_force_max_ / norm_left;
    }
    double norm_right = learned_force_right_.norm();
    if (norm_right > learned_force_max_) {
      ROS_WARN_THROTTLE(1.0, "Learned right force norm (%.2f N) exceeds limit (%.1f N), rescaling.", norm_right, learned_force_max_);
      learned_force_right_ *= learned_force_max_ / norm_right;
    }
  }
}

void BiManualCartesianImpedanceControlWithLearnedController::updateArmLeft() {
  // get state variables
  auto& left_arm_data = arms_data_.at(left_arm_id_);
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  franka::RobotState robot_state_left = left_arm_data.state_handle_->getRobotState();
  // franka::RobotState robot_state_right = right_arm_data.state_handle_->getRobotState();

  std::array<double, 49> inertia_array = left_arm_data.model_handle_->getMass();
  std::array<double, 7> coriolis_array = left_arm_data.model_handle_->getCoriolis();

  std::array<double, 42> jacobian_array = this->get_jacobian(robot_state_left, model_pin_left_,  data_pin_left_, frame_name_left_);
  // std::array<double, 42> jacobian_array_right = this->get_jacobian(robot_state_right, model_pin_right_,  data_pin_right_);

  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  // Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_right(jacobian_array_right.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state_left.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state_left.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d( robot_state_left.tau_J_d.data());
  double* O_T_EE = this->get_fk(robot_state_left, model_pin_left_,  data_pin_left_, frame_name_left_);
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(O_T_EE));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());
  Eigen::MatrixXd jacobian_transpose_pinv;
  franka_bimanual_controllers::pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  // find the transformatio in base frame
  Eigen::Affine3d transform_left_base_frame = transform_base_to_left * transform;
  // publish this transformation to the topic
  geometry_msgs::PoseStamped msg_left_global;
  msg_left_global.pose.position.x=transform_left_base_frame.translation()[0];
  msg_left_global.pose.position.y=transform_left_base_frame.translation()[1];
  msg_left_global.pose.position.z=transform_left_base_frame.translation()[2];
  Eigen::Quaterniond orientation_base_to_left(transform_left_base_frame.linear());
  msg_left_global.pose.orientation.x=orientation_base_to_left.x();
  msg_left_global.pose.orientation.y=orientation_base_to_left.y();
  msg_left_global.pose.orientation.z=orientation_base_to_left.z();
  msg_left_global.pose.orientation.w=orientation_base_to_left.w();
  pub_left_global_frame.publish(msg_left_global);
  // left_arm_data.position_other_arm_=position_right;
  // compute error to desired pose
  // position error
  geometry_msgs::PoseStamped msg_left;
  msg_left.pose.position.x=position[0];
  msg_left.pose.position.y=position[1];
  msg_left.pose.position.z=position[2];

  msg_left.pose.orientation.x=orientation.x();
  msg_left.pose.orientation.y=orientation.y();
  msg_left.pose.orientation.z=orientation.z();
  msg_left.pose.orientation.w=orientation.w();
  pub_left.publish(msg_left);


  Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_ext(robot_state_left.tau_ext_hat_filtered.data());
  Eigen::Matrix<double, 7, 1>  tau_f;

  // Compute the value of the friction
  tau_f(0) =  FI_11/(1+exp(-FI_21*(dq(0)+FI_31))) - TAU_F_CONST_1;
  tau_f(1) =  FI_12/(1+exp(-FI_22*(dq(1)+FI_32))) - TAU_F_CONST_2;
  tau_f(2) =  FI_13/(1+exp(-FI_23*(dq(2)+FI_33))) - TAU_F_CONST_3;
  tau_f(3) =  FI_14/(1+exp(-FI_24*(dq(3)+FI_34))) - TAU_F_CONST_4;
  tau_f(4) =  FI_15/(1+exp(-FI_25*(dq(4)+FI_35))) - TAU_F_CONST_5;
  tau_f(5) =  FI_16/(1+exp(-FI_26*(dq(5)+FI_36))) - TAU_F_CONST_6;
  tau_f(6) =  FI_17/(1+exp(-FI_27*(dq(6)+FI_37))) - TAU_F_CONST_7;

  float iCutOffFrequency=10.0;
  left_arm_data.force_torque+=(-jacobian_transpose_pinv*(tau_ext-tau_f)-left_arm_data.force_torque)*(1-exp(-0.001 * 2.0 * M_PI * iCutOffFrequency));
  geometry_msgs::WrenchStamped force_torque_msg;
  force_torque_msg.wrench.force.x=left_arm_data.force_torque[0];
  force_torque_msg.wrench.force.y=left_arm_data.force_torque[1];
  force_torque_msg.wrench.force.z=left_arm_data.force_torque[2];
  force_torque_msg.wrench.torque.x=left_arm_data.force_torque[3];
  force_torque_msg.wrench.torque.y=left_arm_data.force_torque[4];
  force_torque_msg.wrench.torque.z=left_arm_data.force_torque[5];
  pub_force_torque_left.publish(force_torque_msg);


  // std::array< double, 3 > gravity_global={{0., 0.,-9.81}};
  Eigen::Vector3d gravity_global(0., 0.,-9.81);
    std::array<double, 3> gravity_global_array = {gravity_global[0], gravity_global[1], gravity_global[2]};
  Eigen::Vector3d gravity_local = transform_left_to_base * gravity_global;
  std::array<double, 3> gravity_local_array = {gravity_local[0], gravity_local[1], gravity_local[2]};
  // std::cout << "Gravity Local left: " << gravity_local << std::endl;
  std::array<double, 7> tau_gravity_internal_array = left_arm_data.model_handle_->getGravity(gravity_global_array);
  std::array<double, 7> tau_gravity_real_array = left_arm_data.model_handle_->getGravity(gravity_local_array); //change the new gravity vector in lines 128 and 130. They should have opposite sign!
  Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_gravity_internal(tau_gravity_internal_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_gravity_real(tau_gravity_real_array.data());


  Eigen::Matrix<double, 6, 1> error_left;
  error_left.head(3) << position - left_arm_data.position_d_;

  error_left[0]=std::max(-delta_lim, std::min(error_left[0], delta_lim));
  error_left[1]=std::max(-delta_lim, std::min(error_left[1], delta_lim));
  error_left[2]=std::max(-delta_lim, std::min(error_left[2], delta_lim));

  // orientation error
  if (left_arm_data.orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation * left_arm_data.orientation_d_.inverse());
  // convert to axis angle
  Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
  // compute "orientation error"
  error_left.tail(3) << error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();


  error_left[3]=std::max(-delta_lim*3, std::min(error_left[3], delta_lim*3));
  error_left[4]=std::max(-delta_lim*3, std::min(error_left[4], delta_lim*3));
  error_left[5]=std::max(-delta_lim*3, std::min(error_left[5], delta_lim*3));

  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace_left(7), tau_d_left(7), tau_joint_limit(7), null_space_error(7);

  tau_task.setZero();
  tau_d_left.setZero();
  tau_nullspace_left.setZero();
  tau_joint_limit.setZero();
  // pseudoinverse for nullspace handling
  // kinematic pseuoinverse
  null_space_error.setZero();
  null_space_error(0)=(left_arm_data.q_d_nullspace_(0) - q(0));
  null_space_error(1)=(left_arm_data.q_d_nullspace_(1) - q(1));
  null_space_error(2)=(left_arm_data.q_d_nullspace_(2) - q(2));
  null_space_error(3)=(left_arm_data.q_d_nullspace_(3) - q(3));
  null_space_error(4)=(left_arm_data.q_d_nullspace_(4) - q(4));
  null_space_error(5)=(left_arm_data.q_d_nullspace_(5) - q(5));
  null_space_error(6)=(left_arm_data.q_d_nullspace_(6) - q(6));

  // Transform learned forces to robot base frame depending on configured frame
  Eigen::Vector3d force_left_base;
  switch (learned_force_frame_) {
    case LearnedForceFrame::END_EFFECTOR:
      force_left_base = transform.linear() * learned_force_left_;
      break;
    case LearnedForceFrame::WORLD:
      force_left_base = R_O_W_left_ * learned_force_left_;
      break;
    default:  // ROBOT_BASE
      force_left_base = learned_force_left_;
      break;
  }
  Eigen::Matrix<double, 6, 1> learned_wrench_left;
  learned_wrench_left.setZero();
  learned_wrench_left.head(3) << force_left_base;

  // Cartesian PD control with damping ratio = 1 + learned controller forces
  Eigen::Matrix<double, 6, 1> cartesian_wrench_task_left =
      -left_arm_data.cartesian_stiffness_ * error_left -
      left_arm_data.cartesian_damping_ * (jacobian * dq);
  tau_task << jacobian.transpose() * (cartesian_wrench_task_left + learned_wrench_left);

  if (cartesian_left_publish_decimation_counter_++ >= static_cast<uint>(decimation_factor_)) {
    geometry_msgs::WrenchStamped cartesian_wrench_task_left_msg;
    cartesian_wrench_task_left_msg.header.stamp = ros::Time::now();
    cartesian_wrench_task_left_msg.header.frame_id = "base";
    cartesian_wrench_task_left_msg.wrench.force.x = cartesian_wrench_task_left[0];
    cartesian_wrench_task_left_msg.wrench.force.y = cartesian_wrench_task_left[1];
    cartesian_wrench_task_left_msg.wrench.force.z = cartesian_wrench_task_left[2];
    cartesian_wrench_task_left_msg.wrench.torque.x = cartesian_wrench_task_left[3];
    cartesian_wrench_task_left_msg.wrench.torque.y = cartesian_wrench_task_left[4];
    cartesian_wrench_task_left_msg.wrench.torque.z = cartesian_wrench_task_left[5];
    pub_cartesian_wrench_task_left_.publish(cartesian_wrench_task_left_msg);
    cartesian_left_publish_decimation_counter_ = 0;
  }

  // nullspace PD control with damping ratio = 1
  tau_nullspace_left << (Eigen::MatrixXd::Identity(7, 7) -
                    jacobian.transpose() * jacobian_transpose_pinv) *
                       (left_arm_data.nullspace_stiffness_ * null_space_error -
                        (2.0 * sqrt(left_arm_data.nullspace_stiffness_)) * dq);

  //Avoid joint limits
  tau_joint_limit.setZero();

  // (double q_value, double threshold, double magnitude, double upper_bound, double lower_bound)
  tau_joint_limit(0) = calculateTauJointLimit(q(0), 0.05, 4.0, joint_limits[0][1], joint_limits[0][0]);
  tau_joint_limit(1) = calculateTauJointLimit(q(1), 0.05, 4.0, joint_limits[1][1], joint_limits[1][0]);
  tau_joint_limit(2) = calculateTauJointLimit(q(2), 0.05, 4.0, joint_limits[2][1], joint_limits[2][0]);
  tau_joint_limit(3) = calculateTauJointLimit(q(3), 0.05, 4.0, joint_limits[3][1], joint_limits[3][0]);
  tau_joint_limit(4) = calculateTauJointLimit(q(4), 0.05, 4.0, joint_limits[4][1], joint_limits[4][0]);
  tau_joint_limit(5) = calculateTauJointLimit(q(5), 0.05, 4.0, joint_limits[5][1], joint_limits[5][0]);
  tau_joint_limit(6) = calculateTauJointLimit(q(6), 0.05, 4.0, joint_limits[6][1], joint_limits[6][0]);



for (int i = 0; i < 7; ++i) {
    tau_joint_limit(i) = std::max(std::min(tau_joint_limit(i), 5.0), -5.0);
}

  // Desired torque
  tau_d_left << tau_task + tau_nullspace_left + coriolis+ tau_joint_limit;
  // Saturate torque rate to avoid discontinuities
  // tau_d_left << tau_d_left - tau_gravity_internal + tau_gravity_real;
  tau_d_left << saturateTorqueRateLeft(tau_d_left, tau_J_d);
  for (size_t i = 0; i < 7; ++i) {
    left_arm_data.joint_handles_[i].setCommand(tau_d_left(i));
  }
}

Eigen::Matrix<double, 7, 1> BiManualCartesianImpedanceControlWithLearnedController::saturateTorqueRateLeft(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
    auto& left_arm_data = arms_data_.at(left_arm_id_);
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, left_arm_data.delta_tau_max_),
                                               -left_arm_data.delta_tau_max_);
  }
  return tau_d_saturated;
}

void BiManualCartesianImpedanceControlWithLearnedController::updateArmRight() {
  auto& left_arm_data = arms_data_.at(left_arm_id_);
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  // get state variables
  franka::RobotState robot_state_right = right_arm_data.state_handle_->getRobotState();
  // franka::RobotState robot_state_left = left_arm_data.state_handle_->getRobotState();
  std::array<double, 49> inertia_array = right_arm_data.model_handle_->getMass();
  std::array<double, 7> coriolis_array = right_arm_data.model_handle_->getCoriolis();
  std::array<double, 42> jacobian_array = this->get_jacobian(robot_state_right, model_pin_right_,  data_pin_right_, frame_name_right_);
  // std::array<double, 42> jacobian_array_left = this->get_jacobian(robot_state_left, model_pin_left_,  data_pin_left_);
  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  // Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_left(jacobian_array_left.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state_right.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state_right.dq.data());

  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state_right.tau_J_d.data());
  double* O_T_EE = this->get_fk(robot_state_right, model_pin_right_,  data_pin_right_, frame_name_right_);
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(O_T_EE));

  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());
  Eigen::MatrixXd jacobian_transpose_pinv;
  franka_bimanual_controllers::pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);
  // compute error to desired pose

  // position error
  Eigen::Matrix<double, 6, 1> error_right;
  error_right.head(3) << position - right_arm_data.position_d_;

  error_right[0]=std::max(-delta_lim, std::min(error_right[0], delta_lim));
  error_right[1]=std::max(-delta_lim, std::min(error_right[1], delta_lim));
  error_right[2]=std::max(-delta_lim, std::min(error_right[2], delta_lim));

  // find the transformatio in base frame
  Eigen::Affine3d transform_right_base_frame = transform_base_to_right * transform;
  // publish this transformation to the topic
  geometry_msgs::PoseStamped msg_right_global;
  msg_right_global.pose.position.x=transform_right_base_frame.translation()[0];
  msg_right_global.pose.position.y=transform_right_base_frame.translation()[1];
  msg_right_global.pose.position.z=transform_right_base_frame.translation()[2];
  Eigen::Quaterniond orientation_base_to_right(transform_right_base_frame.linear());
  msg_right_global.pose.orientation.x=orientation_base_to_right.x();
  msg_right_global.pose.orientation.y=orientation_base_to_right.y();
  msg_right_global.pose.orientation.z=orientation_base_to_right.z();
  msg_right_global.pose.orientation.w=orientation_base_to_right.w();
  pub_right_global_frame.publish(msg_right_global);

  geometry_msgs::PoseStamped msg_right;
  msg_right.pose.position.x=position[0];
  msg_right.pose.position.y=position[1];
  msg_right.pose.position.z=position[2];

  msg_right.pose.orientation.x=orientation.x();
  msg_right.pose.orientation.y=orientation.y();
  msg_right.pose.orientation.z=orientation.z();
  msg_right.pose.orientation.w=orientation.w();
  pub_right.publish(msg_right);

  Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_ext(robot_state_right.tau_ext_hat_filtered.data());
  Eigen::Matrix<double, 7, 1>  tau_f;

  // Compute the value of the friction
  tau_f(0) =  FI_11/(1+exp(-FI_21*(dq(0)+FI_31))) - TAU_F_CONST_1;
  tau_f(1) =  FI_12/(1+exp(-FI_22*(dq(1)+FI_32))) - TAU_F_CONST_2;
  tau_f(2) =  FI_13/(1+exp(-FI_23*(dq(2)+FI_33))) - TAU_F_CONST_3;
  tau_f(3) =  FI_14/(1+exp(-FI_24*(dq(3)+FI_34))) - TAU_F_CONST_4;
  tau_f(4) =  FI_15/(1+exp(-FI_25*(dq(4)+FI_35))) - TAU_F_CONST_5;
  tau_f(5) =  FI_16/(1+exp(-FI_26*(dq(5)+FI_36))) - TAU_F_CONST_6;
  tau_f(6) =  FI_17/(1+exp(-FI_27*(dq(6)+FI_37))) - TAU_F_CONST_7;

  float iCutOffFrequency=10.0;
  right_arm_data.force_torque+=(-jacobian_transpose_pinv*(tau_ext-tau_f)-right_arm_data.force_torque)*(1-exp(-0.001 * 2.0 * M_PI * iCutOffFrequency));
  geometry_msgs::WrenchStamped force_torque_msg;
  force_torque_msg.wrench.force.x=right_arm_data.force_torque[0];
  force_torque_msg.wrench.force.y=right_arm_data.force_torque[1];
  force_torque_msg.wrench.force.z=right_arm_data.force_torque[2];
  force_torque_msg.wrench.torque.x=right_arm_data.force_torque[3];
  force_torque_msg.wrench.torque.y=right_arm_data.force_torque[4];
  force_torque_msg.wrench.torque.z=right_arm_data.force_torque[5];
  pub_force_torque_right.publish(force_torque_msg);

  Eigen::Vector3d gravity_global(0., 0.,-9.81);
  std::array<double, 3> gravity_global_array = {gravity_global[0], gravity_global[1], gravity_global[2]};
  Eigen::Vector3d gravity_local = transform_right_to_base * gravity_global;
  std::array<double, 3> gravity_local_array = {gravity_local[0], gravity_local[1], gravity_local[2]};
  std::array<double, 7> tau_gravity_internal_array = right_arm_data.model_handle_->getGravity(gravity_global_array);
  std::array<double, 7> tau_gravity_real_array = right_arm_data.model_handle_->getGravity(gravity_local_array); //change the new gravity vector in lines 128 and 130. They should have opposite sign!
  Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_gravity_internal(tau_gravity_internal_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_gravity_real(tau_gravity_real_array.data());

  // orientation error
  if (right_arm_data.orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation * right_arm_data.orientation_d_.inverse());
  // convert to axis angle
  Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
  // compute "orientation error"
  error_right.tail(3) << error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();

  error_right[3]=std::max(-delta_lim*3, std::min(error_right[3], delta_lim*3));
  error_right[4]=std::max(-delta_lim*3, std::min(error_right[4], delta_lim*3));
  error_right[5]=std::max(-delta_lim*3, std::min(error_right[5], delta_lim*3));

  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace_right(7), tau_d(7), tau_joint_limit(7), null_space_error(7);

  null_space_error.setZero();
  null_space_error(0)=(right_arm_data.q_d_nullspace_(0) - q(0));
  null_space_error(1)=(right_arm_data.q_d_nullspace_(1) - q(1));
  null_space_error(2)=(right_arm_data.q_d_nullspace_(2) - q(2));
  null_space_error(3)=(right_arm_data.q_d_nullspace_(3) - q(3));
  null_space_error(4)=(right_arm_data.q_d_nullspace_(4) - q(4));
  null_space_error(5)=(right_arm_data.q_d_nullspace_(5) - q(5));
  null_space_error(6)=(right_arm_data.q_d_nullspace_(6) - q(6));

  // Transform learned forces to robot base frame depending on configured frame
  Eigen::Vector3d force_right_base;
  switch (learned_force_frame_) {
    case LearnedForceFrame::END_EFFECTOR:
      force_right_base = transform.linear() * learned_force_right_;
      break;
    case LearnedForceFrame::WORLD:
      force_right_base = R_O_W_right_ * learned_force_right_;
      break;
    default:  // ROBOT_BASE
      force_right_base = learned_force_right_;
      break;
  }
  Eigen::Matrix<double, 6, 1> learned_wrench_right;
  learned_wrench_right.setZero();
  learned_wrench_right.head(3) << force_right_base;

  // Cartesian PD control with damping ratio = 1 + learned controller forces
  Eigen::Matrix<double, 6, 1> cartesian_wrench_task_right =
      -right_arm_data.cartesian_stiffness_ * error_right -
      right_arm_data.cartesian_damping_ * (jacobian * dq);
  tau_task << jacobian.transpose() * (cartesian_wrench_task_right + learned_wrench_right);

  if (cartesian_right_publish_decimation_counter_++ >= static_cast<uint>(decimation_factor_)) {
    geometry_msgs::WrenchStamped cartesian_wrench_task_right_msg;
    cartesian_wrench_task_right_msg.header.stamp = ros::Time::now();
    cartesian_wrench_task_right_msg.header.frame_id = "base";
    cartesian_wrench_task_right_msg.wrench.force.x = cartesian_wrench_task_right[0];
    cartesian_wrench_task_right_msg.wrench.force.y = cartesian_wrench_task_right[1];
    cartesian_wrench_task_right_msg.wrench.force.z = cartesian_wrench_task_right[2];
    cartesian_wrench_task_right_msg.wrench.torque.x = cartesian_wrench_task_right[3];
    cartesian_wrench_task_right_msg.wrench.torque.y = cartesian_wrench_task_right[4];
    cartesian_wrench_task_right_msg.wrench.torque.z = cartesian_wrench_task_right[5];
    pub_cartesian_wrench_task_right_.publish(cartesian_wrench_task_right_msg);
    cartesian_right_publish_decimation_counter_ = 0;
  }

  // nullspace PD control with damping ratio = 1
  tau_nullspace_right << (Eigen::MatrixXd::Identity(7, 7) -
                    jacobian.transpose() * jacobian_transpose_pinv) *
                       (right_arm_data.nullspace_stiffness_ * null_space_error -
                        (2.0 * sqrt(right_arm_data.nullspace_stiffness_)) * dq);

  //Avoid joint limits
  tau_joint_limit.setZero(); // the comment on the right side is the joint limit reported by
  // (double q_value, double threshold, double magnitude, double upper_bound, double lower_bound)
  tau_joint_limit(0) = calculateTauJointLimit(q(0), 0.05, 4.0, joint_limits[0][1], joint_limits[0][0]);
  tau_joint_limit(1) = calculateTauJointLimit(q(1), 0.05, 4.0, joint_limits[1][1], joint_limits[1][0]);
  tau_joint_limit(2) = calculateTauJointLimit(q(2), 0.05, 4.0, joint_limits[2][1], joint_limits[2][0]);
  tau_joint_limit(3) = calculateTauJointLimit(q(3), 0.05, 4.0, joint_limits[3][1], joint_limits[3][0]);
  tau_joint_limit(4) = calculateTauJointLimit(q(4), 0.05, 4.0, joint_limits[4][1], joint_limits[4][0]);
  tau_joint_limit(5) = calculateTauJointLimit(q(5), 0.05, 4.0, joint_limits[5][1], joint_limits[5][0]);
  tau_joint_limit(6) = calculateTauJointLimit(q(6), 0.05, 4.0, joint_limits[6][1], joint_limits[6][0]);



for (int i = 0; i < 7; ++i) {
    tau_joint_limit(i) = std::max(std::min(tau_joint_limit(i), 5.0), -5.0);
}


  // Desired torque
  tau_d << tau_task + tau_nullspace_right + coriolis+tau_joint_limit;
  // Saturate torque rate to avoid discontinuities
  // tau_d << tau_d - tau_gravity_internal + tau_gravity_real;
  tau_d << saturateTorqueRateRight(tau_d, tau_J_d);

  for (size_t i = 0; i < 7; ++i) {
    right_arm_data.joint_handles_[i].setCommand(tau_d(i));
  }
}

Eigen::Matrix<double, 7, 1> BiManualCartesianImpedanceControlWithLearnedController::saturateTorqueRateRight(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, right_arm_data.delta_tau_max_),
                                               -right_arm_data.delta_tau_max_);
  }
  return tau_d_saturated;
    }


void BiManualCartesianImpedanceControlWithLearnedController::complianceParamCallback(
    franka_combined_bimanual_controllers::dual_arm_compliance_paramConfig& config,
    uint32_t /*level*/) {

   auto& left_arm_data = arms_data_.at(left_arm_id_);
   delta_lim=config.delta_lim;
   left_arm_data.cartesian_stiffness_.setIdentity();
   left_arm_data.cartesian_stiffness_(0,0)=config.panda_left_translational_stiffness_X;
   left_arm_data.cartesian_stiffness_(1,1)=config.panda_left_translational_stiffness_Y;
   left_arm_data.cartesian_stiffness_(2,2)=config.panda_left_translational_stiffness_Z;
   left_arm_data.cartesian_stiffness_(3,3)=config.panda_left_rotational_stiffness_X;
   left_arm_data.cartesian_stiffness_(4,4)=config.panda_left_rotational_stiffness_Y;
   left_arm_data.cartesian_stiffness_(5,5)=config.panda_left_rotational_stiffness_Z;

  left_arm_data.cartesian_damping_(0,0)=2.0 * sqrt(config.panda_left_translational_stiffness_X)*config.panda_left_damping_ratio;
  left_arm_data.cartesian_damping_(1,1)=2.0 * sqrt(config.panda_left_translational_stiffness_Y)*config.panda_left_damping_ratio;
  left_arm_data.cartesian_damping_(2,2)=2.0 * sqrt(config.panda_left_translational_stiffness_Z)*config.panda_left_damping_ratio;
  left_arm_data.cartesian_damping_(3,3)=2.0 * sqrt(config.panda_left_rotational_stiffness_X)*config.panda_left_damping_ratio;
  left_arm_data.cartesian_damping_(4,4)=2.0 * sqrt(config.panda_left_rotational_stiffness_Y)*config.panda_left_damping_ratio;
  left_arm_data.cartesian_damping_(5,5)=2.0 * sqrt(config.panda_left_rotational_stiffness_Z)*config.panda_left_damping_ratio;

  Eigen::AngleAxisd rollAngle_left(config.panda_left_stiffness_roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd yawAngle_left(config.panda_left_stiffness_yaw, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd pitchAngle_left(config.panda_left_stiffness_pitch, Eigen::Vector3d::UnitY());
  Eigen::Quaternion<double> q_left = rollAngle_left *  pitchAngle_left * yawAngle_left;
  Eigen::Matrix3d rotationMatrix_left = q_left.matrix();
  Eigen::Matrix3d rotationMatrix_transpose_left= rotationMatrix_left.transpose();
  left_arm_data.cartesian_stiffness_.topLeftCorner(3, 3) << rotationMatrix_left*left_arm_data.cartesian_stiffness_.topLeftCorner(3, 3)*rotationMatrix_transpose_left;
  left_arm_data.cartesian_stiffness_.bottomRightCorner(3, 3) << rotationMatrix_left*left_arm_data.cartesian_stiffness_.bottomRightCorner(3, 3)*rotationMatrix_transpose_left;


  left_arm_data.nullspace_stiffness_ = config.panda_left_nullspace_stiffness;


  auto& right_arm_data = arms_data_.at(right_arm_id_);

  right_arm_data.cartesian_stiffness_.setIdentity();

  right_arm_data.cartesian_stiffness_(0,0)=config.panda_right_translational_stiffness_X;
  right_arm_data.cartesian_stiffness_(1,1)=config.panda_right_translational_stiffness_Y;
  right_arm_data.cartesian_stiffness_(2,2)=config.panda_right_translational_stiffness_Z;
  right_arm_data.cartesian_stiffness_(3,3)=config.panda_right_rotational_stiffness_X;
  right_arm_data.cartesian_stiffness_(4,4)=config.panda_right_rotational_stiffness_Y;
  right_arm_data.cartesian_stiffness_(5,5)=config.panda_right_rotational_stiffness_Z;

  right_arm_data.cartesian_damping_(0,0)=2.0 * sqrt(config.panda_right_translational_stiffness_X)*config.panda_right_damping_ratio;
  right_arm_data.cartesian_damping_(1,1)=2.0 * sqrt(config.panda_right_translational_stiffness_Y)*config.panda_right_damping_ratio;
  right_arm_data.cartesian_damping_(2,2)=2.0 * sqrt(config.panda_right_translational_stiffness_Z)*config.panda_right_damping_ratio;
  right_arm_data.cartesian_damping_(3,3)=2.0 * sqrt(config.panda_right_rotational_stiffness_X)*config.panda_right_damping_ratio;
  right_arm_data.cartesian_damping_(4,4)=2.0 * sqrt(config.panda_right_rotational_stiffness_Y)*config.panda_right_damping_ratio;
  right_arm_data.cartesian_damping_(5,5)=2.0 * sqrt(config.panda_right_rotational_stiffness_Z)*config.panda_right_damping_ratio;

  Eigen::AngleAxisd rollAngle_right(config.panda_right_stiffness_roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd yawAngle_right(config.panda_right_stiffness_yaw, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd pitchAngle_right(config.panda_right_stiffness_pitch, Eigen::Vector3d::UnitY());
  Eigen::Quaternion<double> q_right = rollAngle_right *  pitchAngle_right * yawAngle_right;
  Eigen::Matrix3d rotationMatrix_right = q_right.matrix();
  Eigen::Matrix3d rotationMatrix_transpose_right= rotationMatrix_right.transpose();
  right_arm_data.cartesian_stiffness_.topLeftCorner(3, 3) << rotationMatrix_right*right_arm_data.cartesian_stiffness_.topLeftCorner(3, 3)*rotationMatrix_transpose_right;
  right_arm_data.cartesian_stiffness_.bottomRightCorner(3, 3) << rotationMatrix_right*right_arm_data.cartesian_stiffness_.bottomRightCorner(3, 3)*rotationMatrix_transpose_right;

  right_arm_data.nullspace_stiffness_ = config.panda_right_nullspace_stiffness;

}


double BiManualCartesianImpedanceControlWithLearnedController::calculateTauJointLimit(double q_value, double threshold, double magnitude, double upper_bound, double lower_bound) {
    double upper_limit = upper_bound - threshold;
    double lower_limit = lower_bound + threshold;
    if (q_value > (upper_limit)) {
        return -magnitude * (std::exp( std::abs( q_value - upper_limit )/threshold) - 1);
    } else if (q_value < lower_limit) {
        return +magnitude * (std::exp( std::abs( q_value - lower_limit )/threshold) - 1);
    } else {
        return 0;
    }
}

void BiManualCartesianImpedanceControlWithLearnedController::equilibriumPoseCallback_left(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  auto& left_arm_data = arms_data_.at(left_arm_id_);
  left_arm_data.position_d_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  // Eigen::Quaterniond last_orientation_d_(left_arm_data.orientation_d_);
  left_arm_data.orientation_d_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
}

void BiManualCartesianImpedanceControlWithLearnedController::equilibriumPoseCallback_right(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  right_arm_data.position_d_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  // Eigen::Quaterniond last_orientation_d_(right_arm_data.orientation_d_);
  right_arm_data.orientation_d_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
}

void BiManualCartesianImpedanceControlWithLearnedController::equilibriumPoseCallback_right_global(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  Eigen::Affine3d transform_global;
  transform_global.translation() << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  transform_global.linear() = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x,
                                          msg->pose.orientation.y, msg->pose.orientation.z)
                           .toRotationMatrix();
  Eigen::Affine3d transform_local = transform_right_to_base * transform_global;

  Eigen::Vector3d local_position(transform_local.translation());
  Eigen::Quaterniond local_orientation(transform_local.linear());

  right_arm_data.position_d_ << local_position[0], local_position[1], local_position[2];
  right_arm_data.orientation_d_.coeffs() << local_orientation.x(), local_orientation.y(),
      local_orientation.z(), local_orientation.w();
}

void BiManualCartesianImpedanceControlWithLearnedController::equilibriumPoseCallback_left_global(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  auto& left_arm_data = arms_data_.at(left_arm_id_);
  Eigen::Affine3d transform_global;
  transform_global.translation() << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  transform_global.linear() = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x,
                                          msg->pose.orientation.y, msg->pose.orientation.z)
                           .toRotationMatrix();
  Eigen::Affine3d transform_local = transform_left_to_base * transform_global;

  Eigen::Vector3d local_position(transform_local.translation());
  Eigen::Quaterniond local_orientation(transform_local.linear());

  left_arm_data.position_d_ << local_position[0], local_position[1], local_position[2];
  left_arm_data.orientation_d_.coeffs() << local_orientation.x(), local_orientation.y(),
      local_orientation.z(), local_orientation.w();
    }

void BiManualCartesianImpedanceControlWithLearnedController::equilibriumConfigurationCallback_right(const sensor_msgs::JointState::ConstPtr& joint) {

  auto& right_arm_data = arms_data_.at(right_arm_id_);
  std::vector<double> read_joint_right;
  read_joint_right= joint -> position;
  right_arm_data.q_d_nullspace_(0) = read_joint_right[0];
  right_arm_data.q_d_nullspace_(1) = read_joint_right[1];
  right_arm_data.q_d_nullspace_(2) = read_joint_right[2];
  right_arm_data.q_d_nullspace_(3) = read_joint_right[3];
  right_arm_data.q_d_nullspace_(4) = read_joint_right[4];
  right_arm_data.q_d_nullspace_(5) = read_joint_right[5];
  right_arm_data.q_d_nullspace_(6) = read_joint_right[6];
}

void BiManualCartesianImpedanceControlWithLearnedController::equilibriumConfigurationCallback_left(const sensor_msgs::JointState::ConstPtr& joint) {

  auto& left_arm_data = arms_data_.at(left_arm_id_);
  std::vector<double> read_joint_left;
  read_joint_left= joint -> position;
  left_arm_data.q_d_nullspace_(0) = read_joint_left[0];
  left_arm_data.q_d_nullspace_(1) = read_joint_left[1];
  left_arm_data.q_d_nullspace_(2) = read_joint_left[2];
  left_arm_data.q_d_nullspace_(3) = read_joint_left[3];
  left_arm_data.q_d_nullspace_(4) = read_joint_left[4];
  left_arm_data.q_d_nullspace_(5) = read_joint_left[5];
  left_arm_data.q_d_nullspace_(6) = read_joint_left[6];

}
}  // namespace franka_bimanual_controllers

PLUGINLIB_EXPORT_CLASS(franka_bimanual_controllers::BiManualCartesianImpedanceControlWithLearnedController,
                       controller_interface::ControllerBase)
