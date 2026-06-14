#include "simple_biped_gaits_tutorial/biped_controller_core.hpp"

#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/core/activations/weighted-quadratic-barrier.hpp>
#include <crocoddyl/core/activations/weighted-quadratic.hpp>

#include <iostream>
#include <stdexcept>

BipedControllerCore::BipedControllerCore(const Config& config) : config_(config)
{
  initializeModelFromUrdf();
  setupGaitModel();
}

void BipedControllerCore::initializeModelFromUrdf()
{
  model_ = std::make_shared<pinocchio::Model>();
  pinocchio::urdf::buildModelFromXML(config_.robot_description, pinocchio::JointModelFreeFlyer(), *model_);
  data_ = std::make_shared<pinocchio::Data>(*model_);
  printModelInfo();
}

void BipedControllerCore::setupGaitModel()
{
  const auto foot_frame_type =
      (pinocchio::FrameType)(pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY);
  left_foot_id_ = model_->getFrameId(config_.lleg, foot_frame_type);
  right_foot_id_ = model_->getFrameId(config_.rleg, foot_frame_type);
  state_ = std::make_shared<crocoddyl::StateMultibody>(model_);
  actuation_ = std::make_shared<crocoddyl::ActuationModelFloatingBase>(state_);
  if (config_.cost_weight.x_weights.size() != 2 * model_->nv)
    config_.cost_weight.x_weights = Eigen::VectorXd::Ones(2 * model_->nv);
}

void BipedControllerCore::printModelInfo() const
{
  for (int i = 0; i < model_->njoints; i++)
  {
    std::string joint_type = model_->joints[i].shortname();
    std::cout << model_->names[i] << " " << joint_type << " "
              << model_->joints[model_->getJointId(model_->names[i])].idx_q() << std::endl;
  }
  std::cout << "nq: " << model_->nq << ", nv: " << model_->nv << ", njoints: " << model_->njoints
            << ", nframes: " << model_->nframes << std::endl;
}

void BipedControllerCore::createGait()
{
  config_.printCostWeight();
  config_.printConfig();

  Eigen::VectorXd q0 = Eigen::VectorXd::Zero(model_->nq);
  if (config_.initial_configuration.size() == model_->nq)
    q0 = Eigen::Map<const Eigen::VectorXd>(config_.initial_configuration.data(), config_.initial_configuration.size());
  else
  {
    std::cerr << "initial_configuration size: " << config_.initial_configuration.size()
              << " is not correct. expected size: " << model_->nq << std::endl;
  }

  pinocchio::forwardKinematics(*model_, *data_, q0);
  pinocchio::updateFramePlacements(*model_, *data_);
  Eigen::Vector3d left_foot_pos = data_->oMf[left_foot_id_].translation();
  Eigen::Vector3d right_foot_pos = data_->oMf[right_foot_id_].translation();
  q0.segment<3>(0) -= 0.5 * (left_foot_pos + right_foot_pos);
  std::cout << "adjusted root position: " << q0.segment<3>(0).transpose() << std::endl;

  x0_ = Eigen::VectorXd::Zero(model_->nq + model_->nv);
  x0_.head(model_->nq) = q0;
  std::cout << "initial state: " << x0_.transpose() << std::endl;

  q0_ = x0_.head(model_->nq);
}

void BipedControllerCore::createSolver()
{
  switch (config_.solver_type)
  {
    case SolverType::FDDP: {
      solver_ = std::make_shared<crocoddyl::SolverFDDP>(walking_problem_);
      break;
    }
    case SolverType::BOX_FDDP: {
      solver_ = std::make_shared<crocoddyl::SolverBoxFDDP>(walking_problem_);
      break;
    }
    case SolverType::INTRO: {
      solver_ = std::make_shared<crocoddyl::SolverIntro>(walking_problem_);
      break;
    }
  case SolverType::HPIPM_SQP:{
    solver_ = std::make_shared<crocoddyl::SolverHpipmSQP>(walking_problem_);
    break;
  }
    default: {
      std::cerr << "Invalid solver type: " << (int)config_.solver_type << std::endl;
      break;
    }
  }

  createInitialGuess();

  solver_->get_problem()->set_nthreads(config_.num_threads);
  std::cout << solver_->get_problem()->get_T() << " nodes in the walking problem." << std::endl;
}

void BipedControllerCore::createInitialGuess()
{
  xs_.clear();
  us_.clear();
  for (size_t i = 0; i < walking_problem_->get_T(); ++i)
    xs_.push_back(x0_);
  us_ = walking_problem_->quasiStatic_xs(xs_);
  xs_.push_back(x0_);
}

void BipedControllerCore::createPlanningProblem()
{
  walking_problem_ = createWalkingProblem(x0_, config_.step_length, config_.step_height, config_.time_step,
                                          config_.step_knots, config_.support_knots);
}

void BipedControllerCore::solvePlanningProblem()
{
  solveProblem(config_.max_iter, true);
}

void BipedControllerCore::solveProblem(int max_iter, bool verbose, const std::string& label)
{
  crocoddyl::Timer timer;
  bool solved = solver_->solve(xs_, us_, max_iter, false);
  last_solve_time_ = timer.get_duration();

  if (verbose)
  {
    std::cout << label << " solved: " << solved << std::endl;
    std::cout << "total calculation time:" << last_solve_time_ << std::endl;
    std::cout << "Number of iterations: " << solver_->get_iter() << std::endl;
    std::cout << "time per iterate:" << last_solve_time_ / solver_->get_iter() << std::endl;
    std::cout << "Total cost: " << solver_->get_cost() << std::endl;
    std::cout << "Gradient norm: " << solver_->get_stop() << std::endl;
    std::cout << std::endl;
  }

  xs_ = solver_->get_xs();
  us_ = solver_->get_us();
}

std::shared_ptr<crocoddyl::ActuationDataAbstract> BipedControllerCore::getActuationData(int idx)
{
  std::shared_ptr<crocoddyl::IntegratedActionModelEuler> action_model =
      std::dynamic_pointer_cast<crocoddyl::IntegratedActionModelEuler>(walking_problem_->get_runningModels()[idx]);
  std::shared_ptr<crocoddyl::IntegratedActionDataEuler> action_data =
      std::dynamic_pointer_cast<crocoddyl::IntegratedActionDataEuler>(walking_problem_->get_runningDatas()[idx]);
  std::shared_ptr<crocoddyl::ActuationDataAbstract> actuation_data;
  if (config_.fwddyn)
  {
    std::shared_ptr<crocoddyl::DifferentialActionDataContactFwdDynamics> ddata =
        std::dynamic_pointer_cast<crocoddyl::DifferentialActionDataContactFwdDynamics>(action_data->differential);
    crocoddyl::DataCollectorJointActMultibodyInContact* multibody = &ddata->multibody;
    actuation_data = multibody->actuation;
  }
  else
  {
    std::shared_ptr<crocoddyl::DifferentialActionDataContactInvDynamics> ddata =
        std::dynamic_pointer_cast<crocoddyl::DifferentialActionDataContactInvDynamics>(action_data->differential);
    crocoddyl::DataCollectorJointActMultibodyInContact* multibody = &ddata->multibody;
    actuation_data = multibody->actuation;
  }
  return actuation_data;
}

std::pair<std::shared_ptr<crocoddyl::ContactModelMultiple>, std::shared_ptr<crocoddyl::ContactDataMultiple>>
BipedControllerCore::getContactModelAndData(int idx)
{
  std::shared_ptr<crocoddyl::IntegratedActionModelEuler> action_model =
      std::dynamic_pointer_cast<crocoddyl::IntegratedActionModelEuler>(walking_problem_->get_runningModels()[idx]);
  std::shared_ptr<crocoddyl::IntegratedActionDataEuler> action_data =
      std::dynamic_pointer_cast<crocoddyl::IntegratedActionDataEuler>(walking_problem_->get_runningDatas()[idx]);

  std::shared_ptr<crocoddyl::ContactModelMultiple> contacts_model;
  std::shared_ptr<crocoddyl::ContactDataMultiple> contacts_data;

  if (config_.fwddyn)
  {
    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> dmodel =
        std::dynamic_pointer_cast<crocoddyl::DifferentialActionModelContactFwdDynamics>(
            action_model->get_differential());
    std::shared_ptr<crocoddyl::DifferentialActionDataContactFwdDynamics> ddata =
        std::dynamic_pointer_cast<crocoddyl::DifferentialActionDataContactFwdDynamics>(action_data->differential);
    contacts_model = dmodel->get_contacts();
    crocoddyl::DataCollectorJointActMultibodyInContact* multibody = &ddata->multibody;
    contacts_data = multibody->contacts;
  }
  else
  {
    std::shared_ptr<crocoddyl::DifferentialActionModelContactInvDynamics> dmodel =
        std::dynamic_pointer_cast<crocoddyl::DifferentialActionModelContactInvDynamics>(
            action_model->get_differential());
    std::shared_ptr<crocoddyl::DifferentialActionDataContactInvDynamics> ddata =
        std::dynamic_pointer_cast<crocoddyl::DifferentialActionDataContactInvDynamics>(action_data->differential);
    contacts_model = dmodel->get_contacts();
    crocoddyl::DataCollectorJointActMultibodyInContact* multibody = &ddata->multibody;
    contacts_data = multibody->contacts;
  }

  return std::make_pair(contacts_model, contacts_data);
}

std::shared_ptr<crocoddyl::ShootingProblem>
BipedControllerCore::createWalkingProblem(const Eigen::VectorXd& x0, const double stepLength, const double stepHeight,
                                          const double timeStep, const std::size_t stepKnots,
                                          const std::size_t supportKnots)
{
  q0_ = x0.head(model_->nq);

  pinocchio::forwardKinematics(*model_, *data_, q0_);
  pinocchio::updateFramePlacements(*model_, *data_);

  Eigen::Vector3d rfPos0 = data_->oMf[right_foot_id_].translation();
  Eigen::Vector3d lfPos0 = data_->oMf[left_foot_id_].translation();
  std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>> feetPos0;
  feetPos0.push_back(std::make_pair(right_foot_id_, rfPos0));
  feetPos0.push_back(std::make_pair(left_foot_id_, lfPos0));
  Eigen::Vector3d comRef = (rfPos0 + lfPos0) / 2.0;
  comRef[2] = pinocchio::centerOfMass(*model_, *data_, q0_)[2];

  std::vector<pinocchio::FrameIndex> lf_ids;
  lf_ids.push_back(left_foot_id_);
  std::vector<pinocchio::FrameIndex> rf_ids;
  rf_ids.push_back(right_foot_id_);

  std::vector<pinocchio::FrameIndex> rf_lf_ids;
  rf_lf_ids.push_back(right_foot_id_);
  rf_lf_ids.push_back(left_foot_id_);

  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> loco3dModel;

  // initial double support phase
  for (size_t i = 0; i < supportKnots; i++)
    loco3dModel.push_back(createSwingFootModel(timeStep, rf_lf_ids, comRef));

  // walking steps
  for (int i = 1; i <= config_.num_steps; i++)
  {
    if (i % 2 == 1)  // right step
    {
      std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> rightStepModels;
      if (first_step_)
      {
        rightStepModels =
            createFootStepModels(timeStep, comRef, feetPos0, 0.5 * stepLength, stepHeight, stepKnots, lf_ids, rf_ids);
        first_step_ = false;
      }
      else
      {
        rightStepModels =
            createFootStepModels(timeStep, comRef, feetPos0, stepLength, stepHeight, stepKnots, lf_ids, rf_ids);
      }
      loco3dModel.insert(loco3dModel.end(), rightStepModels.begin(), rightStepModels.end());
    }
    else  // left step
    {
      std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> leftStepModels =
          createFootStepModels(timeStep, comRef, feetPos0, stepLength, stepHeight, stepKnots, rf_ids, lf_ids);
      loco3dModel.insert(loco3dModel.end(), leftStepModels.begin(), leftStepModels.end());
    }
    // double support
    loco3dModel.push_back(createSwingFootModel(timeStep, rf_lf_ids, comRef, feetPos0));
  }

  // final half step
  if (config_.num_steps > 0)
  {
    if (config_.num_steps % 2 == 0)  // ended with left step. add half right step
    {
      std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> rightStepModels =
          createFootStepModels(timeStep, comRef, feetPos0, 0.5 * stepLength, stepHeight, stepKnots, lf_ids, rf_ids);
      loco3dModel.insert(loco3dModel.end(), rightStepModels.begin(), rightStepModels.end());
    }
    else  // ended with right step -> move left half-step
    {
      std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> leftStepModels =
          createFootStepModels(timeStep, comRef, feetPos0, 0.5 * stepLength, stepHeight, stepKnots, rf_ids, lf_ids);
      loco3dModel.insert(loco3dModel.end(), leftStepModels.begin(), leftStepModels.end());
    }
  }

  // final double support phase
  for (size_t i = 0; i < supportKnots; i++)
    loco3dModel.push_back(createSwingFootModel(timeStep, rf_lf_ids, comRef, feetPos0));

  // terminal state
  std::shared_ptr<crocoddyl::ActionModelAbstract> terminalModel = loco3dModel.back();
  loco3dModel.pop_back();

  return std::make_shared<crocoddyl::ShootingProblem>(x0, loco3dModel, terminalModel);
}

std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> BipedControllerCore::createFootStepModels(
    double timeStep, Eigen::Vector3d& comPos0,
    std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>& feetPos0, const double stepLength,
    const double stepHeight, const std::size_t numKnots, const std::vector<pinocchio::FrameIndex>& supportFootIds,
    const std::vector<pinocchio::FrameIndex>& swingFootIds)
{
  int num_legs = (int)supportFootIds.size() + (int)swingFootIds.size();
  double com_percentage = (double)swingFootIds.size() / (double)num_legs;

  // action models for the foot swing
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> footStepModels;
  std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>> footTask;
  for (size_t k = 0; k < numKnots; k++)  // assume numKnots is odd
  {
    footTask.clear();
    for (size_t i = 0; i < swingFootIds.size(); i++)  // i: swing foot index
    {
      Eigen::Vector3d dp = Eigen::Vector3d::Zero();
      if (k < (numKnots - 1) / 2)
      {
        dp = Eigen::Vector3d(stepLength * ((double)k + 1) / (double)numKnots, 0.0,
                             stepHeight * (double)k / ((double)(numKnots - 1) / 2.0));
      }
      else if (k == (numKnots - 1) / 2)
      {
        dp = Eigen::Vector3d(stepLength * ((double)k + 1) / (double)numKnots, 0.0, stepHeight);
      }
      else
      {
        dp = Eigen::Vector3d(
            stepLength * ((double)k + 1) / (double)numKnots, 0.0,
            stepHeight * (1.0 - ((double)k - (double)(numKnots - 1) / 2.0) / ((double)(numKnots - 1) / 2.0)));
      }

      Eigen::Vector3d footPos = Eigen::Vector3d::Zero();
      for (size_t j = 0; j < feetPos0.size(); ++j)
      {
        if (feetPos0[j].first == swingFootIds[i])  // get only swing foot
        {
          footPos = feetPos0[j].second;
          break;
        }
      }
      footTask.push_back(std::make_pair(swingFootIds[i], footPos + dp));
    }
    Eigen::Vector3d comTask =
        Eigen::Vector3d(stepLength * ((double)(k + 1) / (double)numKnots), 0.0, 0.0) * com_percentage + comPos0;
    footStepModels.push_back(createSwingFootModel(timeStep, supportFootIds, comTask, footTask));
  }

  // updating the current foot position for next step
  comPos0 += Eigen::Vector3d(stepLength * com_percentage, 0., 0.);
  for (size_t i = 0; i < swingFootIds.size(); i++)
    for (size_t j = 0; j < feetPos0.size(); ++j)
    {
      if (feetPos0[j].first == swingFootIds[i])
      {
        feetPos0[j].second += Eigen::Vector3d(stepLength, 0., 0.);
        break;
      }
    }

  return footStepModels;
}

std::shared_ptr<crocoddyl::ActionModelAbstract> BipedControllerCore::createSwingFootModel(
    double timeStep, const std::vector<pinocchio::FrameIndex>& supportFootIds, const Eigen::Vector3d& comTask,
    const std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>& swingFootTask)
{
  size_t nu;
  if (config_.fwddyn)
    nu = actuation_->get_nu();
  else
    nu = state_->get_nv() + 6 * supportFootIds.size();

  // Creating a 6D multi-contact model at the supporting foot
  std::shared_ptr<crocoddyl::ContactModelMultiple> contactModel =
      std::make_shared<crocoddyl::ContactModelMultiple>(state_, nu);
  for (size_t i = 0; i < supportFootIds.size(); i++)
  {
    std::shared_ptr<crocoddyl::ContactModelAbstract> supportContactModel =
        std::make_shared<crocoddyl::ContactModel6D>(state_, supportFootIds[i], pinocchio::SE3::Identity(),
                                                    pinocchio::LOCAL_WORLD_ALIGNED, nu, Eigen::Vector2d(0., 50.));
    contactModel->addContact(model_->frames[supportFootIds[i]].name + "_contact", supportContactModel);
  }

  // creating the cost model for a contact phase
  std::shared_ptr<crocoddyl::CostModelSum> costModel = std::make_shared<crocoddyl::CostModelSum>(state_, nu);

  // Com tracking cost
  if (comTask != Eigen::Vector3d::Zero())
  {
    std::shared_ptr<crocoddyl::ResidualModelAbstract> comResidual =
        std::make_shared<crocoddyl::ResidualModelCoMPosition>(state_, comTask, nu);
    std::shared_ptr<crocoddyl::CostModelAbstract> comTrack =
        std::make_shared<crocoddyl::CostModelResidual>(state_, comResidual);
    costModel->addCost("comTrack", comTrack, config_.cost_weight.com_track_weight);
  }

  // supporting foot contact wrench cone cost
  for (size_t i = 0; i < supportFootIds.size(); ++i)
  {
    Eigen::Matrix3d Rsurf = Eigen::Matrix3d::Identity();
    crocoddyl::WrenchCone cone(Rsurf, 0.7, Eigen::Vector2d(0.1, 0.05));

    std::shared_ptr<crocoddyl::ResidualModelAbstract> wrench_residual =
        std::make_shared<crocoddyl::ResidualModelContactWrenchCone>(state_, supportFootIds[i], cone, nu,
                                                                    config_.fwddyn);

    std::shared_ptr<crocoddyl::ActivationModelAbstract> wrench_activation =
        std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(
            crocoddyl::ActivationBounds(cone.get_lb(), cone.get_ub()));

    std::shared_ptr<crocoddyl::CostModelAbstract> wrench_cone =
        std::make_shared<crocoddyl::CostModelResidual>(state_, wrench_activation, wrench_residual);

    costModel->addCost(model_->frames[supportFootIds[i]].name + "_wrenchCone", wrench_cone,
                       config_.cost_weight.contact_wrench_weight);
  }

  // swing foot tracking cost
  if (!swingFootTask.empty())
  {
    for (size_t i = 0; i < swingFootTask.size(); i++)
    {
      pinocchio::FrameIndex id = swingFootTask[i].first;
      Eigen::Vector3d posTask = swingFootTask[i].second;

      std::shared_ptr<crocoddyl::ResidualModelAbstract> foot_placement_residual =
          std::make_shared<crocoddyl::ResidualModelFramePlacement>(
              state_, id, pinocchio::SE3(Eigen::Matrix3d::Identity(), posTask), nu);

      std::shared_ptr<crocoddyl::CostModelAbstract> foot_track =
          std::make_shared<crocoddyl::CostModelResidual>(state_, foot_placement_residual);

      costModel->addCost(model_->frames[id].name + "_footTrack", foot_track, config_.cost_weight.foot_track_weight);
    }
  }

  // state regularization cost
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(model_->nq + model_->nv);
  x0 << q0_, Eigen::VectorXd::Zero(model_->nv);
  std::shared_ptr<crocoddyl::ResidualModelAbstract> stateResidual =
      std::make_shared<crocoddyl::ResidualModelState>(state_, x0, nu);
  std::shared_ptr<crocoddyl::ActivationModelAbstract> stateActivation =
      std::make_shared<crocoddyl::ActivationModelWeightedQuad>(config_.cost_weight.x_weights);
  std::shared_ptr<crocoddyl::CostModelAbstract> stateReg =
      std::make_shared<crocoddyl::CostModelResidual>(state_, stateActivation, stateResidual);

  // control regularization cost
  std::shared_ptr<crocoddyl::CostModelAbstract> ctrlReg;
  if (config_.fwddyn)
  {
    std::shared_ptr<crocoddyl::ResidualModelAbstract> ctrlResidual =
        std::make_shared<crocoddyl::ResidualModelControl>(state_, nu);
    ctrlReg = std::make_shared<crocoddyl::CostModelResidual>(state_, ctrlResidual);
  }
  else
  {
    std::shared_ptr<crocoddyl::ResidualModelAbstract> ctrlResidual =
        std::make_shared<crocoddyl::ResidualModelJointEffort>(state_, actuation_, nu);
    ctrlReg = std::make_shared<crocoddyl::CostModelResidual>(state_, ctrlResidual);
  }

  costModel->addCost("stateReg", stateReg, 1.0);
  costModel->addCost("ctrlReg", ctrlReg, config_.cost_weight.control_weight);

  // Creating the action model for the KKT dynamics with symplectic Euler
  // integration scheme
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> dmodel;
  if (config_.fwddyn)
  {
    dmodel = std::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state_, actuation_, contactModel,
                                                                                    costModel, 0.0, true);
  }
  else
  {
    dmodel = std::make_shared<crocoddyl::DifferentialActionModelContactInvDynamics>(state_, actuation_, contactModel,
                                                                                    costModel);
  }

  return std::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel, timeStep);
}
