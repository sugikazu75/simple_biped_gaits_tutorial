#pragma once

#include <pinocchio/fwd.hpp>  // should be included before any other pinocchio headers
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/frame.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/core/activations/weighted-quadratic.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/fwd.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/core/residuals/joint-effort.hpp>
#include <crocoddyl/core/solvers/box-fddp.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>
#include <crocoddyl/core/solvers/intro.hpp>
#include <crocoddyl/core/solvers/hpipm-sqp.hpp>
#include <crocoddyl/core/utils/timer.hpp>
#include <crocoddyl/multibody/actions/contact-fwddyn.hpp>
#include <crocoddyl/multibody/actions/contact-invdyn.hpp>
#include <crocoddyl/multibody/actuations/floating-base.hpp>
#include <crocoddyl/multibody/contacts/contact-3d.hpp>
#include <crocoddyl/multibody/contacts/contact-6d.hpp>
#include <crocoddyl/multibody/fwd.hpp>
#include <crocoddyl/multibody/residuals/com-position.hpp>
#include <crocoddyl/multibody/residuals/contact-wrench-cone.hpp>
#include <crocoddyl/multibody/residuals/frame-placement.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/multibody/wrench-cone.hpp>

#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <map>
#include <string>
#include <vector>

class BipedControllerCore
{
public:
  enum SolverType
  {
    FDDP = 0,
    BOX_FDDP,
    INTRO,
    HPIPM_SQP
  };

  struct CostWeight
  {
    Eigen::VectorXd x_weights = Eigen::VectorXd::Zero(0);
    double state_weight = 1e1;
    double control_weight = 1e-1;
    double com_track_weight = 1e6;
    double foot_track_weight = 1e6;
    double contact_wrench_weight = 1e1;
  };

  struct Config
  {
    std::string robot_description;
    double step_length = 0.6;
    double step_height = 0.1;
    double time_step = 0.02;
    int step_knots = 51;
    int support_knots = 10;
    int max_iter = 100;
    int num_steps = 1;
    bool fwddyn = true;
    int num_threads = 1;

    std::string root_link_name;
    std::string lleg;
    std::string rleg;

    SolverType solver_type = SolverType::FDDP;
    std::vector<double> initial_configuration;
    CostWeight cost_weight;

    void printConfig() const
    {
      std::cout << "step_length: " << step_length << std::endl;
      std::cout << "step_height: " << step_height << std::endl;
      std::cout << "time_step: " << time_step << std::endl;
      std::cout << "step_knots: " << step_knots << std::endl;
      std::cout << "support_knots: " << support_knots << std::endl;
      std::cout << "max_iter: " << max_iter << std::endl;
      std::cout << "num_steps: " << num_steps << std::endl;
      std::cout << "fwddyn: " << fwddyn << std::endl;
      std::cout << "num_threads: " << num_threads << std::endl;
      std::cout << "root_link_name: " << root_link_name << std::endl;
      std::cout << "lleg: " << lleg << std::endl;
      std::cout << "rleg: " << rleg << std::endl;
      std::cout << "solver type: "
                << (solver_type == SolverType::FDDP ? "FDDP" :
                                                      (solver_type == SolverType::BOX_FDDP ? "BOX_FDDP" :
                                                       (solver_type == SolverType::INTRO ? "INTRO" : "HPIPM_SQP")))
                << std::endl;
    }

    void printCostWeight() const
    {
      std::cout << "cost weights: " << std::endl;
      std::cout << "x_weights: " << cost_weight.x_weights.transpose() << std::endl;
      std::cout << "state_weight: " << cost_weight.state_weight << std::endl;
      std::cout << "control_weight: " << cost_weight.control_weight << std::endl;
      std::cout << "com_track_weight: " << cost_weight.com_track_weight << std::endl;
      std::cout << "foot_track_weight: " << cost_weight.foot_track_weight << std::endl;
      std::cout << "contact_wrench_weight: " << cost_weight.contact_wrench_weight << std::endl;
    }
  };

  explicit BipedControllerCore(const Config& config);
  ~BipedControllerCore() = default;

  void createGait();
  void createPlanningProblem();
  void createSolver();
  void solvePlanningProblem();

  std::shared_ptr<crocoddyl::ActuationDataAbstract> getActuationData(int idx);
  std::pair<std::shared_ptr<crocoddyl::ContactModelMultiple>, std::shared_ptr<crocoddyl::ContactDataMultiple>> getContactModelAndData(int idx);

  const Config& getConfig() const { return config_;}
  const std::shared_ptr<pinocchio::Model>& getModel() const { return model_;}
  const std::shared_ptr<pinocchio::Data>& getData() const { return data_;}
  const std::shared_ptr<crocoddyl::ShootingProblem>& getWalkingProblem() const { return walking_problem_;}
  const std::shared_ptr<crocoddyl::SolverAbstract>& getSolver() const { return solver_;}
  const std::vector<Eigen::VectorXd>& getXs() const { return xs_;}
  const std::vector<Eigen::VectorXd>& getUs() const { return us_;}
  double getLastSolveTime() const { return last_solve_time_;}

private:
  void initializeModelFromUrdf();
  void setupGaitModel();
  void createInitialGuess();
  void solveProblem(int max_iter, bool verbose, const std::string& label = "Problem");
  void printModelInfo() const;

  std::shared_ptr<crocoddyl::ShootingProblem> createWalkingProblem(const Eigen::VectorXd& x0, const double stepLength,
                                                                   const double stepHeight, const double timeStep,
                                                                   const std::size_t stepKnots,
                                                                   const std::size_t supportKnots);
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>
  createFootStepModels(double timeStep, Eigen::Vector3d& comPos0,
                       std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>& feetPos0,
                       const double stepLength, const double stepHeight, const std::size_t numKnots,
                       const std::vector<pinocchio::FrameIndex>& supportFootIds,
                       const std::vector<pinocchio::FrameIndex>& swingFootIds);
  std::shared_ptr<crocoddyl::ActionModelAbstract> createSwingFootModel(
      double timeStep, const std::vector<pinocchio::FrameIndex>& supportFootIds,
      const Eigen::Vector3d& comTask = Eigen::Vector3d::Zero(),
      const std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>& swingFootTask =
          std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>());

  Config config_;
  std::shared_ptr<pinocchio::Model> model_;
  std::shared_ptr<pinocchio::Data> data_;
  pinocchio::FrameIndex left_foot_id_;
  pinocchio::FrameIndex right_foot_id_;
  std::shared_ptr<crocoddyl::StateMultibody> state_;
  std::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation_;
  std::shared_ptr<crocoddyl::ShootingProblem> walking_problem_;
  std::shared_ptr<crocoddyl::SolverAbstract> solver_;

  Eigen::VectorXd x0_;
  Eigen::VectorXd q0_;
  std::vector<Eigen::VectorXd> xs_;
  std::vector<Eigen::VectorXd> us_;
  double last_solve_time_ = 0.0;
  bool first_step_ = true;
};
