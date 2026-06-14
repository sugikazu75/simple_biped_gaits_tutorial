#include "simple_biped_gaits_tutorial/biped_controller_interface.hpp"

int main(int argc, char **argv) {
  ros::init(argc, argv, "simple_biped_gaits_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  auto controller = std::make_shared<BipedControllerInterface>(nh, pnh);
  double rate_hz = 1.0 / controller->getTimeStep();
  ros::Rate rate(rate_hz);
  while (ros::ok()) {
    ros::spinOnce();
    controller->update();
    rate.sleep();
  }

  return 0;
}
