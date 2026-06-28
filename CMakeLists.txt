cmake_minimum_required(VERSION 3.0.2)
project(simple_biped_gaits_tutorial)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  sensor_msgs
  std_msgs
  tf
  tf2_ros
  )

find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(crocoddyl REQUIRED)
find_package(OpenMP REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp sensor_msgs std_msgs tf tf2_ros
  )

option(USE_DOUBLE_PRECISION "Build with double precision" OFF)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  )

add_library(simple_biped_gaits_planner
  src/biped_controller_core.cpp
  src/biped_controller_interface.cpp
  )
target_link_libraries(simple_biped_gaits_planner
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
  pinocchio::pinocchio
  crocoddyl::crocoddyl
  )
if(USE_DOUBLE_PRECISION)
  message(STATUS "Building simple_biped_gaits_planner with DOUBLE precision.")
  target_compile_definitions(simple_biped_gaits_planner PUBLIC USE_DOUBLE_PRECISION)
else()
  message(STATUS "Building simple_biped_gaits_planner with FLOAT precision.")
endif()


add_executable(simple_biped_gaits src/simple_biped_gaits_node.cpp)
target_link_libraries(simple_biped_gaits
  simple_biped_gaits_planner
  ${catkin_LIBRARIES}
  )

install(TARGETS simple_biped_gaits_planner simple_biped_gaits
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  )

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )
