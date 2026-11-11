cmake_minimum_required(VERSION 3.8)
project(mission_planner_pkg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(mission_planner_node src/mission_planner_node.cpp)
ament_target_dependencies(mission_planner_node
  rclcpp
  std_msgs
)

install(TARGETS mission_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
