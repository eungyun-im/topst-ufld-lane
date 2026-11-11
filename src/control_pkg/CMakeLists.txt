cmake_minimum_required(VERSION 3.8)
project(control_pkg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(control_msgs REQUIRED)

add_executable(control_node src/control_node.cpp)
ament_target_dependencies(control_node
  rclcpp
  std_msgs
  control_msgs
)

install(TARGETS control_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
