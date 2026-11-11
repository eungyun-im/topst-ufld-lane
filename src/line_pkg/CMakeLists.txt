cmake_minimum_required(VERSION 3.8)
project(line_pkg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(line_node src/line_node.cpp)
ament_target_dependencies(line_node
  rclcpp
  std_msgs
  sensor_msgs
)
target_link_libraries(line_node 
  ${OpenCV_LIBRARIES}
  /home/topst/ufld_deploy/TNN/build_topst_arm/libTNN.so
)
target_include_directories(line_node PRIVATE 
  ${OpenCV_INCLUDE_DIRS}
  /home/topst/ufld_deploy/TNN/include
)

install(TARGETS line_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
