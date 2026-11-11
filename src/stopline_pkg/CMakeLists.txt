cmake_minimum_required(VERSION 3.8)
project(stopline_pkg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(stopline_node src/stopline_node.cpp)
ament_target_dependencies(stopline_node
  rclcpp
  std_msgs
  sensor_msgs
)
target_link_libraries(stopline_node ${OpenCV_LIBRARIES})
target_include_directories(stopline_node PRIVATE ${OpenCV_INCLUDE_DIRS})

install(TARGETS stopline_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
