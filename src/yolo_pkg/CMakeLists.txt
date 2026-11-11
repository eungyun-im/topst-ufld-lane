cmake_minimum_required(VERSION 3.8)
project(yolo_pkg)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/yolo_node_ort.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
