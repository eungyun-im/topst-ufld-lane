cmake_minimum_required(VERSION 3.8)
project(ufld_line_pkg)

option(UFLD_USE_TNN "Build UFLD node with Tencent TNN inference backend" OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(ufld_line_node
  src/ufld_line_node.cpp
  src/ufld_decoder.cpp
)

target_include_directories(ufld_line_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(ufld_line_node
  rclcpp
  std_msgs
  sensor_msgs
)

target_link_libraries(ufld_line_node ${OpenCV_LIBRARIES})

if(UFLD_USE_TNN)
  if(NOT DEFINED TNN_ROOT)
    message(FATAL_ERROR "Set -DTNN_ROOT=/path/to/TNN when UFLD_USE_TNN=ON")
  endif()

  set(TNN_INCLUDE_DIR "${TNN_ROOT}/include" CACHE PATH "TNN include directory")
  if(NOT DEFINED TNN_LIB_DIR)
    set(TNN_LIB_DIR "${TNN_ROOT}/build")
  endif()

  find_library(TNN_LIB
    NAMES TNN tnn
    PATHS
      "${TNN_LIB_DIR}"
      "${TNN_ROOT}/build"
      "${TNN_ROOT}/build_armlinux"
      "${TNN_ROOT}/benchmark/benchmark_armlinux/build"
    NO_DEFAULT_PATH
  )

  if(NOT TNN_LIB)
    message(FATAL_ERROR "Could not find libTNN. Set -DTNN_LIB_DIR=/path/to/libTNN.so")
  endif()

  target_sources(ufld_line_node PRIVATE src/tnn_ufld_runner.cpp)
  target_include_directories(ufld_line_node PRIVATE "${TNN_INCLUDE_DIR}")
  target_link_libraries(ufld_line_node ${TNN_LIB})
  target_compile_definitions(ufld_line_node PRIVATE UFLD_USE_TNN=1)
  set_target_properties(ufld_line_node PROPERTIES
    BUILD_RPATH "${TNN_LIB_DIR}"
    INSTALL_RPATH "${TNN_LIB_DIR}"
  )
else()
  target_sources(ufld_line_node PRIVATE src/tnn_ufld_runner_stub.cpp)
endif()

install(TARGETS ufld_line_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
