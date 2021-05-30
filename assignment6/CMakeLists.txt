cmake_minimum_required(VERSION 3.1)
set(CMAKE_C_COMPILER "gcc")

project(assignment6)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -fopenmp")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(BOOST_ROOT /usr/local/boost_1_76_0)
set(BOOST_LIBRARYDIR /usr/local/boost_1_76_0/libs)
set(Boost_USE_STATIC_LIBS ON)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"   ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"     ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"    ON)

find_package(LIBIGL REQUIRED QUIET)
find_package(Boost COMPONENTS system filesystem REQUIRED)

if (Boost_FOUND)
  include_directories (SYSTEM ${Boost_INCLUDE_DIR})
endif ()


# Add your project files
FILE(GLOB SRCFILES src/*.cpp)
add_executable(${PROJECT_NAME} ${SRCFILES} src/LandmarkSelector.cpp)
target_link_libraries(${PROJECT_NAME} igl::core igl::opengl_glfw igl::opengl_glfw_imgui ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})
