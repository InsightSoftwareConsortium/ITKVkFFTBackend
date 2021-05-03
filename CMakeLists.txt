cmake_minimum_required(VERSION 3.10.2)
project(VkFFT)

include(FetchContent)
set(_itk_build_testing ${BUILD_TESTING})
#set(BUILD_TESTING OFF)
set(_itk_build_shared ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
set(vulkan_GIT_REPOSITORY "https://github.com/DTolm/VkFFT")
set(vulkan_GIT_TAG        "4608dd00899e3c50cd3cfb3f4b8ef056f09261a9")
# set(GIT_REPOSITORY        "https://github.com/VZout/vk_headers_lib.git")
# set(GIT_TAG               "e166a6948b0b4cd9a2b6a5e924d79f12d8aed99c")
FetchContent_Declare(
  vulkan_lib
  GIT_REPOSITORY ${vulkan_GIT_REPOSITORY}
  GIT_TAG ${vulkan_GIT_TAG}
  )
FetchContent_GetProperties(vulkan_lib)
if(NOT vulkan_lib_POPULATED)
  FetchContent_Populate(vulkan_lib)
  # find_library(VULKAN_LIBRARY NAMES vulkan-1 vulkan PATHS ${vulkan_SOURCE_DIR}/libs)
  # add_subdirectory(${vulkan_lib_SOURCE_DIR} ${vulkan_lib_BINARY_DIR})
endif()
set(BUILD_TESTING ${_itk_build_testing})
set(BUILD_SHARED_LIBS ${_itk_build_shared})

set(VkFFT_LIBRARIES VkFFT)

if(NOT ITK_SOURCE_DIR)
  find_package(ITK REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  set(ITK_DIR ${CMAKE_BINARY_DIR})
  itk_module_impl()
endif()
