cmake_minimum_required(VERSION 3.10.2)
project(VkFFTBackend)
include(FetchContent)
set(_itk_build_testing ${BUILD_TESTING})
#set(BUILD_TESTING OFF)
set(_itk_build_shared ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)

if(NOT OpenCL-ICD-Loader_SOURCE_DIR)
  find_package(OpenCL-ICD-Loader REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${OpenCL-ICD-Loader_CMAKE_DIR})
else()
  set(OpenCL-ICD-Loader_DIR ${CMAKE_BINARY_DIR})
endif()

set(VKFFT_BACKEND 3 CACHE STRING "0 - Vulkan, 1 - CUDA, 2 - HIP, 3 - OpenCL")
set(vulkan_GIT_REPOSITORY "https://github.com/DTolm/VkFFT") # original source
set(vulkan_GIT_TAG        "5dc53fb5c28dd57f922bfa454827264a7aac9477")
# set(vulkan_GIT_REPOSITORY "https://github.com/thewtex/VkFFT") # our copy of original source
# set(vulkan_GIT_TAG        "86d8b45c7465ccec8cc20d4bf1710eef04c92c47") # backend-cmake-cache
# set(GIT_REPOSITORY        "https://github.com/VZout/vk_headers_lib.git") # source for something similar
# set(GIT_TAG               "e166a6948b0b4cd9a2b6a5e924d79f12d8aed99c")
FetchContent_Declare(
  vulkan_lib
  # URL /home/lee.newberg/git/VkFFT # Remove me!!!
  GIT_REPOSITORY ${vulkan_GIT_REPOSITORY}
  GIT_TAG ${vulkan_GIT_TAG}
  )
FetchContent_MakeAvailable(vulkan_lib)
set(VkFFTBackend_LIBRARIES VkFFT)

set(BUILD_TESTING ${_itk_build_testing})
set(BUILD_SHARED_LIBS ${_itk_build_shared})

if(NOT ITK_SOURCE_DIR)
  find_package(ITK REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  set(ITK_DIR ${CMAKE_BINARY_DIR})
  itk_module_impl()
endif()
