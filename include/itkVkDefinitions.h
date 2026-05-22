/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkVkDefinitions_h
#define itkVkDefinitions_h

// Backend selection
#define VULKAN 0
#define CUDA 1
#define HIP 2
#define OPENCL 3
#define LEVEL_ZERO 4
#define METAL 5

// Defensive default: when VKFFT_BACKEND is not set on the command line
// (e.g. castxml wrapping invocations that bypass our top-level
// add_compile_definitions), fall back to OpenCL so vkFFT.h does not
// take the Vulkan branch and try to #include <vulkan/vulkan.h>.
#ifndef VKFFT_BACKEND
#  define VKFFT_BACKEND OPENCL
#endif

#endif // itkVkDefinitions_h
