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
#include "itkVkDefinitions.h"
#if (VKFFT_BACKEND == METAL)
#  define NS_PRIVATE_IMPLEMENTATION
#  define MTL_PRIVATE_IMPLEMENTATION
#  define CA_PRIVATE_IMPLEMENTATION
#  include "Foundation/Foundation.hpp"
#  include "Metal/Metal.hpp"
#  include "QuartzCore/QuartzCore.hpp"
#endif
#include "itkVkCommon.h"
#include "vkFFT.h"
#include "itkMacro.h"
#include <complex>
#include <cstring>
#include <iostream>
#include <memory>

namespace itk
{

VkFFTResult
VkCommon::Run(const VkGPU & vkGPU, const VkParameters & vkParameters)
{
  VkFFTResult resFFT{ VKFFT_SUCCESS };

  // Create the GPU context once and reuse it across calls. Reconfigure only when the
  // requested device or the transform shape changes -- never merely because the CPU
  // buffer pointers differ (they change on every call). The cached context is released
  // in the destructor, so transforms no longer create (and leak) a context per call.
  const bool deviceChanged{ vkGPU.device_id != m_VkGPU.device_id };
  const bool shapeChanged{ !vkParameters.SameShapeAs(m_VkParameters) };
  if (m_MustConfigure || deviceChanged || shapeChanged)
  {
    resFFT = this->ReleaseBackend();
    if (resFFT != VKFFT_SUCCESS)
    {
      return resFFT;
    }
    m_VkGPU.device_id = vkGPU.device_id;
    m_VkParameters = vkParameters;
    resFFT = this->ConfigureBackend();
    if (resFFT != VKFFT_SUCCESS)
    {
      return resFFT;
    }
    this->m_MustConfigure = false;
  }
  else
  {
    // Cached context and plan configuration are still valid; only the per-call CPU
    // buffer pointers and byte counts differ.
    m_VkParameters.inputCPUBuffer = vkParameters.inputCPUBuffer;
    m_VkParameters.inputBufferBytes = vkParameters.inputBufferBytes;
    m_VkParameters.outputCPUBuffer = vkParameters.outputCPUBuffer;
    m_VkParameters.outputBufferBytes = vkParameters.outputBufferBytes;
  }

  resFFT = this->PerformFFT();
  if (resFFT != VKFFT_SUCCESS)
  {
    return resFFT;
  }

  return resFFT;
}

VkFFTResult
VkCommon::ConfigureBackend()
{
  VkFFTResult resFFT{ VKFFT_SUCCESS };
#if (VKFFT_BACKEND == CUDA)
  CUresult    res{ CUDA_SUCCESS };
  cudaError_t res2{ cudaSuccess };
  res = cuInit(0);
  if (res != CUDA_SUCCESS)
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_INITIALIZE };
  res2 = cudaSetDevice((int)m_VkGPU.device_id);
  if (res2 != cudaSuccess)
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID };
  res = cuDeviceGet(&m_VkGPU.device, (int)m_VkGPU.device_id);
  if (res != CUDA_SUCCESS)
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_GET_DEVICE };
#  if CUDA_VERSION >= 13000
  res = cuCtxCreate(&m_VkGPU.context, nullptr, 0, (int)m_VkGPU.device);
#  else
  res = cuCtxCreate(&m_VkGPU.context, 0, (int)m_VkGPU.device);
#  endif
  if (res != CUDA_SUCCESS)
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT };

#elif (VKFFT_BACKEND == OPENCL)
  cl_int resCL{ CL_SUCCESS };

  // Begin code that mimics launchVkFFT from VkFFT/Vulkan_FFT.cpp, though just the OpenCL part.
  cl_uint numPlatforms;
  resCL = clGetPlatformIDs(0, nullptr, &numPlatforms);
  if (resCL != CL_SUCCESS)
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): clGetPlatformIDs returned " << resCL << std::endl;
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_INITIALIZE };
  }
  std::unique_ptr<cl_platform_id[]> platformsArray{ std::make_unique<cl_platform_id[]>(numPlatforms) };
  cl_platform_id *                  platforms{ &platformsArray[0] };
  if (!platforms)
    return VkFFTResult{ VKFFT_ERROR_MALLOC_FAILED };
  resCL = clGetPlatformIDs(numPlatforms, platforms, nullptr);
  if (resCL != CL_SUCCESS)
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): clGetPlatformIDs returned " << resCL << std::endl;
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_INITIALIZE };
  }
  uint64_t k{ 0 };
  for (uint64_t j{ 0 }; j < numPlatforms; j++)
  {
    // First probe: how many devices does this platform expose? An OpenCL
    // platform with zero compute devices is legitimate (e.g. Apple's
    // deprecated OpenCL framework on macOS 15 returns CL_DEVICE_NOT_FOUND
    // for CL_DEVICE_TYPE_ALL). Skip such platforms; calling clGetDeviceIDs
    // again with num_entries=0 would return CL_INVALID_VALUE.
    cl_uint numDevices{ 0 };
    resCL = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
    if (resCL == CL_DEVICE_NOT_FOUND || numDevices == 0)
    {
      continue;
    }
    if (resCL != CL_SUCCESS)
    {
      std::cerr << __FILE__ "(" << __LINE__ << "): clGetDeviceIDs(count) returned " << resCL << std::endl;
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_GET_DEVICE };
    }
    std::unique_ptr<cl_device_id[]> deviceListArray{ std::make_unique<cl_device_id[]>(numDevices) };
    cl_device_id *                  deviceList{ &deviceListArray[0] };
    if (!deviceList)
      return VkFFTResult{ VKFFT_ERROR_MALLOC_FAILED };
    resCL = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, numDevices, deviceList, nullptr);
    if (resCL != CL_SUCCESS)
    {
      std::cerr << __FILE__ "(" << __LINE__ << "): clGetDeviceIDs returned " << resCL << std::endl;
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_GET_DEVICE };
    }
    for (uint64_t i{ 0 }; i < numDevices; i++)
    {
      if (k == m_VkGPU.device_id)
      {
        m_VkGPU.platform = platforms[j];
        m_VkGPU.device = deviceList[i];
        const cl_context_properties contextProperties[]{ CL_CONTEXT_PLATFORM,
                                                         reinterpret_cast<cl_context_properties>(m_VkGPU.platform),
                                                         0 };
        m_VkGPU.context = clCreateContext(contextProperties, 1, &m_VkGPU.device, NULL, NULL, &resCL);
        if (resCL != CL_SUCCESS)
        {
          std::cerr << __FILE__ "(" << __LINE__ << "): clCreateContext returned " << resCL << std::endl;
          return VkFFTResult{ VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT };
        }
        m_VkGPU.commandQueue = clCreateCommandQueue(m_VkGPU.context, m_VkGPU.device, 0, &resCL);
        if (resCL != CL_SUCCESS)
        {
          std::cerr << __FILE__ "(" << __LINE__ << "): clCreateCommandQueue returned " << resCL << std::endl;
          return VkFFTResult{ VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE };
        }
        k++;
      }
      else
      {
        k++;
      }
    }
  }
#elif (VKFFT_BACKEND == LEVEL_ZERO)
  ze_result_t resZE{ ZE_RESULT_SUCCESS };
  resZE = zeInit(0);
  if (resZE != ZE_RESULT_SUCCESS)
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): zeInit returned " << resZE << std::endl;
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_INITIALIZE };
  }
  uint32_t numDrivers{ 0 };
  resZE = zeDriverGet(&numDrivers, nullptr);
  if (resZE != ZE_RESULT_SUCCESS || numDrivers == 0)
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_INITIALIZE };
  std::unique_ptr<ze_driver_handle_t[]> drivers{ std::make_unique<ze_driver_handle_t[]>(numDrivers) };
  resZE = zeDriverGet(&numDrivers, drivers.get());
  if (resZE != ZE_RESULT_SUCCESS)
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_INITIALIZE };
  uint64_t k{ 0 };
  bool     found{ false };
  for (uint32_t j{ 0 }; j < numDrivers && !found; ++j)
  {
    uint32_t numDevices{ 0 };
    resZE = zeDeviceGet(drivers[j], &numDevices, nullptr);
    if (resZE != ZE_RESULT_SUCCESS)
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_GET_DEVICE };
    std::unique_ptr<ze_device_handle_t[]> deviceList{ std::make_unique<ze_device_handle_t[]>(numDevices) };
    resZE = zeDeviceGet(drivers[j], &numDevices, deviceList.get());
    if (resZE != ZE_RESULT_SUCCESS)
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_GET_DEVICE };
    for (uint32_t i{ 0 }; i < numDevices && !found; ++i)
    {
      if (k == m_VkGPU.device_id)
      {
        m_VkGPU.driver = drivers[j];
        m_VkGPU.device = deviceList[i];

        ze_context_desc_t contextDescription{};
        contextDescription.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
        resZE = zeContextCreate(m_VkGPU.driver, &contextDescription, &m_VkGPU.context);
        if (resZE != ZE_RESULT_SUCCESS)
          return VkFFTResult{ VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT };

        uint32_t queueGroupCount{ 0 };
        resZE = zeDeviceGetCommandQueueGroupProperties(m_VkGPU.device, &queueGroupCount, nullptr);
        if (resZE != ZE_RESULT_SUCCESS || queueGroupCount == 0)
          return VkFFTResult{ VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE };
        std::unique_ptr<ze_command_queue_group_properties_t[]> queueGroupProps{
          std::make_unique<ze_command_queue_group_properties_t[]>(queueGroupCount)
        };
        resZE = zeDeviceGetCommandQueueGroupProperties(m_VkGPU.device, &queueGroupCount, queueGroupProps.get());
        if (resZE != ZE_RESULT_SUCCESS)
          return VkFFTResult{ VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE };
        uint32_t commandQueueID{ static_cast<uint32_t>(-1) };
        for (uint32_t g{ 0 }; g < queueGroupCount; ++g)
        {
          if ((queueGroupProps[g].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) &&
              (queueGroupProps[g].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY))
          {
            commandQueueID = g;
            break;
          }
        }
        if (commandQueueID == static_cast<uint32_t>(-1))
          return VkFFTResult{ VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE };
        m_VkGPU.commandQueueID = commandQueueID;

        ze_command_queue_desc_t commandQueueDescription{};
        commandQueueDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        commandQueueDescription.ordinal = commandQueueID;
        commandQueueDescription.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        commandQueueDescription.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
        resZE = zeCommandQueueCreate(m_VkGPU.context, m_VkGPU.device, &commandQueueDescription, &m_VkGPU.commandQueue);
        if (resZE != ZE_RESULT_SUCCESS)
          return VkFFTResult{ VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE };
        found = true;
      }
      else
      {
        ++k;
      }
    }
  }
  if (!found)
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): Level Zero device_id " << m_VkGPU.device_id << " not found"
              << std::endl;
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_GET_DEVICE };
  }
#elif (VKFFT_BACKEND == METAL)
  // Enumerate Metal devices and pick by device_id.
  NS::Array * devices = MTL::CopyAllDevices();
  if (devices == nullptr || devices->count() == 0)
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): no Metal devices found" << std::endl;
    if (devices != nullptr)
      devices->release();
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_INITIALIZE };
  }
  if (m_VkGPU.device_id >= devices->count())
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): Metal device_id " << m_VkGPU.device_id << " out of range (have "
              << devices->count() << ")" << std::endl;
    devices->release();
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_GET_DEVICE };
  }
  m_VkGPU.device = devices->object<MTL::Device>(m_VkGPU.device_id);
  m_VkGPU.device->retain();
  devices->release();
  m_VkGPU.queue = m_VkGPU.device->newCommandQueue();
  if (m_VkGPU.queue == nullptr)
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): newCommandQueue failed" << std::endl;
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE };
  }
#endif

  // Proceed by doing something similar to user_benchmark_VkFFT from
  // VkFFT/benchmark_scripts/vkFFT_scripts/src/user_benchmark_VkFFT.cpp, but without file_output and
  // output.

  m_VkFFTConfiguration.size[0] = std::max(m_VkParameters.X, (decltype(m_VkParameters.X))1);
  m_VkFFTConfiguration.size[1] = std::max(m_VkParameters.Y, (decltype(m_VkParameters.Y))1);
  m_VkFFTConfiguration.size[2] = std::max(m_VkParameters.Z, (decltype(m_VkParameters.Z))1);
  m_VkFFTConfiguration.FFTdim = 3;
  if (m_VkFFTConfiguration.size[2] == 1)
  {
    --m_VkFFTConfiguration.FFTdim;
    if (m_VkFFTConfiguration.size[1] == 1)
    {
      --m_VkFFTConfiguration.FFTdim;
    }
  }
  m_VkFFTConfiguration.numberBatches = m_VkParameters.B;
  m_VkFFTConfiguration.performR2C = m_VkParameters.fft == FFTEnum::C2C ? 0 : 1;
  if (m_VkParameters.P == PrecisionEnum::DOUBLE)
  {
    m_VkFFTConfiguration.doublePrecision = 1;
  }
  for (size_t dim{ 0 }; dim < 3; ++dim)
  {
    m_VkFFTConfiguration.omitDimension[dim] = m_VkParameters.omitDimension[dim];
  }
  // if (m_VkParameters.P == HALF)
  //   m_VkFFTConfiguration.halfPrecision = 1;
  m_VkFFTConfiguration.normalize = m_VkParameters.normalized == NormalizationEnum::NORMALIZED ? 1 : 0;
  // After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device
  // - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU
  // memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] -
  // allocated GPU memory, where kernel for convolution is stored.
#if (VKFFT_BACKEND == CUDA)
  m_VkFFTConfiguration.device = &m_VkGPU.device;
#elif (VKFFT_BACKEND == OPENCL)
  m_VkFFTConfiguration.device = &m_VkGPU.device;
  m_VkFFTConfiguration.platform = &m_VkGPU.platform;
  m_VkFFTConfiguration.context = &m_VkGPU.context;
#elif (VKFFT_BACKEND == LEVEL_ZERO)
  m_VkFFTConfiguration.device = &m_VkGPU.device;
  m_VkFFTConfiguration.context = &m_VkGPU.context;
  m_VkFFTConfiguration.commandQueue = &m_VkGPU.commandQueue;
  m_VkFFTConfiguration.commandQueueID = m_VkGPU.commandQueueID;
#elif (VKFFT_BACKEND == METAL)
  // Metal's VkFFTConfiguration takes single pointers, not pointer-to-pointer.
  m_VkFFTConfiguration.device = m_VkGPU.device;
  m_VkFFTConfiguration.queue = m_VkGPU.queue;
#endif

  m_VkFFTConfiguration.makeInversePlanOnly = (m_VkParameters.I == DirectionEnum::INVERSE);
  m_VkFFTConfiguration.makeForwardPlanOnly = (m_VkParameters.I == DirectionEnum::FORWARD);

  if (m_VkParameters.fft == FFTEnum::C2C)
  {
    // For C2C computation we can do everything in the in-place-computation buffer.
    m_VkFFTConfiguration.bufferNum = 1;
    m_VkFFTConfiguration.bufferStride[0] = m_VkFFTConfiguration.size[0];
    m_VkFFTConfiguration.bufferStride[1] = m_VkFFTConfiguration.bufferStride[0] * m_VkFFTConfiguration.size[1];
    m_VkFFTConfiguration.bufferStride[2] = m_VkFFTConfiguration.bufferStride[1] * m_VkFFTConfiguration.size[2];
    m_VkFFTConfiguration.bufferSize = &m_VkFFTConfiguration.bufferStride[2];
    const uint64_t bufferBytes{ 2UL * m_VkParameters.PSize * *m_VkFFTConfiguration.bufferSize };
    itkAssertOrThrowMacro(bufferBytes == m_VkParameters.inputBufferBytes,
                          "CPU and GPU input buffers are of different sizes.");
    itkAssertOrThrowMacro(bufferBytes == m_VkParameters.outputBufferBytes,
                          "CPU and GPU output buffers are of different sizes.");
  }
  else
  {
    // Either R2HalfH or R2FullH computation. Either forward or inverse.
    m_VkFFTConfiguration.bufferNum = 1;
    if (m_VkParameters.fft == FFTEnum::R2HalfH)
    {
      // R2HalfH computation, either forward or inverse.
      m_VkFFTConfiguration.bufferStride[0] = m_VkFFTConfiguration.size[0] / 2 + 1;
    }
    else
    {
      // R2FullH computation, either forward or inverse.
      m_VkFFTConfiguration.bufferStride[0] = m_VkFFTConfiguration.size[0];
    }
    m_VkFFTConfiguration.bufferStride[1] = m_VkFFTConfiguration.bufferStride[0] * m_VkFFTConfiguration.size[1];
    m_VkFFTConfiguration.bufferStride[2] = m_VkFFTConfiguration.bufferStride[1] * m_VkFFTConfiguration.size[2];
    m_VkFFTConfiguration.bufferSize = &m_VkFFTConfiguration.bufferStride[2];
    const uint64_t bufferBytes{ 2UL * m_VkParameters.PSize * *m_VkFFTConfiguration.bufferSize };

    if (m_VkParameters.I == DirectionEnum::FORWARD)
    {
      // Either R2FullH or R2HalfH.  For forward computation, we have a smaller input buffer.
      m_VkFFTConfiguration.isInputFormatted = 1;
      m_VkFFTConfiguration.inputBufferNum = 1;
      m_VkFFTConfiguration.inputBufferStride[0] = m_VkFFTConfiguration.size[0];
      m_VkFFTConfiguration.inputBufferStride[1] =
        m_VkFFTConfiguration.inputBufferStride[0] * m_VkFFTConfiguration.size[1];
      m_VkFFTConfiguration.inputBufferStride[2] =
        m_VkFFTConfiguration.inputBufferStride[1] * m_VkFFTConfiguration.size[2];
      m_VkFFTConfiguration.inputBufferSize = &m_VkFFTConfiguration.inputBufferStride[2];
      const uint64_t inputBufferBytes{ 1UL * m_VkParameters.PSize * *m_VkFFTConfiguration.inputBufferSize };
      itkAssertOrThrowMacro(inputBufferBytes == m_VkParameters.inputBufferBytes,
                            "CPU and GPU input buffers are of different sizes.");
      itkAssertOrThrowMacro(bufferBytes == m_VkParameters.outputBufferBytes,
                            "CPU and GPU output buffers are of different sizes.");
    }
    else
    {
      // Either R2FullH or R2HalfH.  For inverse computation, we have a smaller output buffer.
      m_VkFFTConfiguration.isOutputFormatted = 1;
      m_VkFFTConfiguration.outputBufferNum = 1;
      m_VkFFTConfiguration.outputBufferStride[0] = m_VkFFTConfiguration.size[0];
      m_VkFFTConfiguration.outputBufferStride[1] =
        m_VkFFTConfiguration.outputBufferStride[0] * m_VkFFTConfiguration.size[1];
      m_VkFFTConfiguration.outputBufferStride[2] =
        m_VkFFTConfiguration.outputBufferStride[1] * m_VkFFTConfiguration.size[2];
      m_VkFFTConfiguration.outputBufferSize = &m_VkFFTConfiguration.outputBufferStride[2];
      uint64_t outputBufferBytes{ 1UL * m_VkParameters.PSize * *m_VkFFTConfiguration.outputBufferSize };
      itkAssertOrThrowMacro(bufferBytes == m_VkParameters.inputBufferBytes,
                            "CPU and GPU input buffers are of different sizes.");
      itkAssertOrThrowMacro(outputBufferBytes == m_VkParameters.outputBufferBytes,
                            "CPU and GPU output buffers are of different sizes.");
    }
  }

  return resFFT;
}

VkFFTResult
VkCommon::PerformFFT()
{
  VkFFTResult resFFT{ VKFFT_SUCCESS };

#if (VKFFT_BACKEND == CUDA)
  cudaError resCu{ cudaSuccess };

  // Make this instance's context current. With several filters (each its own VkCommon and
  // CUDA context), the current context otherwise belongs to whichever filter configured last,
  // and the runtime API would touch this instance's cached buffers under the wrong context.
  cuCtxSetCurrent(m_VkGPU.context);

  if (!m_PlanConfigured)
  {
    // Allocate the in-place-computation buffer (persistent for this shape).
    const uint64_t bufferBytes{ 2UL * m_VkParameters.PSize * *m_VkFFTConfiguration.bufferSize };
    resCu = cudaMalloc((void **)&m_GPUBuffer, bufferBytes);
    if (resCu != cudaSuccess)
    {
      std::cerr << __FILE__ "(" << __LINE__ << "): cudaMalloc returned " << resCu << std::endl;
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
    }
    m_VkFFTConfiguration.buffer = reinterpret_cast<void **>(&m_GPUBuffer);

    if (m_VkParameters.fft == FFTEnum::C2C)
    {
      // For C2C computation we can do everything in the in-place-computation buffer.
      m_InputGPUBuffer = m_GPUBuffer;
      m_OutputGPUBuffer = m_GPUBuffer;
    }
    else if (m_VkParameters.I == DirectionEnum::FORWARD)
    {
      m_OutputGPUBuffer = m_GPUBuffer;

      // Either R2FullH or R2HalfH.  For forward computation, we have a smaller input buffer.
      const uint64_t inputBufferBytes{ 1UL * m_VkParameters.PSize * *m_VkFFTConfiguration.inputBufferSize };
      resCu = cudaMalloc((void **)&m_InputGPUBuffer, inputBufferBytes);
      if (resCu != cudaSuccess)
      {
        std::cerr << __FILE__ "(" << __LINE__ << "): cudaMalloc returned " << resCu << std::endl;
        return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
      }
      m_VkFFTConfiguration.inputBuffer = reinterpret_cast<void **>(&m_InputGPUBuffer);
    }
    else
    {
      m_InputGPUBuffer = m_GPUBuffer;

      // Either R2FullH or R2HalfH.  For inverse computation, we have a smaller output buffer.
      const uint64_t outputBufferBytes{ 1UL * m_VkParameters.PSize * *m_VkFFTConfiguration.outputBufferSize };
      resCu = cudaMalloc((void **)&m_OutputGPUBuffer, outputBufferBytes);
      if (resCu != cudaSuccess)
      {
        std::cerr << __FILE__ "(" << __LINE__ << "): cudaMalloc returned " << resCu << std::endl;
        return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
      }
      m_VkFFTConfiguration.outputBuffer = reinterpret_cast<void **>(&m_OutputGPUBuffer);
    }
  }

  // Copy input from CPU to GPU (per call)
  resCu = cudaMemcpy(
    m_InputGPUBuffer, m_VkParameters.inputCPUBuffer, m_VkParameters.inputBufferBytes, cudaMemcpyHostToDevice);
  if (resCu != cudaSuccess)
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): cudaMemcpy returned " << resCu << std::endl;
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_COPY };
  }

#elif (VKFFT_BACKEND == OPENCL)
  cl_int resCL{ CL_SUCCESS };

  if (!m_PlanConfigured)
  {
    // Configure the persistent buffers for this shape. Some of m_*GPUBuffer alias each other
    // (C2C, or the in-place direction), which the distinct-buffer free in ReleaseBackend handles.
    if (m_VkParameters.fft == FFTEnum::C2C)
    {
      // For C2C computation we can do everything in the in-place-computation buffer.
      const uint64_t bufferBytes{ 2UL * m_VkParameters.PSize * *m_VkFFTConfiguration.bufferSize };
      m_GPUBuffer = clCreateBuffer(m_VkGPU.context, CL_MEM_READ_WRITE, bufferBytes, nullptr, &resCL);
      m_InputGPUBuffer = m_GPUBuffer;
      m_OutputGPUBuffer = m_GPUBuffer;
      if (resCL != CL_SUCCESS)
      {
        std::cerr << __FILE__ "(" << __LINE__ << "): clCreateBuffer returned " << resCL << std::endl;
        return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
      }
      m_VkFFTConfiguration.buffer = &m_GPUBuffer;
    }
    else
    {
      // Either R2HalfH or R2FullH computation. Either forward or inverse.
      const uint64_t bufferBytes{ 2UL * m_VkParameters.PSize * *m_VkFFTConfiguration.bufferSize };
      m_GPUBuffer = clCreateBuffer(m_VkGPU.context, CL_MEM_READ_WRITE, bufferBytes, nullptr, &resCL);
      if (resCL != CL_SUCCESS)
      {
        std::cerr << __FILE__ "(" << __LINE__ << "): clCreateBuffer returned " << resCL << std::endl;
        return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
      }
      m_VkFFTConfiguration.buffer = &m_GPUBuffer;

      if (m_VkParameters.I == DirectionEnum::FORWARD)
      {
        // Either R2FullH or R2HalfH.  For forward computation, we have a smaller input buffer.
        const uint64_t inputBufferBytes{ 1UL * m_VkParameters.PSize * *m_VkFFTConfiguration.inputBufferSize };
        m_InputGPUBuffer = clCreateBuffer(m_VkGPU.context, CL_MEM_READ_WRITE, inputBufferBytes, nullptr, &resCL);
        m_OutputGPUBuffer = m_GPUBuffer;
        if (resCL != CL_SUCCESS)
        {
          std::cerr << __FILE__ "(" << __LINE__ << "): clCreateBuffer returned " << resCL << std::endl;
          return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
        }
        m_VkFFTConfiguration.inputBuffer = &m_InputGPUBuffer;
      }
      else
      {
        // Either R2FullH or R2HalfH.  For inverse computation, we have a smaller output buffer.
        const uint64_t outputBufferBytes{ 1UL * m_VkParameters.PSize * *m_VkFFTConfiguration.outputBufferSize };
        m_InputGPUBuffer = m_GPUBuffer;
        m_OutputGPUBuffer = clCreateBuffer(m_VkGPU.context, CL_MEM_READ_WRITE, outputBufferBytes, nullptr, &resCL);
        if (resCL != CL_SUCCESS)
        {
          std::cerr << __FILE__ "(" << __LINE__ << "): clCreateBuffer returned " << resCL << std::endl;
          return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
        }
        m_VkFFTConfiguration.outputBuffer = &m_OutputGPUBuffer;
      }
    }
  }

  // Copy input from CPU to GPU (per call)
  resCL = clEnqueueWriteBuffer(m_VkGPU.commandQueue,
                               m_InputGPUBuffer,
                               CL_TRUE,
                               0,
                               m_VkParameters.inputBufferBytes,
                               m_VkParameters.inputCPUBuffer,
                               0,
                               nullptr,
                               nullptr);
  if (resCL != CL_SUCCESS)
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): clEnqueueWriteBuffer returned " << resCL << std::endl;
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_COPY };
  }
#elif (VKFFT_BACKEND == LEVEL_ZERO)
  ze_result_t resZE{ ZE_RESULT_SUCCESS };

  if (!m_PlanConfigured)
  {
    ze_device_mem_alloc_desc_t deviceMemDesc{};
    deviceMemDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;

    const uint64_t bufferBytes{ 2UL * m_VkParameters.PSize * *m_VkFFTConfiguration.bufferSize };
    resZE = zeMemAllocDevice(
      m_VkGPU.context, &deviceMemDesc, bufferBytes, m_VkParameters.PSize, m_VkGPU.device, &m_GPUBuffer);
    if (resZE != ZE_RESULT_SUCCESS)
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
    m_VkFFTConfiguration.buffer = &m_GPUBuffer;

    if (m_VkParameters.fft == FFTEnum::C2C)
    {
      m_InputGPUBuffer = m_GPUBuffer;
      m_OutputGPUBuffer = m_GPUBuffer;
    }
    else if (m_VkParameters.I == DirectionEnum::FORWARD)
    {
      const uint64_t inputBufferBytes{ 1UL * m_VkParameters.PSize * *m_VkFFTConfiguration.inputBufferSize };
      resZE = zeMemAllocDevice(
        m_VkGPU.context, &deviceMemDesc, inputBufferBytes, m_VkParameters.PSize, m_VkGPU.device, &m_InputGPUBuffer);
      if (resZE != ZE_RESULT_SUCCESS)
        return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
      m_OutputGPUBuffer = m_GPUBuffer;
      m_VkFFTConfiguration.inputBuffer = &m_InputGPUBuffer;
    }
    else
    {
      const uint64_t outputBufferBytes{ 1UL * m_VkParameters.PSize * *m_VkFFTConfiguration.outputBufferSize };
      resZE = zeMemAllocDevice(
        m_VkGPU.context, &deviceMemDesc, outputBufferBytes, m_VkParameters.PSize, m_VkGPU.device, &m_OutputGPUBuffer);
      if (resZE != ZE_RESULT_SUCCESS)
        return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
      m_InputGPUBuffer = m_GPUBuffer;
      m_VkFFTConfiguration.outputBuffer = &m_OutputGPUBuffer;
    }
  }

  // Host -> device copy via an immediate command list on the compute/copy queue group (per call).
  {
    ze_command_queue_desc_t copyQueueDesc{};
    copyQueueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    copyQueueDesc.ordinal = m_VkGPU.commandQueueID;
    copyQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    copyQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_list_handle_t copyCommandList{ nullptr };
    resZE = zeCommandListCreateImmediate(m_VkGPU.context, m_VkGPU.device, &copyQueueDesc, &copyCommandList);
    if (resZE != ZE_RESULT_SUCCESS)
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST };
    resZE = zeCommandListAppendMemoryCopy(copyCommandList,
                                          m_InputGPUBuffer,
                                          m_VkParameters.inputCPUBuffer,
                                          m_VkParameters.inputBufferBytes,
                                          nullptr,
                                          0,
                                          nullptr);
    if (resZE != ZE_RESULT_SUCCESS)
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_COPY };
    resZE = zeCommandQueueSynchronize(m_VkGPU.commandQueue, UINT32_MAX);
    if (resZE != ZE_RESULT_SUCCESS)
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_SYNCHRONIZE };
    zeCommandListDestroy(copyCommandList);
  }
#elif (VKFFT_BACKEND == METAL)
  // Metal shared-storage buffers are CPU-visible on Apple unified-memory systems,
  // so host<->device transfers reduce to memcpy into/out of MTL::Buffer::contents().
  const auto storageMode = MTL::ResourceStorageModeShared;
  if (!m_PlanConfigured)
  {
    if (m_VkParameters.fft == FFTEnum::C2C)
    {
      const uint64_t bufferBytes{ 2UL * m_VkParameters.PSize * *m_VkFFTConfiguration.bufferSize };
      m_GPUBuffer = m_VkGPU.device->newBuffer(bufferBytes, storageMode);
      if (m_GPUBuffer == nullptr)
        return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
      m_InputGPUBuffer = m_GPUBuffer;
      m_OutputGPUBuffer = m_GPUBuffer;
      m_VkFFTConfiguration.buffer = &m_GPUBuffer;
    }
    else
    {
      const uint64_t bufferBytes{ 2UL * m_VkParameters.PSize * *m_VkFFTConfiguration.bufferSize };
      m_GPUBuffer = m_VkGPU.device->newBuffer(bufferBytes, storageMode);
      if (m_GPUBuffer == nullptr)
        return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
      m_VkFFTConfiguration.buffer = &m_GPUBuffer;

      if (m_VkParameters.I == DirectionEnum::FORWARD)
      {
        const uint64_t inputBufferBytes{ 1UL * m_VkParameters.PSize * *m_VkFFTConfiguration.inputBufferSize };
        m_InputGPUBuffer = m_VkGPU.device->newBuffer(inputBufferBytes, storageMode);
        if (m_InputGPUBuffer == nullptr)
          return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
        m_OutputGPUBuffer = m_GPUBuffer;
        m_VkFFTConfiguration.inputBuffer = &m_InputGPUBuffer;
      }
      else
      {
        const uint64_t outputBufferBytes{ 1UL * m_VkParameters.PSize * *m_VkFFTConfiguration.outputBufferSize };
        m_InputGPUBuffer = m_GPUBuffer;
        m_OutputGPUBuffer = m_VkGPU.device->newBuffer(outputBufferBytes, storageMode);
        if (m_OutputGPUBuffer == nullptr)
          return VkFFTResult{ VKFFT_ERROR_FAILED_TO_ALLOCATE };
        m_VkFFTConfiguration.outputBuffer = &m_OutputGPUBuffer;
      }
    }
  }

  std::memcpy(m_InputGPUBuffer->contents(), m_VkParameters.inputCPUBuffer, m_VkParameters.inputBufferBytes);
#endif

  // Initialize the application once per plan. This loads shaders, creates the pipeline, and
  // compiles the FFT kernels (nvrtc for CUDA, clBuildProgram for OpenCL); it allocates no
  // buffers. The compiled plan and the persistent buffers above are reused across same-shape
  // calls -- buffers are bound per call through VkFFTLaunchParams below.
  if (!m_PlanConfigured)
  {
    m_VkFFTApplication = new VkFFTApplication{};
    resFFT = initializeVkFFT(m_VkFFTApplication, m_VkFFTConfiguration);
    if (resFFT != VKFFT_SUCCESS)
      return resFFT;
    m_PlanConfigured = true;
  }

  // Submit FFT or iFFT.
  VkFFTLaunchParams launchParams{};
  launchParams.inputBuffer = m_VkFFTConfiguration.inputBuffer;
  launchParams.buffer = m_VkFFTConfiguration.buffer;
  launchParams.outputBuffer = m_VkFFTConfiguration.outputBuffer;
#if (VKFFT_BACKEND == CUDA)
  // pass
#elif (VKFFT_BACKEND == OPENCL)
  launchParams.commandQueue = &m_VkGPU.commandQueue;
#elif (VKFFT_BACKEND == LEVEL_ZERO)
  ze_command_list_desc_t commandListDescription{};
  commandListDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  commandListDescription.commandQueueGroupOrdinal = m_VkGPU.commandQueueID;
  ze_command_list_handle_t launchCommandList{ nullptr };
  resZE = zeCommandListCreate(m_VkGPU.context, m_VkGPU.device, &commandListDescription, &launchCommandList);
  if (resZE != ZE_RESULT_SUCCESS)
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST };
  launchParams.commandList = &launchCommandList;
#elif (VKFFT_BACKEND == METAL)
  MTL::CommandBuffer *         metalCommandBuffer = m_VkGPU.queue->commandBuffer();
  MTL::ComputeCommandEncoder * metalEncoder = metalCommandBuffer->computeCommandEncoder();
  launchParams.commandBuffer = metalCommandBuffer;
  launchParams.commandEncoder = metalEncoder;
#endif

  resFFT = VkFFTAppend(m_VkFFTApplication, m_VkParameters.I == DirectionEnum::INVERSE ? 1 : -1, &launchParams);
  if (resFFT != VKFFT_SUCCESS)
    return resFFT;

#if (VKFFT_BACKEND == CUDA)
  resCu = cudaDeviceSynchronize();
  if (resCu != cudaSuccess)
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): cudaDeviceSynchronize returned " << resCu << std::endl;
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_SYNCHRONIZE };
  }

  // Copy result from GPU to CPU (per call); persistent buffers are freed in ReleaseBackend().
  resCu = cudaMemcpy(
    m_VkParameters.outputCPUBuffer, m_OutputGPUBuffer, m_VkParameters.outputBufferBytes, cudaMemcpyDeviceToHost);
  if (resCu != cudaSuccess)
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): cudaMemcpy returned " << resCu << std::endl;
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_COPY };
  }

#elif (VKFFT_BACKEND == OPENCL)
  resCL = clFinish(m_VkGPU.commandQueue);
  if (resCL != CL_SUCCESS)
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): clFinish returned " << resCL << std::endl;
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_SYNCHRONIZE };
  }

  // Copy result from GPU to CPU (per call); persistent buffers are freed in ReleaseBackend().
  resCL = clEnqueueReadBuffer(m_VkGPU.commandQueue,
                              m_OutputGPUBuffer,
                              CL_TRUE,
                              0,
                              m_VkParameters.outputBufferBytes,
                              m_VkParameters.outputCPUBuffer,
                              0,
                              nullptr,
                              nullptr);
  if (resCL != CL_SUCCESS)
  {
    std::cerr << __FILE__ "(" << __LINE__ << "): clEnqueueReadBuffer returned " << resCL << std::endl;
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_COPY };
  }
#elif (VKFFT_BACKEND == LEVEL_ZERO)
  resZE = zeCommandListClose(launchCommandList);
  if (resZE != ZE_RESULT_SUCCESS)
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE };
  resZE = zeCommandQueueExecuteCommandLists(m_VkGPU.commandQueue, 1, &launchCommandList, nullptr);
  if (resZE != ZE_RESULT_SUCCESS)
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE };
  resZE = zeCommandQueueSynchronize(m_VkGPU.commandQueue, UINT32_MAX);
  if (resZE != ZE_RESULT_SUCCESS)
    return VkFFTResult{ VKFFT_ERROR_FAILED_TO_SYNCHRONIZE };
  zeCommandListDestroy(launchCommandList);

  // Device -> host copy via an immediate command list.
  {
    ze_command_queue_desc_t copyQueueDesc{};
    copyQueueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    copyQueueDesc.ordinal = m_VkGPU.commandQueueID;
    copyQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    copyQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_list_handle_t copyCommandList{ nullptr };
    resZE = zeCommandListCreateImmediate(m_VkGPU.context, m_VkGPU.device, &copyQueueDesc, &copyCommandList);
    if (resZE != ZE_RESULT_SUCCESS)
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST };
    resZE = zeCommandListAppendMemoryCopy(copyCommandList,
                                          m_VkParameters.outputCPUBuffer,
                                          m_OutputGPUBuffer,
                                          m_VkParameters.outputBufferBytes,
                                          nullptr,
                                          0,
                                          nullptr);
    if (resZE != ZE_RESULT_SUCCESS)
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_COPY };
    resZE = zeCommandQueueSynchronize(m_VkGPU.commandQueue, UINT32_MAX);
    if (resZE != ZE_RESULT_SUCCESS)
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_SYNCHRONIZE };
    zeCommandListDestroy(copyCommandList);
  }
  // Persistent buffers are freed in ReleaseBackend().
#elif (VKFFT_BACKEND == METAL)
  metalEncoder->endEncoding();
  metalCommandBuffer->commit();
  metalCommandBuffer->waitUntilCompleted();

  std::memcpy(m_VkParameters.outputCPUBuffer, m_OutputGPUBuffer->contents(), m_VkParameters.outputBufferBytes);
  // Persistent buffers are released in ReleaseBackend().
#endif

  if (m_VkParameters.fft == FFTEnum::R2FullH && m_VkParameters.I == DirectionEnum::FORWARD)
  {
    // Compute complex conjugates for the R2FullH forward computation
    switch (m_VkParameters.P)
    {
      case PrecisionEnum::FLOAT:
      {
        using ComplexType = std::complex<float>;
        ComplexType * const outputCPUFloat{ reinterpret_cast<ComplexType *>(m_VkParameters.outputCPUBuffer) };
        for (uint64_t z{ 0 }; z < m_VkFFTConfiguration.size[2]; ++z)
        {
          for (uint64_t y{ 0 }; y < m_VkFFTConfiguration.size[1]; ++y)
          {
            const uint64_t offsetStart{ z * m_VkFFTConfiguration.bufferStride[1] +
                                        y * m_VkFFTConfiguration.bufferStride[0] };
            const uint64_t offsetEnd{ offsetStart + m_VkFFTConfiguration.bufferStride[0] };
            for (uint64_t x = (m_VkFFTConfiguration.size[0] - 1) / 2; x >= 1; --x)
            {
              outputCPUFloat[offsetEnd - x] = std::conj(outputCPUFloat[offsetStart + x]);
            }
          }
        }
      }
      break;
      case PrecisionEnum::DOUBLE:
      {
        using ComplexType = std::complex<double>;
        ComplexType * const outputCPUDouble{ reinterpret_cast<ComplexType *>(m_VkParameters.outputCPUBuffer) };
        for (uint64_t z{ 0 }; z < m_VkFFTConfiguration.size[2]; ++z)
        {
          for (uint64_t y{ 0 }; y < m_VkFFTConfiguration.size[1]; ++y)
          {
            const uint64_t offsetStart{ z * m_VkFFTConfiguration.bufferStride[1] +
                                        y * m_VkFFTConfiguration.bufferStride[0] };
            const uint64_t offsetEnd{ offsetStart + m_VkFFTConfiguration.bufferStride[0] };
            for (uint64_t x = (m_VkFFTConfiguration.size[0] - 1) / 2; x >= 1; --x)
            {
              outputCPUDouble[offsetEnd - x] = std::conj(outputCPUDouble[offsetStart + x]);
            }
          }
        }
      }
      break;
    } // end switch (m_VkParameters.P)
  } // end if(m_VkParameters.fft == R2FullH && m_VkParameters.I == DirectionEnum::FORWARD)
  // The plan (m_VkFFTApplication) and the GPU buffers persist for reuse and are released in
  // ReleaseBackend() on the next device/shape change or at destruction.

  return resFFT;
}

VkFFTResult
VkCommon::ReleaseBackend()
{
  VkFFTResult resFFT{ VKFFT_SUCCESS };

#if (VKFFT_BACKEND == CUDA)
  // Make this instance's context current so deleteVkFFT and the buffer frees below operate on
  // the context that owns these GPU resources, not whichever filter configured last.
  if (m_VkGPU.context)
    cuCtxSetCurrent(m_VkGPU.context);
#endif

  // Release the cached plan and persistent buffers before the context is destroyed; their GPU
  // resources belong to it. Distinct, non-aliased buffers are each freed exactly once.
  if (m_PlanConfigured)
  {
    deleteVkFFT(m_VkFFTApplication);
    delete m_VkFFTApplication;
    m_VkFFTApplication = nullptr;
    m_PlanConfigured = false;
  }

#if (VKFFT_BACKEND == CUDA)
  if (m_GPUBuffer)
    cudaFree(m_GPUBuffer);
  if (m_InputGPUBuffer && m_InputGPUBuffer != m_GPUBuffer)
    cudaFree(m_InputGPUBuffer);
  if (m_OutputGPUBuffer && m_OutputGPUBuffer != m_GPUBuffer && m_OutputGPUBuffer != m_InputGPUBuffer)
    cudaFree(m_OutputGPUBuffer);
  m_GPUBuffer = nullptr;
  m_InputGPUBuffer = nullptr;
  m_OutputGPUBuffer = nullptr;
  if (m_VkGPU.context)
  {
    cuCtxDestroy(m_VkGPU.context);
    m_VkGPU.context = 0;
  }
#elif (VKFFT_BACKEND == OPENCL)
  cl_int resCL{ CL_SUCCESS };

  if (m_GPUBuffer)
    clReleaseMemObject(m_GPUBuffer);
  if (m_InputGPUBuffer && m_InputGPUBuffer != m_GPUBuffer)
    clReleaseMemObject(m_InputGPUBuffer);
  if (m_OutputGPUBuffer && m_OutputGPUBuffer != m_GPUBuffer && m_OutputGPUBuffer != m_InputGPUBuffer)
    clReleaseMemObject(m_OutputGPUBuffer);
  m_GPUBuffer = nullptr;
  m_InputGPUBuffer = nullptr;
  m_OutputGPUBuffer = nullptr;

  if (m_VkGPU.commandQueue)
  {
    resCL = clReleaseCommandQueue(m_VkGPU.commandQueue);
    if (resCL != CL_SUCCESS)
    {
      std::cerr << __FILE__ "(" << __LINE__ << "): clReleaseCommandQueue returned " << resCL << std::endl;
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE };
    }
    m_VkGPU.commandQueue = 0;
  }

  if (m_VkGPU.context)
  {
    resCL = clReleaseContext(m_VkGPU.context);
    if (resCL != CL_SUCCESS)
    {
      std::cerr << __FILE__ "(" << __LINE__ << "): clReleaseContext returned " << resCL << std::endl;
      return VkFFTResult{ VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE };
    }
    m_VkGPU.context = 0;
  }
#elif (VKFFT_BACKEND == LEVEL_ZERO)
  if (m_VkGPU.context)
  {
    if (m_GPUBuffer)
      zeMemFree(m_VkGPU.context, m_GPUBuffer);
    if (m_InputGPUBuffer && m_InputGPUBuffer != m_GPUBuffer)
      zeMemFree(m_VkGPU.context, m_InputGPUBuffer);
    if (m_OutputGPUBuffer && m_OutputGPUBuffer != m_GPUBuffer && m_OutputGPUBuffer != m_InputGPUBuffer)
      zeMemFree(m_VkGPU.context, m_OutputGPUBuffer);
  }
  m_GPUBuffer = nullptr;
  m_InputGPUBuffer = nullptr;
  m_OutputGPUBuffer = nullptr;

  if (m_VkGPU.commandQueue)
  {
    zeCommandQueueDestroy(m_VkGPU.commandQueue);
    m_VkGPU.commandQueue = nullptr;
  }
  if (m_VkGPU.context)
  {
    zeContextDestroy(m_VkGPU.context);
    m_VkGPU.context = nullptr;
  }
#elif (VKFFT_BACKEND == METAL)
  if (m_GPUBuffer)
    m_GPUBuffer->release();
  if (m_InputGPUBuffer && m_InputGPUBuffer != m_GPUBuffer)
    m_InputGPUBuffer->release();
  if (m_OutputGPUBuffer && m_OutputGPUBuffer != m_GPUBuffer && m_OutputGPUBuffer != m_InputGPUBuffer)
    m_OutputGPUBuffer->release();
  m_GPUBuffer = nullptr;
  m_InputGPUBuffer = nullptr;
  m_OutputGPUBuffer = nullptr;

  if (m_VkGPU.queue)
  {
    m_VkGPU.queue->release();
    m_VkGPU.queue = nullptr;
  }
  if (m_VkGPU.device)
  {
    m_VkGPU.device->release();
    m_VkGPU.device = nullptr;
  }
#endif

  // Invalidate cache description!!!

  return resFFT;
}

} // end namespace itk
