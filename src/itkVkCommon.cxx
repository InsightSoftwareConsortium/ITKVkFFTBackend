/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "itkVkCommon.h"
#include "itkMacro.h"
#include <complex>
#include <iostream>
#include <memory>

namespace itk
{

VkFFTResult
VkCommon::run(VkGPU * vkGPU, const VkParameters * vkParameters)
{
  VkFFTResult resFFT{ VKFFT_SUCCESS };
  cl_int      resCL{ CL_SUCCESS };

  {
    // Begin code that mimics launchVkFFT from VkFFT/Vulkan_FFT.cpp, though just the OpenCL part.
    cl_uint numPlatforms;
    resCL = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (resCL != CL_SUCCESS)
      return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    std::unique_ptr<cl_platform_id[]> platformsArray{ std::make_unique<cl_platform_id[]>(numPlatforms) };
    cl_platform_id *                  platforms{ &platformsArray[0] };
    if (!platforms)
      return VKFFT_ERROR_MALLOC_FAILED;
    resCL = clGetPlatformIDs(numPlatforms, platforms, nullptr);
    if (resCL != CL_SUCCESS)
      return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    uint64_t k{ 0 };
    for (uint64_t j = 0; j < numPlatforms; j++)
    {
      cl_uint numDevices;
      resCL = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
      std::unique_ptr<cl_device_id[]> deviceListArray{ std::make_unique<cl_device_id[]>(numDevices) };
      cl_device_id *                  deviceList{ &deviceListArray[0] };
      if (!deviceList)
        return VKFFT_ERROR_MALLOC_FAILED;
      resCL = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, numDevices, deviceList, nullptr);
      if (resCL != CL_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
      for (uint64_t i = 0; i < numDevices; i++)
      {
        if (k == vkGPU->device_id)
        {
          vkGPU->platform = platforms[j];
          vkGPU->device = deviceList[i];
          vkGPU->context = clCreateContext(NULL, 1, &vkGPU->device, NULL, NULL, &resCL);
          if (resCL != CL_SUCCESS)
            return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
          cl_command_queue commandQueue{ clCreateCommandQueue(vkGPU->context, vkGPU->device, 0, &resCL) };
          if (resCL != CL_SUCCESS)
            return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
          vkGPU->commandQueue = commandQueue;
          k++;
        }
        else
        {
          k++;
        }
      }
    }
  }

  {
    // Proceed by doing something similar to user_benchmark_VkFFT from
    // VkFFT/benchmark_scripts/vkFFT_scripts/src/user_benchmark_VkFFT.cpp, but without file_output and
    // output, and just the OpenCL part.

    VkFFTConfiguration configuration{};
    VkFFTApplication   app{};
    configuration.size[0] = std::max(vkParameters->X, (decltype(vkParameters->X))1);
    configuration.size[1] = std::max(vkParameters->Y, (decltype(vkParameters->Y))1);
    configuration.size[2] = std::max(vkParameters->Z, (decltype(vkParameters->Z))1);
    configuration.FFTdim = 3;
    if (configuration.size[2] == 1)
    {
      --configuration.FFTdim;
      if (configuration.size[1] == 1)
      {
        --configuration.FFTdim;
      }
    }
    configuration.numberBatches = vkParameters->B;
    configuration.performR2C = vkParameters->fftType == C2C ? 0 : 1;
    if (vkParameters->P == DOUBLE)
      configuration.doublePrecision = 1;
    // if (vkParameters->P == HALF)
    //   configuration.halfPrecision = 1;
    configuration.normalize = vkParameters->normalized == NORMALIZED ? 1 : 0;
    // After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device
    // - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU
    // memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] -
    // allocated GPU memory, where kernel for convolution is stored.
    configuration.device = &vkGPU->device;
    configuration.platform = &vkGPU->platform;
    configuration.context = &vkGPU->context;

    configuration.makeInversePlanOnly = (vkParameters->I == INVERSE);
    configuration.makeForwardPlanOnly = (vkParameters->I == FORWARD);

    // Configure the buffers.  Some of these three pointers will be nullptr or be duplicates of each
    // other, so don't release all of them at the end.  All re-striding of data (for R2HH or R2FH,
    // regardless of forward vs. inverse) is done by VkFFT between the two GPU buffers it uses.
    cl_mem inputGPUBuffer{ nullptr };  // Copy from CPU input buffer to this GPU buffer
    cl_mem GPUBuffer{ nullptr };       // GPU buffer where main computation occurs
    cl_mem outputGPUBuffer{ nullptr }; // Copy from this GPU buffer to CPU output buffer
    if (vkParameters->fftType == C2C)
    {
      // For C2C computation we can do everything in the in-place-computation buffer.
      configuration.bufferNum = 1;
      configuration.bufferStride[0] = configuration.size[0];
      configuration.bufferStride[1] = configuration.bufferStride[0] * configuration.size[1];
      configuration.bufferStride[2] = configuration.bufferStride[1] * configuration.size[2];
      configuration.bufferSize = &configuration.bufferStride[2];
      const uint64_t bufferBytes{ 2UL * vkParameters->PSize * *configuration.bufferSize };
      itkAssertOrThrowMacro(bufferBytes == vkParameters->inputBufferBytes,
                            "CPU and GPU input buffers are of different sizes.");
      itkAssertOrThrowMacro(bufferBytes == vkParameters->outputBufferBytes,
                            "CPU and GPU output buffers are of different sizes.");
      GPUBuffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferBytes, nullptr, &resCL);
      inputGPUBuffer = GPUBuffer;
      outputGPUBuffer = GPUBuffer;
      if (resCL != CL_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_ALLOCATE;
      configuration.buffer = &GPUBuffer;
    }
    else
    {
      // Either R2HH or R2FH computation. Either forward or inverse.
      configuration.bufferNum = 1;
      if (vkParameters->fftType == R2HH)
      {
        // R2HH computation, either forward or inverse.
        configuration.bufferStride[0] = configuration.size[0] / 2 + 1;
      }
      else
      {
        // R2FH computation, either forward or inverse.
        configuration.bufferStride[0] = configuration.size[0];
      }
      configuration.bufferStride[1] = configuration.bufferStride[0] * configuration.size[1];
      configuration.bufferStride[2] = configuration.bufferStride[1] * configuration.size[2];
      configuration.bufferSize = &configuration.bufferStride[2];
      const uint64_t bufferBytes{ 2UL * vkParameters->PSize * *configuration.bufferSize };
      GPUBuffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferBytes, nullptr, &resCL);
      if (resCL != CL_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_ALLOCATE;
      configuration.buffer = &GPUBuffer;

      if (vkParameters->I == FORWARD)
      {
        // Either R2FH or R2HH.  For forward computation, we have a smaller input buffer.
        configuration.isInputFormatted = 1;
        configuration.inputBufferNum = 1;
        configuration.inputBufferStride[0] = configuration.size[0];
        configuration.inputBufferStride[1] = configuration.inputBufferStride[0] * configuration.size[1];
        configuration.inputBufferStride[2] = configuration.inputBufferStride[1] * configuration.size[2];
        configuration.inputBufferSize = &configuration.inputBufferStride[2];
        const uint64_t inputBufferBytes{ 1UL * vkParameters->PSize * *configuration.inputBufferSize };
        itkAssertOrThrowMacro(inputBufferBytes == vkParameters->inputBufferBytes,
                              "CPU and GPU input buffers are of different sizes.");
        itkAssertOrThrowMacro(bufferBytes == vkParameters->outputBufferBytes,
                              "CPU and GPU output buffers are of different sizes.");
        inputGPUBuffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, inputBufferBytes, nullptr, &resCL);
        outputGPUBuffer = GPUBuffer;
        if (resCL != CL_SUCCESS)
          return VKFFT_ERROR_FAILED_TO_ALLOCATE;
        configuration.inputBuffer = &inputGPUBuffer;
      }
      else
      {
        // Either R2FH or R2HH.  For inverse computation, we have a smaller output buffer.
        configuration.isOutputFormatted = 1;
        configuration.outputBufferNum = 1;
        configuration.outputBufferStride[0] = configuration.size[0];
        configuration.outputBufferStride[1] = configuration.outputBufferStride[0] * configuration.size[1];
        configuration.outputBufferStride[2] = configuration.outputBufferStride[1] * configuration.size[2];
        configuration.outputBufferSize = &configuration.outputBufferStride[2];
        uint64_t outputBufferBytes{ 1UL * vkParameters->PSize * *configuration.outputBufferSize };
        itkAssertOrThrowMacro(bufferBytes == vkParameters->inputBufferBytes,
                              "CPU and GPU input buffers are of different sizes.");
        itkAssertOrThrowMacro(outputBufferBytes == vkParameters->outputBufferBytes,
                              "CPU and GPU output buffers are of different sizes.");
        inputGPUBuffer = GPUBuffer;
        outputGPUBuffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, outputBufferBytes, nullptr, &resCL);
        if (resCL != CL_SUCCESS)
          return VKFFT_ERROR_FAILED_TO_ALLOCATE;
        configuration.outputBuffer = &outputGPUBuffer;
      }
    }

    // Copy input from CPU to GPU
    resCL = clEnqueueWriteBuffer(vkGPU->commandQueue,
                                 inputGPUBuffer,
                                 CL_TRUE,
                                 0,
                                 vkParameters->inputBufferBytes,
                                 vkParameters->inputCPUBuffer,
                                 0,
                                 nullptr,
                                 nullptr);
    if (resCL != CL_SUCCESS)
      return VKFFT_ERROR_FAILED_TO_COPY;

    // Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration
    // file. No buffer allocations inside VkFFT library.
    resFFT = initializeVkFFT(&app, configuration);
    if (resFFT != VKFFT_SUCCESS)
      return resFFT;

    // Submit FFT or iFFT.
    VkFFTLaunchParams launchParams{};
    launchParams.commandQueue = &vkGPU->commandQueue;
    launchParams.inputBuffer = configuration.inputBuffer;
    launchParams.buffer = configuration.buffer;
    launchParams.outputBuffer = configuration.outputBuffer;

    resFFT = VkFFTAppend(&app, vkParameters->I == INVERSE ? 1 : -1, &launchParams);
    if (resFFT != VKFFT_SUCCESS)
      return resFFT;
    resCL = clFinish(vkGPU->commandQueue);
    if (resCL != CL_SUCCESS)
      return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;

    // Copy result from GPU to CPU
    resCL = clEnqueueReadBuffer(vkGPU->commandQueue,
                                outputGPUBuffer,
                                CL_TRUE,
                                0,
                                vkParameters->outputBufferBytes,
                                vkParameters->outputCPUBuffer,
                                0,
                                nullptr,
                                nullptr);
    if (resCL != CL_SUCCESS)
      return VKFFT_ERROR_FAILED_TO_COPY;

    clReleaseMemObject(inputGPUBuffer);
    if (vkParameters->fftType != C2C)
    {
      // Release other buffer too
      clReleaseMemObject(outputGPUBuffer);
    }

    if (vkParameters->fftType == R2FH && vkParameters->I == FORWARD)
    {
      // Compute complex conjugates for the R2FH forward computation
      switch (vkParameters->P)
      {
        case FLOAT:
        {
          using ComplexType = std::complex<float>;
          ComplexType * const outputCPUFloat{ reinterpret_cast<ComplexType *>(vkParameters->outputCPUBuffer) };
          for (uint64_t z = 0; z < configuration.size[2]; ++z)
          {
            for (uint64_t y = 0; y < configuration.size[1]; ++y)
            {
              const uint64_t offsetStart{ z * configuration.bufferStride[1] + y * configuration.bufferStride[0] };
              const uint64_t offsetEnd{ offsetStart + configuration.bufferStride[0] };
              for (uint64_t x = (configuration.size[0] - 1) / 2; x >= 1; --x)
              {
                outputCPUFloat[offsetEnd - x] = std::conj(outputCPUFloat[offsetStart + x]);
              }
            }
          }
        }
        break;
        case DOUBLE:
        {
          using ComplexType = std::complex<double>;
          ComplexType * const outputCPUDouble{ reinterpret_cast<ComplexType *>(vkParameters->outputCPUBuffer) };
          for (uint64_t z = 0; z < configuration.size[2]; ++z)
          {
            for (uint64_t y = 0; y < configuration.size[1]; ++y)
            {
              const uint64_t offsetStart{ z * configuration.bufferStride[1] + y * configuration.bufferStride[0] };
              const uint64_t offsetEnd{ offsetStart + configuration.bufferStride[0] };
              for (uint64_t x = (configuration.size[0] - 1) / 2; x >= 1; --x)
              {
                outputCPUDouble[offsetEnd - x] = std::conj(outputCPUDouble[offsetStart + x]);
              }
            }
          }
        }
        break;
      } // end switch (vkParameters->P)
    }   // end if(vkParameters->fftType == R2FH && vkParameters->I == FORWARD)
    deleteVkFFT(&app);
  }

  // Return to launchVkFFT code
  resCL = clReleaseCommandQueue(vkGPU->commandQueue);
  if (resCL != CL_SUCCESS)
    return VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
  clReleaseContext(vkGPU->context);

  return resFFT;
}


} // end namespace itk
