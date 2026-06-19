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
#ifndef itkVkCommon_h
#define itkVkCommon_h

#include "VkFFTBackendExport.h"
#include "itkVkDefinitions.h"
#include "itkDataObject.h"
#if (VKFFT_BACKEND == LEVEL_ZERO)
#  include <level_zero/ze_api.h>
#endif
#if (VKFFT_BACKEND == METAL)
// Include metal-cpp headers BEFORE vkFFT.h. vkFFT.h internally defines
// NS_/MTL_/CA_PRIVATE_IMPLEMENTATION and re-includes these — pre-including
// here primes the header guards so vkFFT.h's re-include is a no-op, and
// the metal-cpp storage symbols are emitted in only one TU (itkVkCommon.cxx,
// which sets the *_PRIVATE_IMPLEMENTATION macros itself before this point).
#  include "Foundation/Foundation.hpp"
#  include "Metal/Metal.hpp"
#  include "QuartzCore/QuartzCore.hpp"
#endif
#include "vkFFT.h"

namespace itk
{

class VkFFTBackend_EXPORT VkCommon
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkCommon);

  /** Standard class typedefs. */
  using Self = VkCommon;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  enum class PrecisionEnum
  {
    FLOAT = 0,
    DOUBLE = 1,
    /* HALF = 2 */
  };

  enum class FFTEnum
  {
    C2C = 0,     // Complex to Complex
    R2HalfH = 1, // Real to Half Hermetian
    R2FullH = 2  // Real to Full Hermetian (aka Complex)
  };

  enum class DirectionEnum
  {
    FORWARD = -1,
    INVERSE = 1
  };

  enum class NormalizationEnum
  {
    UNNORMALIZED = 0,
    NORMALIZED = 1
  };

  struct VkParameters
  {
    uint64_t X{ 0 }; // size of fastest varying dimension
    uint64_t Y{ 1 }; // size of second-fastest varying dimension, if any, otherwise 1.
    uint64_t Z{ 1 }; // size of third-fastest varying dimension, if any, otherwise 1.
    uint64_t omitDimension[3] = { 0,
                                  0,
                                  0 }; // disable FFT for this dimension (0 - FFT enabled, 1 - FFT disabled). Default 0.
                                       // Doesn't work for R2C dimension 0 for now. Doesn't work with convolutions.
    PrecisionEnum P = PrecisionEnum::FLOAT; // type for real numbers
    uint64_t      B{ 1 };                   // Number of batches -- always 1
    uint64_t      N{ 1 };                   // Number of redundant iterations, for benchmarking -- always 1.
    FFTEnum       fft{ FFTEnum::C2C };      // ComplexToComplex, RealToHalfHermetian, RealToFullHermetian
    uint64_t      PSize{ 4 }; // sizeof(float), sizeof(double), or sizeof(half) according to VkParameters.P.
    DirectionEnum I{
      DirectionEnum::FORWARD
    }; // forward or inverse transformation. (R2HalfH inverse is aka HalfH2R, etc.)
    NormalizationEnum normalized{
      NormalizationEnum::UNNORMALIZED
    }; // Whether inverse transformation should be divided by array size
    const void * inputCPUBuffer{ nullptr };  // input buffer in CPU memory
    uint64_t     inputBufferBytes{ 0 };      // number of bytes in inputCPUBuffer
    void *       outputCPUBuffer{ nullptr }; // output buffer in CPU memory
    uint64_t     outputBufferBytes{ 0 };     // number of bytes in outputCPUBuffer

    bool
    operator!=(const VkParameters & rhs) const
    {
      return this->X != rhs.X || this->Y != rhs.Y || this->Z != rhs.Z || this->P != rhs.P || this->B != rhs.B ||
             this->N != rhs.N || this->fft != rhs.fft || this->PSize != rhs.PSize || this->I != rhs.I ||
             this->normalized != rhs.normalized || this->inputCPUBuffer != rhs.inputCPUBuffer ||
             this->inputBufferBytes != rhs.inputBufferBytes || this->outputCPUBuffer != rhs.outputCPUBuffer ||
             this->outputBufferBytes != rhs.outputBufferBytes;
    }

    /** Compare only the transform-shape fields, ignoring the per-call CPU buffer
     * pointers and byte counts, which change on every call. The VkFFT plan depends
     * only on the shape, so a cached plan is reusable across calls that differ
     * only in their buffers. */
    bool
    SameShapeAs(const VkParameters & rhs) const
    {
      return this->X == rhs.X && this->Y == rhs.Y && this->Z == rhs.Z && this->P == rhs.P && this->B == rhs.B &&
             this->N == rhs.N && this->fft == rhs.fft && this->PSize == rhs.PSize && this->I == rhs.I &&
             this->normalized == rhs.normalized && this->omitDimension[0] == rhs.omitDimension[0] &&
             this->omitDimension[1] == rhs.omitDimension[1] && this->omitDimension[2] == rhs.omitDimension[2];
    }
  };

  struct VkGPU
  {
#if (VKFFT_BACKEND == CUDA)
    CUdevice  device{ 0 };
    CUcontext context{ 0 };
#elif (VKFFT_BACKEND == OPENCL)
    cl_platform_id   platform{ 0 };
    cl_device_id     device{ 0 };
    cl_context       context{ 0 };
    cl_command_queue commandQueue{ 0 };
#elif (VKFFT_BACKEND == LEVEL_ZERO)
    ze_driver_handle_t        driver{ nullptr };
    ze_device_handle_t        device{ nullptr };
    ze_context_handle_t       context{ nullptr };
    ze_command_queue_handle_t commandQueue{ nullptr };
    uint32_t                  commandQueueID{ 0 };
#elif (VKFFT_BACKEND == METAL)
    MTL::Device *       device{ nullptr };
    MTL::CommandQueue * queue{ nullptr };
#endif
    uint64_t device_id{ 0 }; // default value

    bool
    operator!=(const VkGPU & rhs) const
    {
#if (VKFFT_BACKEND == CUDA)
      return this->device != rhs.device || this->context != rhs.context || this->device_id != rhs.device_id;
#elif (VKFFT_BACKEND == OPENCL)
      return this->platform != rhs.platform || this->device != rhs.device || this->context != rhs.context ||
             this->commandQueue != rhs.commandQueue || this->device_id != rhs.device_id;
#elif (VKFFT_BACKEND == LEVEL_ZERO)
      return this->driver != rhs.driver || this->device != rhs.device || this->context != rhs.context ||
             this->commandQueue != rhs.commandQueue || this->commandQueueID != rhs.commandQueueID ||
             this->device_id != rhs.device_id;
#elif (VKFFT_BACKEND == METAL)
      return this->device != rhs.device || this->queue != rhs.queue || this->device_id != rhs.device_id;
#else
      return this->device_id != rhs.device_id;
#endif
    }
  };

  VkFFTResult
  Run(const VkGPU & vkGPU, const VkParameters & vkParameters);

  VkFFTResult
  ReleaseBackend();

  uint64_t
  GetGreatestPrimeFactor() const
  {
    return 13UL;
  }

  VkCommon() = default;
  ~VkCommon() { this->ReleaseBackend(); }

protected:
  VkFFTResult
  ConfigureBackend();

  VkFFTResult
  PerformFFT();

private:
  // Backend parameters
  VkGPU              m_VkGPU{};
  VkParameters       m_VkParameters{};
  VkFFTConfiguration m_VkFFTConfiguration{};

  // Cached GPU context + plan configuration are (re)built only on first use or when
  // the device/transform shape changes; m_VkGPU and m_VkParameters hold that cached state.
  bool m_MustConfigure{ true };

  // Cached compiled plan and persistent per-shape GPU buffers. initializeVkFFT (which
  // JIT-compiles the FFT kernels) and the buffer allocations run once per shape and are
  // reused across same-shape transforms; per call only the host<->device copies and the
  // VkFFTAppend run. Released together with the context in ReleaseBackend().
  //
  // The VkFFTApplication is heap-allocated (not an inline member) so that its size is
  // computed in the library translation unit that actually populates it; embedding it
  // by value makes the class layout depend on sizeof(VkFFTApplication) at every include
  // site, which can differ and corrupt the members that follow.
  VkFFTApplication * m_VkFFTApplication{ nullptr };
  bool               m_PlanConfigured{ false };
#if (VKFFT_BACKEND == CUDA)
  cuFloatComplex * m_GPUBuffer{ nullptr };
  cuFloatComplex * m_InputGPUBuffer{ nullptr };
  cuFloatComplex * m_OutputGPUBuffer{ nullptr };
#elif (VKFFT_BACKEND == OPENCL)
  cl_mem m_GPUBuffer{ nullptr };
  cl_mem m_InputGPUBuffer{ nullptr };
  cl_mem m_OutputGPUBuffer{ nullptr };
#elif (VKFFT_BACKEND == LEVEL_ZERO)
  void * m_GPUBuffer{ nullptr };
  void * m_InputGPUBuffer{ nullptr };
  void * m_OutputGPUBuffer{ nullptr };
#elif (VKFFT_BACKEND == METAL)
  MTL::Buffer * m_GPUBuffer{ nullptr };
  MTL::Buffer * m_InputGPUBuffer{ nullptr };
  MTL::Buffer * m_OutputGPUBuffer{ nullptr };
#endif
};

} // namespace itk
#endif
