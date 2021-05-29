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
#ifndef itkVkCommon_h
#define itkVkCommon_h

#include "vkFFT.h"

// Remove me!!!
#define VKFFTBACKEND_HEREIAM                                                                                           \
  do                                                                                                                   \
  {                                                                                                                    \
    std::cout << "func: " << __func__ << " (" __FILE__ "::" << __LINE__ << ")" << std::endl << std::flush;             \
  } while (false)


namespace itk
{
class VkCommon
{
public:
  enum PrecisionType
  {
    FLOAT = 0,
    DOUBLE = 1,
    /* HALF = 2 */
  };

  enum FFTType
  {
    C2C = 0,     // Complex to Complex
    R2HalfH = 1, // Real to Half Hermetian
    R2FullH = 2  // Real to Full Hermetian (aka Complex)
  };

  enum DirectionType
  {
    FORWARD = -1,
    INVERSE = 1
  };

  enum NormalizationType
  {
    UNNORMALIZED = 0,
    NORMALIZED = 1
  };

  struct VkGPU
  {
    cl_platform_id   platform = 0;
    cl_device_id     device = 0;
    cl_context       context = 0;
    cl_command_queue commandQueue = 0;
    uint64_t         device_id = 0; // default value
  };

  struct VkParameters
  {
    uint64_t          X = 0;         // size of fastest varying dimension
    uint64_t          Y = 1;         // size of second-fastest varying dimension, if any, otherwise 1.
    uint64_t          Z = 1;         // size of third-fastest varying dimension, if any, otherwise 1.
    PrecisionType     P = FLOAT;     // type for real numbers
    const uint64_t    B = 1;         // Number of batches -- always 1
    const uint64_t    N = 1;         // Number of redundant iterations, for benchmarking -- always 1.
    FFTType           fftType = C2C; // ComplexToComplex, RealToHalfHermetian, RealToFullHermetian
    uint64_t          PSize = 4;     // sizeof(float), sizeof(double), or sizeof(half) according to VkParameters.P.
    DirectionType     I = FORWARD;   // forward or inverse transformation. (R2HalfH inverse is aka HalfH2R, etc.)
    NormalizationType normalized = UNNORMALIZED; // Whether inverse transformation should be divided by array size
    const void *      inputCPUBuffer = 0;        // input buffer in CPU memory
    uint64_t          inputBufferBytes = 0;      // number of bytes in inputCPUBuffer
    void *            outputCPUBuffer = 0;       // output buffer in CPU memory
    uint64_t          outputBufferBytes = 0;     // number of bytes in outputCPUBuffer
  };

  static VkFFTResult
  run(VkGPU * vkGPU, const VkParameters * vkParameters);

  static constexpr uint64_t GreatestPrimeFactor{ 13 };
};

} // namespace itk
#endif
