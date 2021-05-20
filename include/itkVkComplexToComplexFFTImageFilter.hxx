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
#ifndef itkVkComplexToComplexFFTImageFilter_hxx
#define itkVkComplexToComplexFFTImageFilter_hxx

#include "itkVkComplexToComplexFFTImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

namespace itk
{

namespace vkfft
{

#include "vkFFT.h"

} // end namespace vkfft

template <typename TImage>
VkComplexToComplexFFTImageFilter<TImage>::VkComplexToComplexFFTImageFilter()
{
  this->DynamicMultiThreadingOn();
}

template <typename TImage>
void
VkComplexToComplexFFTImageFilter<TImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

template <typename TImage>
void
VkComplexToComplexFFTImageFilter<TImage>::UpdateOutputData(DataObject * output)
{
  // we need to catch that information now, because it is changed later during the pipeline execution,
  // and thus can't be grabbed in GenerateData().
  m_CanUseDestructiveAlgorithm = this->GetInput()->GetReleaseDataFlag();
  Superclass::UpdateOutputData(output);
}

template <typename TImage>
void
VkComplexToComplexFFTImageFilter<TImage>::BeforeThreadedGenerateData()
{
  // get pointers to the input and output
  const InputImageType * const input = this->GetInput();
  OutputImageType * const      output = this->GetOutput();

  if (!input || !output)
  {
    return;
  }

  // we don't have a nice progress to report, but at least this simple line reports the beginning and
  // the end of the process
  ProgressReporter progress(this, 0, 1);

  // allocate output buffer memory
  output->SetBufferedRegion(output->GetRequestedRegion());
  output->Allocate();

  const OutputSizeType & inputSize = input->GetLargestPossibleRegion().GetSize();
  // Use these somewhere!!!
  const int transformDirection = this->GetTransformDirection() == Superclass::TransformDirectionEnum::INVERSE ? -1 : 1;
  const InputPixelType * const in = input->GetBufferPointer();
  OutputPixelType * const      out = output->GetBufferPointer();

  // Create and fill VkFFTConfiguration
  vkfft::VkFFTConfiguration configuration{};

  // !!! static_assert that ImageDimension <= 3 ???  Use concept checking.
  configuration.FFTdim = ImageDimension;

  // Should remaining values of configuration.size be set to zero or to one, or are they safely ignored?!!!
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    configuration.size[(ImageDimension - 1) - i] = inputSize[i];
  }

#if 0
  // !!! Additional members of vkfft::VkFFTConfiguration are:

  // Use clEsperanto if feasible!!!
  vkfft::cl_platform_id* platform;
  vkfft::cl_device_id* device;
  vkfft::cl_context* context;
  vkfft::cl_command_queue* commandQueue;

  //data parameters:
  uint64_t userTempBuffer; //buffer allocated by app automatically if needed to reorder Four step algorithm. Setting to non zero value enables manual user allocation (0 - off, 1 - on)
  uint64_t bufferNum;//multiple buffer sequence storage is Vulkan only. Default 1
  uint64_t tempBufferNum;//multiple buffer sequence storage is Vulkan only. Default 1, buffer allocated by app automatically if needed to reorder Four step algorithm. Setting to non zero value
                         //enables manual user allocation
  uint64_t inputBufferNum;//multiple buffer sequence storage is Vulkan only. Default 1, if isInputFormatted is enabled
  uint64_t outputBufferNum;//multiple buffer sequence storage is Vulkan only. Default 1, if isOutputFormatted is enabled
  uint64_t kernelNum;//multiple buffer sequence storage is Vulkan only. Default 1, if performConvolution is enabled

  uint64_t* bufferSize;//array of buffers sizes in bytes
  uint64_t* tempBufferSize;//array of temp buffers sizes in bytes. Default set to bufferSize sum, buffer allocated by app automatically if needed to reorder Four step algorithm. Setting to non zero
                           //value enables manual user allocation
  uint64_t* inputBufferSize;//array of input buffers sizes in bytes, if isInputFormatted is enabled
  uint64_t* outputBufferSize;//array of output buffers sizes in bytes, if isOutputFormatted is enabled
  uint64_t* kernelSize;//array of kernel buffers sizes in bytes, if performConvolution is enabled

  vkfft::cl_mem* buffer;//pointer to device buffer used for computations
  vkfft::cl_mem* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
  vkfft::cl_mem* inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
  vkfft::cl_mem* outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
  vkfft::cl_mem* kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled

  //optional: (default 0 if not stated otherwise)
  uint64_t coalescedMemory;//in bits, for Nvidia and AMD is equal to 32, Intel is equal 64, scaled for half precision. Gonna work regardles, but if specified by user correctly, the performance will
                           //be higher.
  uint64_t aimThreads;//aim at this many threads per block. Default 128
  uint64_t numSharedBanks;//how many banks shared memory has. Default 32
  uint64_t inverseReturnToInputBuffer;//return data to the input buffer in inverse transform (0 - off, 1 - on). isInputFormatted must be enabled
  uint64_t numberBatches;// N - used to perform multiple batches of initial data. Default 1
  uint64_t useUint64;//use 64-bit addressing mode in generated kernels

  uint64_t doublePrecision; //perform calculations in double precision (0 - off, 1 - on).
  uint64_t halfPrecision; //perform calculations in half precision (0 - off, 1 - on)
  uint64_t halfPrecisionMemoryOnly; //use half precision only as input/output buffer. Input/Output have to be allocated as half, buffer/tempBuffer have to be allocated as float (out of place mode
                                    //only). Specify isInputFormatted and isOutputFormatted to use (0 - off, 1 - on)

  uint64_t performR2C; //perform R2C/C2R decomposition (0 - off, 1 - on)
  uint64_t disableMergeSequencesR2C; //disable merging of two real sequences to reduce calculations (0 - off, 1 - on)
  uint64_t normalize; //normalize inverse transform (0 - off, 1 - on)
  uint64_t disableReorderFourStep; // disables unshuffling of Four step algorithm. Requires tempbuffer allocation (0 - off, 1 - on)
  uint64_t useLUT; //switches from calculating sincos to using precomputed LUT tables (0 - off, 1 - on). Configured by initialization routine
  uint64_t makeForwardPlanOnly; //generate code only for forward FFT (0 - off, 1 - on)
  uint64_t makeInversePlanOnly; //generate code only for inverse FFT (0 - off, 1 - on)

  uint64_t bufferStride[3];//buffer strides - default set to x - x*y - x*y*z values
  uint64_t isInputFormatted; //specify if input buffer is padded - 0 - padded, 1 - not padded. For example if it is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and
                             //numberKernels==1)
  uint64_t isOutputFormatted; //specify if output buffer is padded - 0 - padded, 1 - not padded. For example if it is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and
                              //numberKernels==1)
  uint64_t inputBufferStride[3];//input buffer strides. Used if isInputFormatted is enabled. Default set to bufferStride values
  uint64_t outputBufferStride[3];//output buffer strides. Used if isInputFormatted is enabled. Default set to bufferStride values

  //optional zero padding control parameters: (default 0 if not stated otherwise)
  uint64_t performZeropadding[3]; // don't read some data/perform computations if some input sequences are zeropadded for each axis (0 - off, 1 - on)
  uint64_t fft_zeropad_left[3];//specify start boundary of zero block in the system for each axis
  uint64_t fft_zeropad_right[3];//specify end boundary of zero block in the system for each axis
  uint64_t frequencyZeroPadding; //set to 1 if zeropadding of frequency domain, default 0 - spatial zeropadding

  //optional convolution control parameters: (default 0 if not stated otherwise)
  uint64_t performConvolution; //perform convolution in this application (0 - off, 1 - on). Disables reorderFourStep parameter
  uint64_t coordinateFeatures; // C - coordinate, or dimension of features vector. In matrix convolution - size of vector
  uint64_t matrixConvolution; //if equal to 2 perform 2x2, if equal to 3 perform 3x3 matrix-vector convolution. Overrides coordinateFeatures
  uint64_t symmetricKernel; //specify if kernel in 2x2 or 3x3 matrix convolution is symmetric
  uint64_t numberKernels;// N - only used in convolution step - specify how many kernels were initialized before. Expands one input to multiple (batched) output
  uint64_t kernelConvolution;// specify if this application is used to create kernel for convolution, so it has the same properties. performConvolution has to be set to 0 for kernel creation

  //register overutilization (experimental): (default 0 if not stated otherwise)
  uint64_t registerBoost; //specify if register file size is bigger than shared memory and can be used to extend it X times (on Nvidia 256KB register file can be used instead of 32KB of shared
                          //memory, set this constant to 4 to emulate 128KB of shared memory). Default 1
  uint64_t registerBoostNonPow2; //specify if register overutilization should be used on non power of 2 sequences (0 - off, 1 - on)
  uint64_t registerBoost4Step; //specify if register file overutilization should be used in big sequences (>2^14), same definition as registerBoost. Default 1

  //not used techniques:
  uint64_t swapTo3Stage4Step; //specify at which power of 2 to switch from 2 upload to 3 upload 4-step FFT, in case if making max sequence size lower than coalesced sequence helps to combat TLB
                              //misses. Default 0 - disabled. Must be at least 17
  uint64_t performHalfBandwidthBoost;//try to reduce coalsesced number by a factor of 2 to get bigger sequence in one upload
  uint64_t devicePageSize;//in KB, the size of a page on the GPU. Setting to 0 disables local buffer split in pages
  uint64_t localPageSize;//in KB, the size to split page into if sequence spans multiple devicePageSize pages

  //automatically filled based on device info (still can be reconfigured by user):
  uint64_t maxComputeWorkGroupCount[3]; // maxComputeWorkGroupCount from VkPhysicalDeviceLimits
  uint64_t maxComputeWorkGroupSize[3]; // maxComputeWorkGroupCount from VkPhysicalDeviceLimits
  uint64_t maxThreadsNum; //max number of threads from VkPhysicalDeviceLimits
  uint64_t sharedMemorySizeStatic; //available for static allocation shared memory size, in bytes
  uint64_t sharedMemorySize; //available for allocation shared memory size, in bytes
  uint64_t sharedMemorySizePow2; //power of 2 which is less or equal to sharedMemorySize, in bytes
  uint64_t warpSize; //number of threads per warp/wavefront.
  uint64_t halfThreads;//Intel fix
  uint64_t allocateTempBuffer; //buffer allocated by app automatically if needed to reorder Four step algorithm. Parameter to check if it has been allocated
  uint64_t reorderFourStep; // unshuffle Four step algorithm. Requires tempbuffer allocation (0 - off, 1 - on). Default 1.
#endif

  vkfft::VkFFTLaunchParams launchParams{};
#if 0
  // !!! Members of vkfft::VkFFTLaunchParams are:
  vkfft::cl_command_queue* commandQueue;//commandBuffer to which FFT is appended

  vkfft::cl_mem* buffer;//pointer to device buffer used for computations
  vkfft::cl_mem* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
  vkfft::cl_mem* inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
  vkfft::cl_mem* outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
  vkfft::cl_mem* kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#endif

  // Do something with these?!!!
  vkfft::VkFFTResult                        result{};
  vkfft::VkFFTSpecializationConstantsLayout specializationConstantsLayout{};
  vkfft::VkFFTPushConstantsLayoutUint32     pushConstantsLayoutUint32{};
  vkfft::VkFFTPushConstantsLayoutUint64     pushConstantsLayoutUint64{};
  vkfft::VkFFTAxis                          axis{};
  vkfft::VkFFTPlan                          plan{};
  vkfft::VkFFTApplication                   application{};

  //!!! const int flags = m_CanUseDestructiveAlgorithm ? m_PlanRigor : m_PlanRigor | FFTW_PRESERVE_INPUT;

  // Write me!!!
}

template <typename TImage>
void
VkComplexToComplexFFTImageFilter<TImage>::DynamicThreadedGenerateData(const OutputRegionType & outputRegionForThread)
{
  // Normalize the output if backward transform
  if (this->GetTransformDirection() == Superclass::TransformDirectionEnum::INVERSE)
  {
    using IteratorType = ImageRegionIterator<OutputImageType>;
    const SizeValueType totalOutputSize = this->GetOutput()->GetRequestedRegion().GetNumberOfPixels();
    IteratorType        it(this->GetOutput(), outputRegionForThread);
    for (; !it.IsAtEnd(); ++it)
    {
      OutputPixelType val = it.Value();
      val /= totalOutputSize;
      it.Set(val);
    }
  }
}

} // end namespace itk

#endif // itkVkComplexToComplexFFTImageFilter_hxx
