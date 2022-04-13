# ==========================================================================
#
#   Copyright NumFOCUS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

# Verify that object factory initialization succeeds on itk module loading
# such that ITK VkFFT classes are instantiated as the default FFT image filter
# implementation through the object factory backend

import itk

itk.auto_progress(2)

real_type = itk.F
dimension = 3
real_image_type = itk.Image[real_type, dimension]
complex_image_type = itk.Image[itk.complex[real_type], dimension]

# Verify all FFT base filter types are instantiated with VkFFT accelerated backend
image_filter_list = [
    (
        itk.ComplexToComplex1DFFTImageFilter[complex_image_type],
        itk.VkComplexToComplex1DFFTImageFilter[complex_image_type],
    ),
    (
        itk.ComplexToComplexFFTImageFilter[complex_image_type],
        itk.VkComplexToComplexFFTImageFilter[complex_image_type],
    ),
    (
        itk.HalfHermitianToRealInverseFFTImageFilter[
            complex_image_type, real_image_type
        ],
        itk.VkHalfHermitianToRealInverseFFTImageFilter[complex_image_type],
    ),
    (
        itk.Forward1DFFTImageFilter[real_image_type],
        itk.VkForward1DFFTImageFilter[real_image_type],
    ),
    (
        itk.ForwardFFTImageFilter[real_image_type, complex_image_type],
        itk.VkForwardFFTImageFilter[real_image_type],
    ),
    (
        itk.Inverse1DFFTImageFilter[complex_image_type],
        itk.VkInverse1DFFTImageFilter[complex_image_type],
    ),
    (
        itk.InverseFFTImageFilter[complex_image_type, real_image_type],
        itk.VkInverseFFTImageFilter[complex_image_type],
    ),
    (
        itk.RealToHalfHermitianForwardFFTImageFilter[
            real_image_type, complex_image_type
        ],
        itk.VkRealToHalfHermitianForwardFFTImageFilter[real_image_type],
    ),
]

for (base_filter_type, vk_filter_type) in image_filter_list:
    # Instantiate through the ITK object factory
    image_filter = base_filter_type.New()
    assert image_filter is not None

    try:
        print(
            f"Instantiated default FFT image filter backend {image_filter.GetNameOfClass()}"
        )
        # Verify object can be cast to ITK VkFFT filter type
        vk_filter_type.cast(image_filter)
    except RuntimeError as e:
        print(f"ITK VkFFT filter was not instantiated as default backend!")
        raise e
