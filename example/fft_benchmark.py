#!/usr/bin/env python3

# Copyright NumFOCUS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        https://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script for benchmarking accelerated ITK FFT performance

import time

import numpy as np
import itk

itk.auto_progress(2)

pixel_type = itk.F
dimension = 3
image_type = itk.Image[pixel_type, dimension]
complex_image_type = itk.Image[itk.complex[pixel_type], dimension]
fft_filter_type = itk.ForwardFFTImageFilter[image_type, complex_image_type]

# Return time for filter.Update()
def benchmark_fft(itk_fft_filter: fft_filter_type) -> float:
    start = time.time()
    itk_fft_filter.Update()
    end = time.time()
    return end - start


def run_fft(itk_fft_filter_type, image_size: list()) -> float:
    assert len(image_size) == dimension
    fft_filter = itk_fft_filter_type.New()
    image = itk.image_from_array(
        np.random.random_sample(image_size[:3]).astype(np.float32)
    )
    fft_filter.SetInput(image)
    return benchmark_fft(fft_filter)


# One side of 3D image
# Total volume of cube is image_len ^ 3 voxels
image_lens = [10, 30, 100, 200, 300, 600, 800, 1000, 1200, 1500, 2000]

# ITK default CPU implementation
vnl_type = itk.VnlForwardFFTImageFilter[image_type, complex_image_type]
# VkFFT accelerated implementation
vk_type = itk.VkForwardFFTImageFilter[image_type]

itk.auto_progress(0)


def run_experiments():
    print(
        f"Experiment   Image Len (px)\tVolume (px)\tCPU FFT Time (s)\tGPU FFT Time(s)\tRelative Speed"
    )
    for idx, image_len in enumerate(image_lens):
        image_size = [image_len] * 3
        n_iterations = 3

        for iteration in range(n_iterations):
            vnl_interval = run_fft(vnl_type, image_size)
            vk_interval = run_fft(vk_type, image_size)
            relative_speed = vnl_interval / vk_interval
            print(
                f"\t{idx:>2}\t\t{image_len:>5}\t{image_len**3:>.2e}\t\t{vnl_interval:>6f}\t{vk_interval:>6f}\t{relative_speed:>.1%}"
            )


if __name__ == "__main__":
    run_experiments()
