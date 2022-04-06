ITKVkFFTBackend
=================================

.. image:: https://github.com/InsightSoftwareConsortium/ITKVkFFTBackend/workflows/Build,%20test,%20package/badge.svg
    :alt:    Build Status

.. image:: https://img.shields.io/pypi/v/itk-vkfft.svg
    :target: https://pypi.python.org/pypi/itk-vkfft
    :alt: PyPI Version

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://github.com/InsightSoftwareConsortium/ITKVkFFTBackend/blob/master/LICENSE
    :alt: License

Overview
--------

VkFFT backends for ITK FFT classes.

ITK is an open-source, cross-platform library that provides developers with an extensive suite of software tools for image analysis. Developed through extreme programming methodologies, ITK employs leading-edge algorithms for registering and segmenting multidimensional scientific images.

VkFFT is an efficient GPU-accelerated multidimensional Fast Fourier Transform library for Vulkan/CUDA/HIP/OpenCL projects. VkFFT aims to provide the community with an open-source alternative to Nvidia's cuFFT library while achieving better performance. VkFFT is written in C language and supports Vulkan, CUDA, HIP and OpenCL as backends.

ITKVkFFTBackend enables ITK objects to perform accelerated FFT via the VkFFT library.

VkFFT source code is available at https://github.com/DTolm/VkFFT
