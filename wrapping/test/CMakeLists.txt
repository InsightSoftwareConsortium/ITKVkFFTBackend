if(ITK_WRAP_PYTHON)
  itk_python_add_test(NAME itkVkFFTInitFactoryPythonTest
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/itkVkFFTInitFactoryTest.py
  )
endif()
