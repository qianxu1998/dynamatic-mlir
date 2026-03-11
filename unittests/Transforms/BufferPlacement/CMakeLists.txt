add_executable(
  test-cfdfc-cache
  CFDFCCacheTest.cpp
)

target_link_libraries(
  test-cfdfc-cache
  PUBLIC
  GTest::gtest_main
  LLVMSupport
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  DynamaticBufferPlacement
  DynamaticHandshake
  DynamaticSupport
)

enable_testing()

include(GoogleTest)

gtest_discover_tests(
  test-cfdfc-cache
  XML_OUTPUT_DIR ${CMAKE_BINARY_DIR}/unittests/Transforms/BufferPlacement/results
)

add_custom_target(
  run-cfdfc-cache-test
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --parallel 8 --timeout 1500 --output-on-failure
  DEPENDS test-cfdfc-cache
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/Transforms/BufferPlacement
  COMMENT "Run unit tests for the CFDFC cache support."
  VERBATIM
  USES_TERMINAL
)
