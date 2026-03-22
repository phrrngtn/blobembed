cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()
project(blobembed_duckdb LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_STANDARD 11)

# Sources are symlinked into this directory from the parent project
set(BE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

include(FetchContent)

# ── llama.cpp ────────────────────────────────────────────────────────
FetchContent_Declare(llama_cpp
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
  GIT_TAG        b5200
  GIT_SHALLOW    TRUE)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

if(APPLE)
  set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(llama_cpp)

# ── Sources ──────────────────────────────────────────────────────────
set(BE_SOURCES ${BE_ROOT}/src/blobembed_core.cpp)
set(BE_INCLUDES ${BE_ROOT}/include ${llama_cpp_SOURCE_DIR}/include)
set(BE_LIBS llama)

# ── DuckDB C API headers ─────────────────────────────────────────────
FetchContent_Declare(duckdb_capi
  GIT_REPOSITORY https://github.com/duckdb/extension-template-c.git
  GIT_TAG        main
  GIT_SHALLOW    TRUE)
FetchContent_Populate(duckdb_capi)
set(DUCKDB_CAPI_INCLUDE "${duckdb_capi_SOURCE_DIR}/duckdb_capi")

# ── DuckDB extension ─────────────────────────────────────────────────
add_library(blobembed_duckdb MODULE
  ${BE_ROOT}/duckdb_ext/src/blobembed_duckdb.c ${BE_SOURCES})
target_include_directories(blobembed_duckdb PRIVATE
  ${BE_INCLUDES} ${DUCKDB_CAPI_INCLUDE})
target_link_libraries(blobembed_duckdb PRIVATE ${BE_LIBS})
set_target_properties(blobembed_duckdb PROPERTIES
  OUTPUT_NAME "blobembed"
  PREFIX "")
if(APPLE)
  target_link_options(blobembed_duckdb PRIVATE -undefined dynamic_lookup)
endif()

# Append DuckDB extension metadata
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  add_custom_command(TARGET blobembed_duckdb POST_BUILD
    COMMAND ${Python3_EXECUTABLE}
      "${BE_ROOT}/duckdb_ext/append_metadata.py"
      "$<TARGET_FILE:blobembed_duckdb>"
      "$<TARGET_FILE_DIR:blobembed_duckdb>/blobembed.duckdb_extension"
    COMMENT "Appending DuckDB extension metadata")
endif()

# Install the .duckdb_extension into the Python package directory
install(FILES "$<TARGET_FILE_DIR:blobembed_duckdb>/blobembed.duckdb_extension"
  DESTINATION blobembed_duckdb)
