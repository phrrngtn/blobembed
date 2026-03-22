cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()
project(blobembed_sqlite LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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

# ── SQLite amalgamation ──────────────────────────────────────────────
FetchContent_Declare(sqlite3
  URL "https://www.sqlite.org/2024/sqlite-amalgamation-3460100.zip"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(sqlite3)

# ── Sources ──────────────────────────────────────────────────────────
set(BE_SOURCES ${BE_ROOT}/src/blobembed_core.cpp)
set(BE_INCLUDES ${BE_ROOT}/include ${llama_cpp_SOURCE_DIR}/include)
set(BE_LIBS llama)

# ── SQLite extension ─────────────────────────────────────────────────
add_library(blobembed_sqlite MODULE
  ${BE_ROOT}/sqlite_ext/src/blobembed_sqlite.c ${BE_SOURCES})
target_include_directories(blobembed_sqlite PRIVATE
  ${BE_INCLUDES} ${sqlite3_SOURCE_DIR})
target_link_libraries(blobembed_sqlite PRIVATE ${BE_LIBS})
set_target_properties(blobembed_sqlite PROPERTIES
  OUTPUT_NAME "blobembed"
  PREFIX "")
if(APPLE)
  target_link_options(blobembed_sqlite PRIVATE -undefined dynamic_lookup)
endif()

# Install the extension into the Python package directory
install(TARGETS blobembed_sqlite LIBRARY DESTINATION blobembed_sqlite)
if(APPLE)
  install(CODE "
    execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
      blobembed.so \"\${CMAKE_INSTALL_PREFIX}/blobembed_sqlite/blobembed.dylib\")
  ")
endif()
