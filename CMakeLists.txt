cmake_minimum_required(VERSION 3.20)
project(blobembed LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_STANDARD 11)

include(FetchContent)

# ── llama.cpp ────────────────────────────────────────────────────────
FetchContent_Declare(llama_cpp
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
  GIT_TAG        b5200
  GIT_SHALLOW    TRUE)

# Disable everything we don't need
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Use Metal on macOS for GPU acceleration
if(APPLE)
  set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(llama_cpp)

# ── Core sources (compiled into each target) ─────────────────────────
set(BE_SOURCES src/blobembed_core.cpp)
set(BE_INCLUDES include ${llama_cpp_SOURCE_DIR}/include)
set(BE_LIBS llama)

# ── Python bindings (nanobind) ───────────────────────────────────────
option(BUILD_PYTHON_BINDINGS "Build nanobind Python module" OFF)
if(BUILD_PYTHON_BINDINGS)
  find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(blobembed_ext NB_STATIC python/bindings.cpp ${BE_SOURCES})
  target_include_directories(blobembed_ext PRIVATE ${BE_INCLUDES})
  target_link_libraries(blobembed_ext PRIVATE ${BE_LIBS})
  set_target_properties(blobembed_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/blobembed")
  install(TARGETS blobembed_ext LIBRARY DESTINATION blobembed)
endif()

# ── DuckDB C API headers (for extension build) ──────────────────────
FetchContent_Declare(duckdb_capi
  GIT_REPOSITORY https://github.com/duckdb/extension-template-c.git
  GIT_TAG        main
  GIT_SHALLOW    TRUE)
FetchContent_Populate(duckdb_capi)
set(DUCKDB_CAPI_INCLUDE "${duckdb_capi_SOURCE_DIR}/duckdb_capi")

# ── DuckDB extension ────────────────────────────────────────────────
option(BUILD_DUCKDB_EXTENSION "Build DuckDB loadable extension" OFF)
if(BUILD_DUCKDB_EXTENSION)
  add_library(blobembed_duckdb MODULE
    duckdb_ext/src/blobembed_duckdb.c ${BE_SOURCES})
  target_include_directories(blobembed_duckdb PRIVATE
    ${BE_INCLUDES} ${DUCKDB_CAPI_INCLUDE})
  target_link_libraries(blobembed_duckdb PRIVATE ${BE_LIBS})
  set_target_properties(blobembed_duckdb PROPERTIES
    OUTPUT_NAME "blobembed"
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/duckdb")
  if(APPLE)
    target_link_options(blobembed_duckdb PRIVATE -undefined dynamic_lookup)
  endif()

  # Append DuckDB extension metadata
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_FOUND)
    add_custom_command(TARGET blobembed_duckdb POST_BUILD
      COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_SOURCE_DIR}/duckdb_ext/append_metadata.py"
        "$<TARGET_FILE:blobembed_duckdb>"
        "${CMAKE_BINARY_DIR}/duckdb/blobembed.duckdb_extension"
      COMMENT "Appending DuckDB extension metadata")
  endif()
endif()

# ── SQLite extension ────────────────────────────────────────────────
option(BUILD_SQLITE_EXTENSION "Build SQLite loadable extension" OFF)
if(BUILD_SQLITE_EXTENSION)
  FetchContent_Declare(sqlite3
    URL "https://www.sqlite.org/2024/sqlite-amalgamation-3460100.zip"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(sqlite3)

  if(NOT TARGET sqlite3_lib)
    add_library(sqlite3_lib STATIC ${sqlite3_SOURCE_DIR}/sqlite3.c)
    target_include_directories(sqlite3_lib PUBLIC ${sqlite3_SOURCE_DIR})
    target_compile_definitions(sqlite3_lib PRIVATE SQLITE_CORE)
  endif()

  add_library(blobembed_sqlite MODULE
    sqlite_ext/src/blobembed_sqlite.c ${BE_SOURCES})
  target_include_directories(blobembed_sqlite PRIVATE
    ${BE_INCLUDES} ${sqlite3_SOURCE_DIR})
  target_link_libraries(blobembed_sqlite PRIVATE ${BE_LIBS})
  set_target_properties(blobembed_sqlite PROPERTIES
    OUTPUT_NAME "blobembed"
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sqlite")
  if(APPLE)
    target_link_options(blobembed_sqlite PRIVATE -undefined dynamic_lookup)
    add_custom_command(TARGET blobembed_sqlite POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink
        blobembed.so "$<TARGET_FILE_DIR:blobembed_sqlite>/blobembed.dylib")
  endif()
endif()
