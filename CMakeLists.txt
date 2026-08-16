# nnc — Neural Net Compiler
#
# Single binary, built from every .cpp in src/. The two OS backends
# (sys_win.cpp / sys_linux.cpp) are both compiled on both platforms; each is
# #if-guarded so the "wrong" one produces an empty object file.
#
#   cmake --preset release && cmake --build --preset release   -> exe/nnc[.exe]
#   cmake --preset debug   && cmake --build --preset debug     -> exe/nnc-d[.exe]
#
# Requires: CMake 3.21+, Ninja, and MSVC v143+ / g++ 10+ / clang++ 12+.
# Target CPU baseline is x86-64 with AVX2 + FMA.

cmake_minimum_required(VERSION 3.21)

project(nnc LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(NNC_SOURCES
	src/emitter_avx2.cpp
	src/emitter_x64.cpp
	src/gemma.cpp
	src/gguf.cpp
	src/jit_buffer.cpp
	src/jit_kernel.cpp
	src/jit_ops.cpp
	src/main.cpp
	src/nn_ops.cpp
	src/runtime.cpp
	src/sys_linux.cpp
	src/sys_win.cpp
	src/tests.cpp
	src/utils.cpp
)

add_executable(nnc ${NNC_SOURCES})

target_include_directories(nnc PRIVATE src)
target_link_libraries(nnc PRIVATE Threads::Threads)

# Debug builds are nnc-d so both binaries can live in exe/ side by side.
set_target_properties(nnc PROPERTIES
	OUTPUT_NAME nnc
	OUTPUT_NAME_DEBUG nnc-d
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/exe"
)
foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
	string(TOUPPER "${cfg}" cfg_upper)
	set_target_properties(nnc PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_SOURCE_DIR}/exe")
endforeach()

if(MSVC)
	target_compile_definitions(nnc PRIVATE
		_CRT_SECURE_NO_WARNINGS
		_CONSOLE
		UNICODE
		_UNICODE
		$<$<CONFIG:Debug>:_DEBUG>
	)
	target_compile_options(nnc PRIVATE
		/W3
		/permissive-
		/Zc:__cplusplus
		/utf-8       # source files contain UTF-8 literals (U+2581, µ)
		/arch:AVX2
		/fp:fast
		/Zi
		$<$<NOT:$<CONFIG:Debug>>:/O2;/Oi;/Ot;/Gy;/GL>
	)
	target_link_options(nnc PRIVATE
		/DEBUG
		$<$<NOT:$<CONFIG:Debug>>:/LTCG;/OPT:REF;/OPT:ICF>
	)
else()
	target_compile_options(nnc PRIVATE
		-Wall -Wextra
		-Wno-unused-parameter
		-Wno-missing-field-initializers
		-mavx2 -mfma -mf16c -mxsave
		$<$<CONFIG:Debug>:-O0;-g>
		$<$<NOT:$<CONFIG:Debug>>:-O3;-ffast-math;-fno-finite-math-only>
	)
	target_compile_definitions(nnc PRIVATE $<$<CONFIG:Debug>:_DEBUG>)
endif()

enable_testing()
add_test(NAME nnc-selftest COMMAND nnc --test)
