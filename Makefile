# nnc — cross-platform Makefile (Linux / WSL).
#
# Mirrors the MSVC project: one `nnc` (release) or `nnc-d` (debug) binary
# built from every .cpp under src/. Windows-only TUs are #if-guarded so
# they compile to nothing on Linux (and vice-versa) — we still pass them
# to the compiler so the file list stays in lockstep with the .vcxproj.
#
# Usage:
#   make                  # release build  -> exe/nnc
#   make debug            # debug build    -> exe/nnc-d
#   make test             # build debug + run --test
#   make clean
#
# Requirements: g++ 10+ (or clang++ 12+) on x86-64 with AVX2 + FMA + F16C.

CXX      ?= g++
CONFIG   ?= release

SRC_DIR  := src
OUT_DIR  := exe
OBJ_ROOT := intermediate

SRCS := $(wildcard $(SRC_DIR)/*.cpp)

CXXSTD   := -std=c++20
WARN     := -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers
ARCH     := -mavx2 -mfma -mf16c -mxsave
COMMON   := $(CXXSTD) $(WARN) $(ARCH) -pthread

ifeq ($(CONFIG),debug)
  CXXFLAGS := $(COMMON) -O0 -g -D_DEBUG
  TARGET   := $(OUT_DIR)/nnc-d
  OBJ_DIR  := $(OBJ_ROOT)/Debug/linux
else
  CXXFLAGS := $(COMMON) -O3 -DNDEBUG -ffast-math -fno-finite-math-only
  TARGET   := $(OUT_DIR)/nnc
  OBJ_DIR  := $(OBJ_ROOT)/Release/linux
endif

LDFLAGS  := -pthread

OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

.PHONY: all debug release test clean

all: $(TARGET)

release:
	$(MAKE) CONFIG=release

debug:
	$(MAKE) CONFIG=debug

$(TARGET): $(OBJS) | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@

$(OUT_DIR) $(OBJ_DIR):
	mkdir -p $@

test: debug
	$(OUT_DIR)/nnc-d --test

clean:
	rm -rf $(OBJ_ROOT)/*/linux $(OUT_DIR)/nnc $(OUT_DIR)/nnc-d
