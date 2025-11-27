include ../../tools/sharcfx/sharcfx.mk

BUILD_DIR := ./build
LIB_SRCS := .
LIB := libadi_sharcfx_nn.a
TARGET_LIB_DIR := $(LIB_SRCS)/../Lib

INCLUDES += \
	-I"../Include" \
	-I"../../cces/examples/shared"

CPP_SRCS += \
	$(wildcard $(LIB_SRCS)/src/*.cpp)

include ../../tools/sharcfx/lib.mk