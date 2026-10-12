CXX ?= g++
CUDA_INSTALL_PATH ?= /usr/local/cuda
OUT := ../../build/tools/tracez_probe
PARSER := ../../src/trace-parser
CPPFLAGS += -I$(PARSER) -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/targets/x86_64-linux/include
CXXFLAGS += -std=c++17 -O2 -Wall -Wextra
LDLIBS += -lzstd -pthread

all: $(OUT)

$(OUT): tracez_probe.cc $(PARSER)/trace_parser.cc $(PARSER)/trace_parser.h $(PARSER)/warp_trace_stream.h $(PARSER)/tracez_format.h
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ tracez_probe.cc $(PARSER)/trace_parser.cc $(LDLIBS)

.PHONY: all
