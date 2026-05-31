CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17
LDFLAGS ?= -lm

TARGET := infer
SRCS := infer.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
