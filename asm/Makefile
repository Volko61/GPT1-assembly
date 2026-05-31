CC ?= gcc
CFLAGS ?= -O3 -march=armv8-a -fno-plt -no-pie
LDFLAGS ?= -lm

TARGET := infer
SRCS := infer.S

all: $(TARGET)

$(TARGET): $(SRCS) model.inc
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
