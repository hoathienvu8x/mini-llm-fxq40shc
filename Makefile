# mini-llm Makefile

CC = gcc
CFLAGS = -O3 -Wall -Wextra -Wno-unused-result
LDFLAGS = -lm -lpthread

UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M),armv7l)
    CFLAGS += -march=armv7-a -mfpu=neon -mfloat-abi=hard
else ifeq ($(UNAME_M),aarch64)
    CFLAGS += -march=armv8-a
else ifeq ($(UNAME_M),x86_64)
    CFLAGS += -march=x86-64
endif

all: mini-llm

mini-llm: mini-llm.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f mini-llm

install: mini-llm
	cp mini-llm /usr/local/bin/

.PHONY: all clean install
