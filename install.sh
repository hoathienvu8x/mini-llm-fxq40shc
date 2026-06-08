#!/bin/bash
# mini-llm installer
set -e

echo "=== mini-llm installer ==="

if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
    echo "Installing compiler..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y gcc
    elif command -v pkg &> /dev/null; then
        pkg install -y clang
    fi
fi

CC=gcc
command -v clang &> /dev/null && CC=clang

ARCH=$(uname -m)
CFLAGS="-O3 -Wall"

case "$ARCH" in
    armv7*|armhf) CFLAGS="$CFLAGS -march=armv7-a -mfpu=neon -mfloat-abi=hard" ;;
    aarch64|arm64) CFLAGS="$CFLAGS -march=armv8-a" ;;
esac

echo "Compiling..."
$CC $CFLAGS -o mini-llm mini-llm.c -lm -lpthread

echo "Done! Run: ./mini-llm model.gguf"
