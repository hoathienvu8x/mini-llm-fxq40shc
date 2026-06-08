# mini-llm

A single-file LLM inference engine. Inspired by Ollama, but ~1000 lines of C.

## Features

- Single file - just one `.c` file
- GGUF model support (F32, F16, Q4_0, Q4_1, Q8_0)
- ARM NEON SIMD (ARMv7 + ARM64)
- Built-in HTTP API server
- Temperature sampling

## Build

```bash
# Linux / macOS
make

# Termux (Android)
pkg install -y git clang make
git clone https://github.com/tundefund0-gif/mini-llm-vmgzotha.git
cd mini-llm-vmgzotha
rm -f mini-llm
make CC=clang

# Or manual
gcc -O3 -o mini-llm mini-llm.c -lm -lpthread
```

## Usage

```bash
# Direct prompt
./mini-llm model.gguf "Hello, how are you?"

# Start API server
./mini-llm model.gguf

# API call
curl http://localhost:8080/api/generate -d '{"prompt":"Hello world"}'
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate text |
| `/api/tags` | GET | List models |
| `/api/version` | GET | Version info |

### Parameters

```json
{
  "prompt": "Your prompt here",
  "max_tokens": 256,
  "temperature": 0.8
}
```

## Supported Models

Works with GGUF format models. Use Q4_0 quantized models for best memory usage.

Small models for testing (~500MB):
```bash
curl -L -o model.gguf https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
```

More models: https://huggingface.co/models?search=gguf

## Termux (Android 32-bit)

```bash
# Install
pkg update -y
pkg install -y git clang make

# Clone and build
git clone https://github.com/tundefund0-gif/mini-llm-vmgzotha.git
cd mini-llm-vmgzotha
rm -f mini-llm
make CC=clang

# Download model
curl -L -o model.gguf https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# Run
./mini-llm model.gguf "Hello!"
```

## License

MIT
