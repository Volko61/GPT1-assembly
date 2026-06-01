# GPT1-assembly

GPT-1 inference implemented three ways: Python (NumPy), C++, and pure ARM64 assembly to benchmark their performance.

<img width="750" height="421" alt="image" src="https://github.com/user-attachments/assets/af494d7e-ca46-47c7-ac5a-bb49fd44f266" />
In the current implementation, we can see the C++ implementation clearly outperform the Python version (6.91x faster) and the Assembly version.

/!\ My implementations might not be ideal, feel free to try to improve the speed and submit as PR

All three read the same model weights and produce identical output. The assembly version uses hand-written NEON vectorization; the C++ version relies on compiler auto-vectorization (`-O3 -march=native`); the Python version uses NumPy for matrix math.

## Quick Start

### 1. Set up dependencies and pack the model
```bash
# Setup python environment
python3 -m venv venv
source venv/bin/activate
pip install numpy

# Download and pack model assets (converts safetensors/json to raw binary offsets)
python3 tools/pack_model.py
```

### 2. Compile and run

```bash
# C++ (portable)
make -C cpp
cd cpp && ./infer

# ARM64 Assembly (requires aarch64)
make -C asm clean && make -C asm
cd asm && ./infer

# Python
python3 main.py
```

## Performance & Verification

A three-way benchmark is included:
```bash
./venv/bin/python3 benchmark.py
```

Results on an ARM64 system:

| Metric | Python (NumPy) | C++ (g++ -O3) | ARM64 Assembly |
|---|---|---|---|
| Model Weight Loading | 1728 ms | ~0 ms (mmap) | ~0 ms (mmap) |
| 10-Token Inference | 53,020 ms | **1,506 ms (35x)** | 3,619 ms (15x) |
| Total Execution | 54,749 ms | **1,506 ms (36x)** | 3,619 ms (15x) |
| **Outputs Match** | ✓ | ✓ | ✓ |

## Code Structure

- `main.py`: Reference Python implementation using NumPy.
- `cpp/infer.cpp`: C++ implementation. Uses `mmap` for zero-copy weight loading, no external dependencies.
- `asm/infer.S`: ARM64 assembly runtime. BPE tokenization, MHA, GELU, LayerNorm, MLP, and detokenization — all in assembly with NEON.
- `asm/model.inc`: Model dimensions and weight byte offsets.
- `tools/pack_model.py`: Packs JSON/safetensors into clean binary headers for the C++ and assembly runtimes.
- `benchmark.py`: Three-way timing comparison and output verification.

## AI Notice
Help of LLMs has been used, especially on the assembly and C++. It's a side project and I did't had enough time to not use these tools.

## Licence 
MIT

## Thanks
Thanks to the openai-community team on HuggingFace for providing the model : https://huggingface.co/openai-community/openai-gpt

## Citation Information
@article{radford2018improving,
  title={Improving language understanding by generative pre-training},
  author={Radford, Alec and Narasimhan, Karthik and Salimans, Tim and Sutskever, Ilya and others},
  year={2018},
  publisher={OpenAI}
}

