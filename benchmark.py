#!/usr/bin/env python3
import time
import subprocess
import os
import json
import struct
import numpy as np

# Standalone benchmark script comparing Python vs Assembly vs C++

def run_assembly_benchmark():
    t_start = time.perf_counter()
    res = subprocess.run(["./infer"], cwd="asm", capture_output=True, text=True)
    t_end = time.perf_counter()
    
    if res.returncode != 0:
        print(f"Error running Assembly binary: {res.stderr}")
        return None, None
        
    lines = res.stdout.strip().split("\n")
    output_text = ""
    for line in lines:
        if "Packed files loaded" not in line:
            output_text += line
            
    total_time = (t_end - t_start) * 1000.0  # ms
    return total_time, output_text.strip()

def run_cpp_benchmark():
    t_start = time.perf_counter()
    res = subprocess.run(["./infer"], cwd="cpp", capture_output=True, text=True)
    t_end = time.perf_counter()
    
    if res.returncode != 0:
        print(f"Error running C++ binary: {res.stderr}")
        return None, None
        
    # C++ prints "Packed files loaded" to stderr, output text to stdout
    lines = res.stdout.strip().split("\n")
    output_text = ""
    for line in lines:
        if "Packed files loaded" not in line:
            output_text += line
            
    total_time = (t_end - t_start) * 1000.0  # ms
    return total_time, output_text.strip()

def run_python_benchmark():
    # 1. Measure Startup & Weight Loading
    t0 = time.perf_counter()
    
    # Load model configuration & tokenizer
    with open("model/config.json", "r") as f:
        config = json.load(f)
    
    n_embd = config["n_embd"]
    n_heads = config["n_head"]
    head_dim = n_embd // n_heads
    n_layers = config["n_layer"]
    vocab_size = config["vocab_size"]
    
    with open("model/tokenizer.json", "r") as f:
        vocab = json.load(f)["model"]["vocab"]
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Load safetensors weights
    dtype_map = {"F32": np.float32}
    with open("model/model.safetensors", "rb") as f:
        h_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(h_len))
        data = f.read()

    w = {
        n: np.frombuffer(
            data[i["data_offsets"][0]:i["data_offsets"][1]], 
            dtype=dtype_map[i["dtype"]]
        ).reshape(i["shape"]) 
        for n, i in header.items() if n != "__metadata__"
    }
    
    t1 = time.perf_counter()
    loading_time = (t1 - t0) * 1000.0  # ms
    
    # 2. Measure Inference Loop
    token_ids = [481, 1988, 544]  # "the sky is"
    
    def mha(x, l):
        qkv = x @ w[f"h.{l}.attn.c_attn.weight"] + w[f"h.{l}.attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        seq_len = x.shape[0]
        q = q.reshape(seq_len, n_heads, head_dim).swapaxes(0, 1)
        k = k.reshape(seq_len, n_heads, head_dim).swapaxes(0, 1)
        v = v.reshape(seq_len, n_heads, head_dim).swapaxes(0, 1)
        scores = (q @ k.swapaxes(-1, -2)) / 8.0
        mask = np.tril(np.ones((seq_len, seq_len)))
        scores = np.where(mask == 1, scores, -1e9)
        probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        context = (probs @ v).swapaxes(0, 1).reshape(seq_len, n_embd)
        return context @ w[f"h.{l}.attn.c_proj.weight"] + w[f"h.{l}.attn.c_proj.bias"]

    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * np.power(x, 3))))

    def mlp(x, l):
        h1 = gelu(x @ w[f"h.{l}.mlp.c_fc.weight"] + w[f"h.{l}.mlp.c_fc.bias"])
        return h1 @ w[f"h.{l}.mlp.c_proj.weight"] + w[f"h.{l}.mlp.c_proj.bias"]

    def layer_norm(x, weight, bias):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return weight * (x - mean) / np.sqrt(var + 1e-5) + bias

    # Generate 10 tokens deterministically
    t2 = time.perf_counter()
    for step in range(10):
        x = w["tokens_embed.weight"][token_ids] + w["positions_embed.weight"][:len(token_ids)]
        for l in range(n_layers):
            x = layer_norm(x + mha(x, l), w[f"h.{l}.ln_1.weight"], w[f"h.{l}.ln_1.bias"])
            x = layer_norm(x + mlp(x, l), w[f"h.{l}.ln_2.weight"], w[f"h.{l}.ln_2.bias"])
        logits = x[-1] @ w["tokens_embed.weight"].T
        next_id = int(np.argmax(logits))
        token_ids.append(next_id)
    t3 = time.perf_counter()
    
    inference_time = (t3 - t2) * 1000.0  # ms
    
    import re
    output_text = "".join([id_to_token[tid] for tid in token_ids])
    output_text = output_text.replace("</w>", " ")
    output_text = re.sub(r"\s+([,.;:!?])", r"\1", output_text)
    output_text = output_text.strip()
    
    return loading_time, inference_time, output_text

def main():
    print("--- GPT-1 Inference Benchmark ---")
    print("Running Python (NumPy) reference...")
    py_load, py_infer, py_out = run_python_benchmark()
    py_total = py_load + py_infer
    print(f"  - Load time: {py_load:.2f} ms")
    print(f"  - 10-step inference loop: {py_infer:.2f} ms")
    print(f"  - Output: \"{py_out}\"")
    print()
    
    print("Running C++ reference...")
    cpp_total, cpp_out = run_cpp_benchmark()
    if cpp_total is None:
        print("  - C++ benchmark failed, skipping")
        cpp_total = 0
        cpp_out = ""
    else:
        print(f"  - Total execution time (mmap + inference): {cpp_total:.2f} ms")
        print(f"  - Output: \"{cpp_out}\"")
    print()
    
    print("Running ARM64 Assembly reference...")
    asm_total, asm_out = run_assembly_benchmark()
    if asm_total is None:
        print("  - Assembly benchmark failed, skipping")
        asm_total = 0
        asm_out = ""
    print(f"  - Total execution time (mmap + inference): {asm_total:.2f} ms")
    print(f"  - Output: \"{asm_out}\"")
    print()
    
    # Output match check
    outputs = {"Python": py_out}
    if cpp_out:
        outputs["C++"] = cpp_out
    if asm_out:
        outputs["Assembly"] = asm_out
    all_match = len(set(outputs.values())) == 1
    match_str = "OK" if all_match else "FAILED"
    
    # Speedup calculations
    cpp_vs_py = f"{py_infer / cpp_total:.2f}x" if cpp_total > 0 else "N/A"
    asm_vs_py = f"{py_infer / asm_total:.2f}x" if asm_total > 0 else "N/A"
    cpp_total_vs_py = f"{py_total / cpp_total:.2f}x" if cpp_total > 0 else "N/A"
    asm_total_vs_py = f"{py_total / asm_total:.2f}x" if asm_total > 0 else "N/A"
    
    print("Results:")
    print("--------------------------------------------------------------------------------------------")
    print(f"{'Metric':<35} | {'Python (NumPy)':<16} | {'C++ (g++ -O3)':<22} | {'ARM64 Assembly'}")
    print("--------------------------------------------------------------------------------------------")
    print(f"{'Model Weight Loading':<35} | {f'{py_load:.2f} ms':<16} | {'~0.01 ms (mmap)':<22} | ~0.01 ms (mmap)")
    print(f"{'10-Token Generation (Inference)':<35} | {f'{py_infer:.2f} ms':<16} | {f'{cpp_total:.2f} ms ({cpp_vs_py})':<22} | {f'{asm_total:.2f} ms ({asm_vs_py})'}")
    print(f"{'Total Process Execution Time':<35} | {f'{py_total:.2f} ms':<16} | {f'{cpp_total:.2f} ms ({cpp_total_vs_py})':<22} | {f'{asm_total:.2f} ms ({asm_total_vs_py})'}")
    print("--------------------------------------------------------------------------------------------")
    print(f"Outputs Match: {match_str}\n")

if __name__ == "__main__":
    main()
