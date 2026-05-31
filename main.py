# 1 find the files                          
import os
import json
import re
import string
import struct
from typing import List
import numpy as np

path = "model"

def load_safetensors(file_path: str):
    dtype_map = {
        "F16": np.float16,
        "F32": np.float32,
        "F64": np.float64,
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U8": np.uint8,
        "BOOL": np.bool_,
    }

    with open(file_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()

    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = dtype_map.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {info['dtype']}")
        start, end = info["data_offsets"]
        arr = np.frombuffer(data[start:end], dtype=dtype).reshape(info["shape"])
        tensors[name] = arr

    return tensors

with open(os.path.join(path, "tokenizer.json"), "r", encoding="utf-8") as f:
    tokenizer_config = json.load(f)

vocab = tokenizer_config["model"]["vocab"]
id_to_token = {v: k for k, v in vocab.items()}
unk_token = tokenizer_config["model"].get("unk_token", "<unk>")
unk_id = vocab.get(unk_token, 0)
bpe_suffix = tokenizer_config["model"].get("end_of_word_suffix", "")

normalizer_config = tokenizer_config.get("normalizer") or {}
lowercase_flag = bool(normalizer_config.get("lowercase", False))

with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)

vocab_size = config["vocab_size"]       
n_positions = config["n_positions"]     
n_embds = config["n_embd"]              
n_layers = config["n_layer"]              
n_heads = config["n_head"]   
head_dim = n_embds // n_heads 

# 2 tokenize                                    -> text to numbers
def merge(sentence: str, add_suffix: bool = False) -> List[str]:
    result = list(sentence)
    if add_suffix and bpe_suffix and result:
        result[-1] = result[-1] + bpe_suffix
    with open(os.path.join(path, "merges.txt"), "r", encoding="utf-8") as f:
        for rule in f.readlines()[1:]:
            rule = rule[:-1].split(" ")
            # print("rule :", rule)
            i = 0
            while i < len(result) - 1:
                a = result[i]
                b = result[i + 1]
                if a == rule[0] and b == rule[1]:
                    # print("merging:", a, "+", b)
                    result[i:i+2] = [rule[0] + rule[1]]
                else:
                    i += 1
    return result


def pretokenize(text: str) -> List[str]:
    tokens = []
    token = ""
    for ch in text:
        if ch.isspace():
            if token:
                tokens.append(token)
                token = ""
            continue
        if ch in string.punctuation:
            if token:
                tokens.append(token)
                token = ""
            tokens.append(ch)
            continue
        token += ch
    if token:
        tokens.append(token)
    return tokens


def tokens_to_tensor(text: str) -> List[int]:
    if lowercase_flag:
        text = text.lower()
    result = []
    for token in pretokenize(text):
        for piece in merge(token, add_suffix=True):
            result.append(vocab.get(piece, unk_id))

    return result

def tensor_to_token(tensors: List[int]) -> str:
    result = ""
    for idx in tensors:
        result += id_to_token[idx]
    if bpe_suffix:
        result = result.replace(bpe_suffix, " ")
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    result = re.sub(r"\s+'\s*", "'", result)
    return result.rstrip()

# print(tokens_to_tensor(merge("there")))

# 3 embedding                                   -> numbers to ideas
weights = load_safetensors(os.path.join(path, "model.safetensors"))
def embedding(token_ids: List[int]) -> np.ndarray:
    word_token_embeddings = weights["tokens_embed.weight"]
    word_position_embeddings = weights["positions_embed.weight"]
    
    result = []
    for i, token in enumerate(token_ids):
        word_token_embedding = word_token_embeddings[token]
        word_position_embedding = word_position_embeddings[i]
        result.append(word_token_embedding + word_position_embedding)
    return np.array(result)


def lm_head(hidden_states):
    return hidden_states @ weights["tokens_embed.weight"].T

# print(embedding([5124]))

# 4 transformer loop (matrix multiplications)   -> guess new idea probabilities

def layer_norm(x, weight, bias, eps=1e-5):
    mean= np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) /np.sqrt(var +eps) + bias

def softmax(x):
    probs = np.exp(x - np.max(x, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return probs

def sample_next_id(logits, token_ids, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    scores = logits.astype(np.float64).copy()

    if repetition_penalty != 1.0:
        for tid in set(token_ids):
            if scores[tid] > 0:
                scores[tid] /= repetition_penalty
            else:
                scores[tid] *= repetition_penalty

    if temperature != 1.0:
        scores = scores / temperature

    probs = softmax(scores)
    if top_p < 1.0:
        sorted_ids = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_ids]
        cumulative = np.cumsum(sorted_probs)
        cutoff = cumulative <= top_p
        if not np.any(cutoff):
            cutoff[0] = True
        kept_ids = sorted_ids[cutoff]
        kept_probs = probs[kept_ids]
        kept_probs = kept_probs / np.sum(kept_probs)
        return int(np.random.choice(kept_ids, p=kept_probs))

    return int(np.random.choice(len(probs), p=probs))

def mha_block(x, l):
    seq_len = x.shape[0]

    #QVK
    qkv = x @ weights[f"h.{l}.attn.c_attn.weight"] + weights[f"h.{l}.attn.c_attn.bias"]
    q, k, v = np.split(qkv, 3, axis=-1)

    q = q.reshape(seq_len, n_heads, head_dim).swapaxes(0, 1)
    k = k.reshape(seq_len, n_heads, head_dim).swapaxes(0, 1)
    v = v.reshape(seq_len, n_heads, head_dim).swapaxes(0, 1)

    scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(head_dim)

    mask = np.tril(np.ones((seq_len, seq_len)))
    scores = np.where(mask == 1, scores, -1e9)

    probs = softmax(scores)

    # combine heads back together
    context = (probs @ v).swapaxes(0, 1).reshape(seq_len, n_embds)

    # projection 
    out = context @ weights[f"h.{l}.attn.c_proj.weight"] + weights[f"h.{l}.attn.c_proj.bias"]
    return out

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def mlp_block(x, l):
    h1 = gelu(x @ weights[f"h.{l}.mlp.c_fc.weight"] + weights[f"h.{l}.mlp.c_fc.bias"] )
    out = h1 @ weights[f"h.{l}.mlp.c_proj.weight"] + weights[f"h.{l}.mlp.c_proj.bias"]
    return out

def transformer_loop(hidden_states):
    for l in range(n_layers):
        # Attention
        attn_out = mha_block(hidden_states, l)
        hidden_states = layer_norm(
            hidden_states + attn_out,
            weights[f"h.{l}.ln_1.weight"],
            weights[f"h.{l}.ln_1.bias"],
        )
        # MLP
        mlp_out = mlp_block(hidden_states, l)
        hidden_states = layer_norm(
            hidden_states + mlp_out,
            weights[f"h.{l}.ln_2.weight"],
            weights[f"h.{l}.ln_2.bias"],
        )

    return hidden_states

prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was"
prompt = "God save the "
prompt = "The sky is"
temperature = 1
top_p = 1
repetition_penalty = 1
token_ids = tokens_to_tensor(prompt)
decoded_prompt = tensor_to_token(token_ids)
print(decoded_prompt, end='', flush=True)

for i in range(10):
    hidden_states = transformer_loop(embedding(token_ids))

    # 5 projection                                  -> new idea probablities to number probabilities
    logits = lm_head(hidden_states)

    # 6 sampling                                    -> select the most probable number and convert back to text thanks to tokenizer
    next_id = sample_next_id(logits[-1], token_ids, temperature, top_p, repetition_penalty)
    token_ids.append(next_id)
    prompt = tensor_to_token(token_ids)
    print(prompt[len(decoded_prompt):], end='', flush=True)
    decoded_prompt = prompt

print()


