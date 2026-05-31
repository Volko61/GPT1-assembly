#!/usr/bin/env python3
import argparse
import json
import os
import struct
import sys

HEADER_SIZE = 64
VERSION = 1

TOKENIZER_MAGIC = b"TKR1"
MERGES_MAGIC = b"MRG1"
WEIGHTS_MAGIC = b"WGT1"

FNV_OFFSET_BASIS_64 = 0xcbf29ce484222325
FNV_PRIME_64 = 0x100000001b3


def align(value, alignment):
    return (value + alignment - 1) & ~(alignment - 1)


def fnv1a_64(data):
    h = FNV_OFFSET_BASIS_64
    for b in data:
        h ^= b
        h = (h * FNV_PRIME_64) & 0xFFFFFFFFFFFFFFFF
    return h


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_safetensors_header(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def write_header(f, magic, fields):
    data = struct.pack("<4s", magic)
    if fields:
        data += struct.pack("<" + "I" * len(fields), *fields)
    if len(data) > HEADER_SIZE:
        raise ValueError("Header too large")
    data += b"\x00" * (HEADER_SIZE - len(data))
    f.write(data)


def build_vocab(tokenizer):
    vocab = tokenizer["model"]["vocab"]
    vocab_size = max(vocab.values()) + 1
    id_to_token = [None] * vocab_size
    for token, idx in vocab.items():
        id_to_token[idx] = token
    missing = [i for i, t in enumerate(id_to_token) if t is None]
    if missing:
        raise ValueError("Vocab has missing ids: {}".format(missing[:8]))
    return vocab, id_to_token


def build_hash_table(id_to_token):
    vocab_size = len(id_to_token)
    table_size = 1
    while table_size < vocab_size * 2:
        table_size <<= 1

    empty = 0xFFFFFFFF
    hashes = [0] * table_size
    ids = [empty] * table_size

    for idx, token in enumerate(id_to_token):
        data = token.encode("utf-8")
        h = fnv1a_64(data)
        slot = h & (table_size - 1)
        while ids[slot] != empty:
            slot = (slot + 1) & (table_size - 1)
        hashes[slot] = h
        ids[slot] = idx

    return hashes, ids


def write_tokenizer_bin(out_path, tokenizer, id_to_token):
    vocab = tokenizer["model"]["vocab"]
    vocab_size = len(id_to_token)
    unk_token = tokenizer["model"].get("unk_token", "<unk>")
    unk_id = vocab.get(unk_token, 0)
    suffix = tokenizer["model"].get("end_of_word_suffix", "")

    normalizer = tokenizer.get("normalizer") or {}
    lowercase = 1 if normalizer.get("lowercase", False) else 0

    offsets = [0]
    data_bytes = bytearray()
    for token in id_to_token:
        b = token.encode("utf-8")
        data_bytes.extend(b)
        offsets.append(len(data_bytes))

    hashes, ids = build_hash_table(id_to_token)

    suffix_bytes = suffix.encode("utf-8")
    offset = HEADER_SIZE

    suffix_offset = offset
    offset = align(offset + len(suffix_bytes), 4)

    offsets_offset = offset
    offset = align(offset + 4 * len(offsets), 4)

    strings_offset = offset
    offset = align(offset + len(data_bytes), 8)

    hash_offset = offset
    offset = align(offset + 8 * len(hashes), 4)

    ids_offset = offset

    with open(out_path, "wb") as f:
        write_header(
            f,
            TOKENIZER_MAGIC,
            [
                VERSION,
                vocab_size,
                unk_id,
                lowercase,
                len(suffix_bytes),
                suffix_offset,
                offsets_offset,
                strings_offset,
                hash_offset,
                len(hashes),
                ids_offset,
            ],
        )
        f.seek(suffix_offset)
        f.write(suffix_bytes)
        f.seek(offsets_offset)
        f.write(struct.pack("<" + "I" * len(offsets), *offsets))
        f.seek(strings_offset)
        f.write(data_bytes)
        f.seek(hash_offset)
        f.write(struct.pack("<" + "Q" * len(hashes), *hashes))
        f.seek(ids_offset)
        f.write(struct.pack("<" + "I" * len(ids), *ids))


def write_merges_bin(out_path, merges, vocab):
    triples = []
    for left, right in merges:
        left_id = vocab.get(left)
        right_id = vocab.get(right)
        if left_id is None or right_id is None:
            raise ValueError("Merge symbol missing in vocab: {} {}".format(left, right))
        merged = left + right
        merged_id = vocab.get(merged)
        if merged_id is None:
            raise ValueError("Merged token missing in vocab: {}".format(merged))
        triples.append((left_id, right_id, merged_id))

    triples_offset = HEADER_SIZE
    with open(out_path, "wb") as f:
        write_header(f, MERGES_MAGIC, [VERSION, len(triples), triples_offset])
        f.seek(triples_offset)
        for left_id, right_id, merged_id in triples:
            f.write(struct.pack("<III", left_id, right_id, merged_id))


def expect_shape(info, expected):
    if info["shape"] != expected:
        raise ValueError("Shape mismatch for {}: {} != {}".format(info["name"], info["shape"], expected))


def write_weights_bin(out_path, header, data_start, model_dir, config):
    n_embd = config["n_embd"]
    n_heads = config["n_head"]
    n_layers = config["n_layer"]
    n_positions = config["n_positions"]
    vocab_size = config["vocab_size"]
    head_dim = n_embd // n_heads
    n_ctx = config.get("n_ctx", n_positions)

    order = [
        "tokens_embed.weight",
        "positions_embed.weight",
    ]
    for l in range(n_layers):
        order.extend(
            [
                f"h.{l}.attn.c_attn.weight",
                f"h.{l}.attn.c_attn.bias",
                f"h.{l}.attn.c_proj.weight",
                f"h.{l}.attn.c_proj.bias",
                f"h.{l}.ln_1.weight",
                f"h.{l}.ln_1.bias",
                f"h.{l}.mlp.c_fc.weight",
                f"h.{l}.mlp.c_fc.bias",
                f"h.{l}.mlp.c_proj.weight",
                f"h.{l}.mlp.c_proj.bias",
                f"h.{l}.ln_2.weight",
                f"h.{l}.ln_2.bias",
            ]
        )

    expected_shapes = {
        "tokens_embed.weight": [vocab_size, n_embd],
        "positions_embed.weight": [n_positions, n_embd],
    }
    for l in range(n_layers):
        expected_shapes.update(
            {
                f"h.{l}.attn.c_attn.weight": [n_embd, 3 * n_embd],
                f"h.{l}.attn.c_attn.bias": [3 * n_embd],
                f"h.{l}.attn.c_proj.weight": [n_embd, n_embd],
                f"h.{l}.attn.c_proj.bias": [n_embd],
                f"h.{l}.ln_1.weight": [n_embd],
                f"h.{l}.ln_1.bias": [n_embd],
                f"h.{l}.mlp.c_fc.weight": [n_embd, 4 * n_embd],
                f"h.{l}.mlp.c_fc.bias": [4 * n_embd],
                f"h.{l}.mlp.c_proj.weight": [4 * n_embd, n_embd],
                f"h.{l}.mlp.c_proj.bias": [n_embd],
                f"h.{l}.ln_2.weight": [n_embd],
                f"h.{l}.ln_2.bias": [n_embd],
            }
        )

    data_offset = HEADER_SIZE
    with open(out_path, "wb") as out_f, open(os.path.join(model_dir, "model.safetensors"), "rb") as in_f:
        write_header(
            out_f,
            WEIGHTS_MAGIC,
            [
                VERSION,
                n_embd,
                n_heads,
                n_layers,
                n_positions,
                vocab_size,
                head_dim,
                n_ctx,
                data_offset,
            ],
        )
        for name in order:
            info = header.get(name)
            if info is None:
                raise ValueError("Missing tensor: {}".format(name))
            info["name"] = name
            if info["dtype"] != "F32":
                raise ValueError("Only F32 supported, got {} for {}".format(info["dtype"], name))
            expect_shape(info, expected_shapes[name])
            start, end = info["data_offsets"]
            in_f.seek(data_start + start)
            chunk = in_f.read(end - start)
            if len(chunk) != end - start:
                raise ValueError("Short read for {}".format(name))
            out_f.write(chunk)


def write_model_inc(out_path, config, tokenizer, merges_count):
    n_embd = config["n_embd"]
    n_heads = config["n_head"]
    n_layers = config["n_layer"]
    n_positions = config["n_positions"]
    vocab_size = config["vocab_size"]
    head_dim = n_embd // n_heads
    n_ctx = config.get("n_ctx", n_positions)

    unk_token = tokenizer["model"].get("unk_token", "<unk>")
    vocab = tokenizer["model"]["vocab"]
    unk_id = vocab.get(unk_token, 0)

    lines = [
        ".equ N_EMBD, {}".format(n_embd),
        ".equ N_HEADS, {}".format(n_heads),
        ".equ HEAD_DIM, {}".format(head_dim),
        ".equ N_LAYERS, {}".format(n_layers),
        ".equ N_POSITIONS, {}".format(n_positions),
        ".equ N_CTX, {}".format(n_ctx),
        ".equ VOCAB_SIZE, {}".format(vocab_size),
        ".equ UNK_ID, {}".format(unk_id),
        ".equ MERGES_COUNT, {}".format(merges_count),
        ".equ TOKENIZER_HEADER_SIZE, {}".format(HEADER_SIZE),
        ".equ MERGES_HEADER_SIZE, {}".format(HEADER_SIZE),
        ".equ WEIGHTS_HEADER_SIZE, {}".format(HEADER_SIZE),
        ".equ MAGIC_TKR1, 0x31524b54",
        ".equ MAGIC_MRG1, 0x3147524d",
        ".equ MAGIC_WGT1, 0x31544757",
        ".equ HASH_EMPTY, 0xFFFFFFFF",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Pack GPT1 model assets into binary blobs.")
    parser.add_argument("--model-dir", default="model", help="Model directory path")
    parser.add_argument("--out-dir", default="packed", help="Output directory path")
    parser.add_argument("--asm-inc", default=os.path.join("asm", "model.inc"), help="Output path for asm include")
    args = parser.parse_args()

    model_dir = args.model_dir
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.asm_inc), exist_ok=True)

    config = read_json(os.path.join(model_dir, "config.json"))
    tokenizer = read_json(os.path.join(model_dir, "tokenizer.json"))

    vocab, id_to_token = build_vocab(tokenizer)

    merges_path = os.path.join(model_dir, "merges.txt")
    with open(merges_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    merges = [tuple(line.split()) for line in lines[1:] if line.strip()]

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    header, data_start = read_safetensors_header(safetensors_path)

    write_tokenizer_bin(os.path.join(out_dir, "tokenizer.bin"), tokenizer, id_to_token)
    write_merges_bin(os.path.join(out_dir, "merges.bin"), merges, vocab)
    write_weights_bin(os.path.join(out_dir, "weights.bin"), header, data_start, model_dir, config)
    write_model_inc(args.asm_inc, config, tokenizer, len(merges))

    print("Packed assets in {}".format(out_dir))


if __name__ == "__main__":
    main()
