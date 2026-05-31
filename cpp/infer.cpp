/* ================================================================
   GPT-1 Inference in C++ — reads the same packed binary files
   as the ARM64 assembly version (packed/tokenizer.bin, merges.bin,
   weights.bin) via mmap for zero-copy weight loading.
   ================================================================ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <algorithm>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ── Model constants ── */
static constexpr int N_EMBD      = 768;
static constexpr int N_HEADS     = 12;
static constexpr int HEAD_DIM    = 64;
static constexpr int N_LAYERS    = 12;
static constexpr int N_POSITIONS = 512;
static constexpr int N_CTX       = 512;
static constexpr int VOCAB_SIZE  = 40478;
static constexpr int N_FFN       = 3072;
static constexpr int QKV_DIM     = 2304;  // 3 * N_EMBD
static constexpr int GEN_TOKENS  = 10;

/* ── Packed file magic numbers ── */
static constexpr uint32_t MAGIC_TKR1 = 0x31524b54;
static constexpr uint32_t MAGIC_MRG1 = 0x3147524d;
static constexpr uint32_t MAGIC_WGT1 = 0x31544757;
static constexpr uint32_t HASH_EMPTY = 0xFFFFFFFF;

/* ── FNV-1a constants ── */
static constexpr uint64_t FNV_OFFSET_BASIS_64 = 0xcbf29ce484222325ULL;
static constexpr uint64_t FNV_PRIME_64        = 0x100000001b3ULL;

/* ── Per-layer weight byte sizes ── */
static constexpr int CATTN_W_BYTES  = N_EMBD * QKV_DIM * 4;
static constexpr int CATTN_B_BYTES  = QKV_DIM * 4;
static constexpr int CPROJ_W_BYTES  = N_EMBD * N_EMBD * 4;
static constexpr int VEC768_BYTES   = N_EMBD * 4;
static constexpr int CFC_W_BYTES    = N_EMBD * N_FFN * 4;
static constexpr int CFC_B_BYTES    = N_FFN * 4;
static constexpr int CPROJ2_W_BYTES = N_FFN * N_EMBD * 4;

/* ── Within-layer byte offsets ── */
static constexpr int OFF_CATTN_W  = 0;
static constexpr int OFF_CATTN_B  = OFF_CATTN_W  + CATTN_W_BYTES;
static constexpr int OFF_CPROJ_W  = OFF_CATTN_B  + CATTN_B_BYTES;
static constexpr int OFF_CPROJ_B  = OFF_CPROJ_W  + CPROJ_W_BYTES;
static constexpr int OFF_LN1_W    = OFF_CPROJ_B  + VEC768_BYTES;
static constexpr int OFF_LN1_B    = OFF_LN1_W    + VEC768_BYTES;
static constexpr int OFF_CFC_W    = OFF_LN1_B    + VEC768_BYTES;
static constexpr int OFF_CFC_B    = OFF_CFC_W    + CFC_W_BYTES;
static constexpr int OFF_CPROJ2_W = OFF_CFC_B    + CFC_B_BYTES;
static constexpr int OFF_CPROJ2_B = OFF_CPROJ2_W + CPROJ2_W_BYTES;
static constexpr int OFF_LN2_W    = OFF_CPROJ2_B + VEC768_BYTES;
static constexpr int OFF_LN2_B    = OFF_LN2_W    + VEC768_BYTES;
static constexpr int LAYER_BYTES  = OFF_LN2_B    + VEC768_BYTES;

static constexpr int WEIGHTS_HEADER_SIZE = 64;

/* ================================================================
   File mapping
   ================================================================ */
struct MappedFile {
    void*  base;
    size_t length;
};

static MappedFile load_file(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "Failed to open: %s\n", path); exit(1); }
    struct stat st;
    fstat(fd, &st);
    void* ptr = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) { fprintf(stderr, "mmap failed: %s\n", path); exit(1); }
    return {ptr, static_cast<size_t>(st.st_size)};
}

static void check_magic(const void* base, uint32_t expected, const char* msg) {
    uint32_t got;
    memcpy(&got, base, 4);
    if (got != expected) { fprintf(stderr, "%s\n", msg); exit(1); }
}

/* ================================================================
   Tokenizer state
   ================================================================ */
struct Tokenizer {
    int      vocab_size;
    int      unk_id;
    int      lowercase;
    int      suffix_len;
    const char*     suffix_ptr;
    const uint32_t* offsets_ptr;
    const char*     strings_ptr;
    const uint64_t* hash_ptr;
    int             hash_len;
    const uint32_t* ids_ptr;
    int             merges_count;
    const uint32_t* merges_ptr;  // triples: (left_id, right_id, merged_id) packed as 3×uint32
};

static Tokenizer g_tok;

static void tokenizer_setup(const uint8_t* tok_base, const uint8_t* merges_base) {
    auto r32 = [](const uint8_t* p, int off) -> uint32_t {
        uint32_t v; memcpy(&v, p + off, 4); return v;
    };

    g_tok.vocab_size = static_cast<int>(r32(tok_base, 8));
    g_tok.unk_id     = static_cast<int>(r32(tok_base, 12));
    g_tok.lowercase  = static_cast<int>(r32(tok_base, 16));
    g_tok.suffix_len = static_cast<int>(r32(tok_base, 20));

    g_tok.suffix_ptr  = reinterpret_cast<const char*>(tok_base + r32(tok_base, 24));
    g_tok.offsets_ptr = reinterpret_cast<const uint32_t*>(tok_base + r32(tok_base, 28));
    g_tok.strings_ptr = reinterpret_cast<const char*>(tok_base + r32(tok_base, 32));
    g_tok.hash_ptr    = reinterpret_cast<const uint64_t*>(tok_base + r32(tok_base, 36));
    g_tok.hash_len    = static_cast<int>(r32(tok_base, 40));
    g_tok.ids_ptr     = reinterpret_cast<const uint32_t*>(tok_base + r32(tok_base, 44));

    g_tok.merges_count = static_cast<int>(r32(merges_base, 8));
    g_tok.merges_ptr   = reinterpret_cast<const uint32_t*>(merges_base + r32(merges_base, 12));
}

/* ================================================================
   FNV-1a hash + lookup
   ================================================================ */
static uint64_t fnv1a(const char* str, int len) {
    uint64_t h = FNV_OFFSET_BASIS_64;
    for (int i = 0; i < len; i++) {
        h ^= static_cast<uint8_t>(str[i]);
        h *= FNV_PRIME_64;
    }
    return h;
}

static int lookup_token(const char* str, int len) {
    uint64_t h = fnv1a(str, len);
    uint64_t mask = static_cast<uint64_t>(g_tok.hash_len) - 1;
    uint64_t slot = h & mask;

    for (;;) {
        uint32_t id = g_tok.ids_ptr[slot];
        if (id == HASH_EMPTY) return g_tok.unk_id;
        if (g_tok.hash_ptr[slot] == h) {
            // verify string match
            uint32_t start = g_tok.offsets_ptr[id];
            uint32_t end   = g_tok.offsets_ptr[id + 1];
            int slen = static_cast<int>(end - start);
            if (slen == len && memcmp(g_tok.strings_ptr + start, str, len) == 0) {
                return static_cast<int>(id);
            }
        }
        slot = (slot + 1) & mask;
    }
}

/* ================================================================
   BPE merge on a single word
   ================================================================ */
static int merge_word(const char* str, int len, int* out_ids, int max_ids) {
    char tmp[32];
    int count = 0;

    for (int i = 0; i < len && count < max_ids; i++) {
        int tlen = 0;
        tmp[tlen++] = str[i];
        // if last char of word, append suffix
        if (i == len - 1 && g_tok.suffix_len > 0) {
            memcpy(tmp + tlen, g_tok.suffix_ptr, g_tok.suffix_len);
            tlen += g_tok.suffix_len;
        }
        out_ids[count++] = lookup_token(tmp, tlen);
    }

    // apply BPE merges
    const uint32_t* mp = g_tok.merges_ptr;
    for (int m = 0; m < g_tok.merges_count; m++) {
        uint32_t left_id  = mp[m * 3 + 0];
        uint32_t right_id = mp[m * 3 + 1];
        uint32_t merged   = mp[m * 3 + 2];
        int i = 0;
        while (i < count - 1) {
            if (static_cast<uint32_t>(out_ids[i]) == left_id &&
                static_cast<uint32_t>(out_ids[i + 1]) == right_id) {
                out_ids[i] = static_cast<int>(merged);
                // shift left
                for (int j = i + 1; j < count - 1; j++)
                    out_ids[j] = out_ids[j + 1];
                count--;
                // don't increment i — check for further merges at same position
            } else {
                i++;
            }
        }
    }
    return count;
}

/* ================================================================
   Pre-tokenize + tokenize
   ================================================================ */
static bool is_punct(char c) {
    static const char* p = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    return strchr(p, c) != nullptr;
}

static int tokenize_prompt(const char* text, int* out_ids, int max_ids) {
    // optional lowercase copy
    char* lc_buf = nullptr;
    if (g_tok.lowercase) {
        int slen = static_cast<int>(strlen(text));
        lc_buf = static_cast<char*>(malloc(slen + 1));
        for (int i = 0; i <= slen; i++)
            lc_buf[i] = static_cast<char>(tolower(static_cast<unsigned char>(text[i])));
        text = lc_buf;
    }

    int total = 0;
    int scratch[N_CTX];
    const char* word_start = nullptr;
    int word_len = 0;

    auto flush_word = [&]() {
        if (word_len == 0) return;
        int n = merge_word(word_start, word_len, scratch, N_CTX);
        for (int i = 0; i < n && total < max_ids; i++)
            out_ids[total++] = scratch[i];
        word_len = 0;
        word_start = nullptr;
    };

    for (const char* p = text; *p; p++) {
        unsigned char c = static_cast<unsigned char>(*p);
        if (isspace(c)) {
            flush_word();
        } else if (is_punct(static_cast<char>(c))) {
            flush_word();
            // punctuation as a single-char word
            int n = merge_word(p, 1, scratch, N_CTX);
            for (int i = 0; i < n && total < max_ids; i++)
                out_ids[total++] = scratch[i];
        } else {
            if (word_len == 0) word_start = p;
            word_len++;
        }
    }
    flush_word();

    free(lc_buf);
    return total;
}

/* ================================================================
   Detokenization
   ================================================================ */
static void detok_print(int token_id, bool& pending_space) {
    uint32_t start = g_tok.offsets_ptr[token_id];
    uint32_t end   = g_tok.offsets_ptr[token_id + 1];
    int slen = static_cast<int>(end - start);
    const char* str = g_tok.strings_ptr + start;

    // check if first char is punctuation that suppresses leading space
    if (slen > 0) {
        char c = str[0];
        if (c == ',' || c == '.' || c == ';' || c == ':' || c == '!' || c == '?') {
            pending_space = false;
        }
    }

    if (pending_space) {
        putchar(' ');
        pending_space = false;
    }

    // check if token ends with suffix
    if (g_tok.suffix_len > 0 && slen >= g_tok.suffix_len) {
        int tail = slen - g_tok.suffix_len;
        if (memcmp(str + tail, g_tok.suffix_ptr, g_tok.suffix_len) == 0) {
            fwrite(str, 1, tail, stdout);
            pending_space = true;
            return;
        }
    }

    fwrite(str, 1, slen, stdout);
    pending_space = false;
}

/* ================================================================
   Weight pointers
   ================================================================ */
struct Weights {
    const float* tok_embed;   // [VOCAB_SIZE, N_EMBD]
    const float* pos_embed;   // [N_POSITIONS, N_EMBD]
    const uint8_t* layers_base;
};

static Weights g_w;

static void weights_setup(const uint8_t* base) {
    const uint8_t* p = base + WEIGHTS_HEADER_SIZE;
    g_w.tok_embed = reinterpret_cast<const float*>(p);
    p += VOCAB_SIZE * N_EMBD * 4;
    g_w.pos_embed = reinterpret_cast<const float*>(p);
    p += N_POSITIONS * N_EMBD * 4;
    g_w.layers_base = p;
}

static const float* layer_ptr(int layer, int offset) {
    return reinterpret_cast<const float*>(g_w.layers_base + static_cast<size_t>(layer) * LAYER_BYTES + offset);
}

/* ================================================================
   GEMM: C[M,N] = A[M,K] @ B[K,N] + bias[N]
   ================================================================ */
static void gemm(float* C, const float* A, const float* B, const float* bias,
                 int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;
        // initialize with bias or zero
        if (bias) {
            memcpy(c_row, bias, N * sizeof(float));
        } else {
            memset(c_row, 0, N * sizeof(float));
        }
        for (int k = 0; k < K; k++) {
            float a_val = a_row[k];
            const float* b_row = B + k * N;
            for (int j = 0; j < N; j++) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

/* ================================================================
   vec_add_inplace: dst[i] += src[i]
   ================================================================ */
static void vec_add_inplace(float* dst, const float* src, int count) {
    for (int i = 0; i < count; i++)
        dst[i] += src[i];
}

/* ================================================================
   Layer normalization (in-place)
   ================================================================ */
static void layer_norm_inplace(float* data, const float* weight, const float* bias,
                               int rows, int cols) {
    constexpr float eps = 1e-5f;
    for (int r = 0; r < rows; r++) {
        float* row = data + r * cols;
        // mean
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) sum += row[c];
        float mean = sum / cols;
        // variance
        float var_sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float d = row[c] - mean;
            var_sum += d * d;
        }
        float inv_std = 1.0f / sqrtf(var_sum / cols + eps);
        // normalize
        for (int c = 0; c < cols; c++) {
            row[c] = weight[c] * (row[c] - mean) * inv_std + bias[c];
        }
    }
}

/* ================================================================
   Softmax (in-place, single row)
   ================================================================ */
static void softmax_row(float* data, int len) {
    float max_val = data[0];
    for (int i = 1; i < len; i++)
        if (data[i] > max_val) max_val = data[i];
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    for (int i = 0; i < len; i++)
        data[i] /= sum;
}

/* ================================================================
   GELU activation (in-place)
   ================================================================ */
static void gelu_inplace(float* data, int count) {
    constexpr float sqrt2pi = 0.7978845608f;
    constexpr float c = 0.044715f;
    for (int i = 0; i < count; i++) {
        float x = data[i];
        float arg = sqrt2pi * (x + c * x * x * x);
        data[i] = 0.5f * x * (1.0f + tanhf(arg));
    }
}

/* ================================================================
   Embedding: out[i] = tok_embed[ids[i]] + pos_embed[i]
   ================================================================ */
static void embedding(float* out, const int* token_ids, int seq_len) {
    for (int i = 0; i < seq_len; i++) {
        const float* te = g_w.tok_embed + token_ids[i] * N_EMBD;
        const float* pe = g_w.pos_embed + i * N_EMBD;
        float* o = out + i * N_EMBD;
        for (int j = 0; j < N_EMBD; j++)
            o[j] = te[j] + pe[j];
    }
}

/* ================================================================
   dot product of two vectors
   ================================================================ */
static float dot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

/* ================================================================
   Multi-Head Attention block
   ================================================================ */
static void mha_block(float* out, const float* hidden, int seq_len, int layer,
                      float* buf_qkv, float* buf_scores, float* buf_ctx) {
    const float* cattn_w = layer_ptr(layer, OFF_CATTN_W);
    const float* cattn_b = layer_ptr(layer, OFF_CATTN_B);
    const float* cproj_w = layer_ptr(layer, OFF_CPROJ_W);
    const float* cproj_b = layer_ptr(layer, OFF_CPROJ_B);

    // QKV = hidden @ cattn_w + cattn_b   [seq_len, QKV_DIM]
    gemm(buf_qkv, hidden, cattn_w, cattn_b, seq_len, N_EMBD, QKV_DIM);

    // zero context buffer
    memset(buf_ctx, 0, static_cast<size_t>(seq_len) * N_EMBD * sizeof(float));

    for (int h = 0; h < N_HEADS; h++) {
        // compute attention scores for this head
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j <= i; j++) {
                // Q[i] at buf_qkv[i*QKV + h*HEAD_DIM]
                const float* q = buf_qkv + i * QKV_DIM + h * HEAD_DIM;
                // K[j] at buf_qkv[j*QKV + N_EMBD + h*HEAD_DIM]
                const float* k = buf_qkv + j * QKV_DIM + N_EMBD + h * HEAD_DIM;
                float score = dot(q, k, HEAD_DIM) * 0.125f;  // 1/sqrt(64)
                buf_scores[i * seq_len + j] = score;
            }
            for (int j = i + 1; j < seq_len; j++) {
                buf_scores[i * seq_len + j] = -1e9f;
            }
            // softmax this row
            softmax_row(buf_scores + i * seq_len, seq_len);
        }

        // compute context: ctx[i, h*HEAD_DIM + d] = sum_j probs[i,j] * V[j,h,d]
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                float acc = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    float prob = buf_scores[i * seq_len + j];
                    // V[j] at buf_qkv[j*QKV + 2*N_EMBD + h*HEAD_DIM + d]
                    float v_val = buf_qkv[j * QKV_DIM + 2 * N_EMBD + h * HEAD_DIM + d];
                    acc += prob * v_val;
                }
                buf_ctx[i * N_EMBD + h * HEAD_DIM + d] = acc;
            }
        }
    }

    // projection: out = ctx @ cproj_w + cproj_b
    gemm(out, buf_ctx, cproj_w, cproj_b, seq_len, N_EMBD, N_EMBD);
}

/* ================================================================
   MLP block
   ================================================================ */
static void mlp_block(float* out, const float* hidden, int seq_len, int layer,
                      float* buf_large) {
    const float* cfc_w    = layer_ptr(layer, OFF_CFC_W);
    const float* cfc_b    = layer_ptr(layer, OFF_CFC_B);
    const float* cproj2_w = layer_ptr(layer, OFF_CPROJ2_W);
    const float* cproj2_b = layer_ptr(layer, OFF_CPROJ2_B);

    // h1 = gelu(hidden @ cfc_w + cfc_b)
    gemm(buf_large, hidden, cfc_w, cfc_b, seq_len, N_EMBD, N_FFN);
    gelu_inplace(buf_large, seq_len * N_FFN);

    // out = h1 @ cproj2_w + cproj2_b
    gemm(out, buf_large, cproj2_w, cproj2_b, seq_len, N_FFN, N_EMBD);
}

/* ================================================================
   Transformer layer
   ================================================================ */
static void transformer_layer(float* hidden, int seq_len, int layer,
                               float* buf_tmp, float* buf_qkv,
                               float* buf_scores, float* buf_ctx,
                               float* buf_large) {
    // attention
    mha_block(buf_tmp, hidden, seq_len, layer, buf_qkv, buf_scores, buf_ctx);
    vec_add_inplace(hidden, buf_tmp, seq_len * N_EMBD);
    layer_norm_inplace(hidden, layer_ptr(layer, OFF_LN1_W),
                       layer_ptr(layer, OFF_LN1_B), seq_len, N_EMBD);

    // MLP
    mlp_block(buf_tmp, hidden, seq_len, layer, buf_large);
    vec_add_inplace(hidden, buf_tmp, seq_len * N_EMBD);
    layer_norm_inplace(hidden, layer_ptr(layer, OFF_LN2_W),
                       layer_ptr(layer, OFF_LN2_B), seq_len, N_EMBD);
}

/* ================================================================
   LM head: logits[v] = dot(hidden_last, tok_embed[v])
   ================================================================ */
static void lm_head(float* logits, const float* hidden_last) {
    for (int v = 0; v < VOCAB_SIZE; v++) {
        const float* te = g_w.tok_embed + v * N_EMBD;
        float sum = 0.0f;
        for (int j = 0; j < N_EMBD; j++)
            sum += hidden_last[j] * te[j];
        logits[v] = sum;
    }
}

/* ================================================================
   Argmax
   ================================================================ */
static int argmax(const float* data, int len) {
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < len; i++) {
        if (data[i] > best_val) {
            best_val = data[i];
            best = i;
        }
    }
    return best;
}

/* ================================================================
   Main
   ================================================================ */
int main() {
    /* ── load packed files ── */
    MappedFile tok_file = load_file("../packed/tokenizer.bin");
    check_magic(tok_file.base, MAGIC_TKR1, "Invalid tokenizer.bin magic");

    MappedFile merges_file = load_file("../packed/merges.bin");
    check_magic(merges_file.base, MAGIC_MRG1, "Invalid merges.bin magic");

    MappedFile weights_file = load_file("../packed/weights.bin");
    check_magic(weights_file.base, MAGIC_WGT1, "Invalid weights.bin magic");

    /* ── setup ── */
    tokenizer_setup(static_cast<const uint8_t*>(tok_file.base),
                    static_cast<const uint8_t*>(merges_file.base));
    weights_setup(static_cast<const uint8_t*>(weights_file.base));

    fprintf(stderr, "Packed files loaded\n");

    /* ── allocate scratch buffers ── */
    float* buf_hidden = static_cast<float*>(malloc(N_CTX * N_EMBD * sizeof(float)));
    float* buf_tmp    = static_cast<float*>(malloc(N_CTX * N_EMBD * sizeof(float)));
    float* buf_ctx    = static_cast<float*>(malloc(N_CTX * N_EMBD * sizeof(float)));
    float* buf_qkv    = static_cast<float*>(malloc(N_CTX * QKV_DIM * sizeof(float)));
    float* buf_large  = static_cast<float*>(malloc(static_cast<size_t>(N_CTX) * N_FFN * sizeof(float)));
    float* buf_scores = static_cast<float*>(malloc(static_cast<size_t>(N_CTX) * N_CTX * sizeof(float)));
    float* buf_logits = static_cast<float*>(malloc(VOCAB_SIZE * sizeof(float)));

    if (!buf_hidden || !buf_tmp || !buf_ctx || !buf_qkv ||
        !buf_large || !buf_scores || !buf_logits) {
        fprintf(stderr, "Out of memory\n");
        return 1;
    }

    /* ── tokenize prompt ── */
    const char* prompt = "The sky is";
    int token_ids[N_CTX];
    int seq_len = tokenize_prompt(prompt, token_ids, N_CTX);

    /* ── print prompt (decoded) ── */
    bool pending_space = false;
    for (int i = 0; i < seq_len; i++)
        detok_print(token_ids[i], pending_space);

    /* ── generation loop ── */
    for (int step = 0; step < GEN_TOKENS; step++) {
        // embedding
        embedding(buf_hidden, token_ids, seq_len);

        // transformer layers
        for (int l = 0; l < N_LAYERS; l++) {
            transformer_layer(buf_hidden, seq_len, l,
                              buf_tmp, buf_qkv, buf_scores, buf_ctx, buf_large);
        }

        // lm_head on last position
        lm_head(buf_logits, buf_hidden + (seq_len - 1) * N_EMBD);

        // argmax
        int next_id = argmax(buf_logits, VOCAB_SIZE);
        token_ids[seq_len++] = next_id;

        // print token
        detok_print(next_id, pending_space);
    }

    printf("\n");

    /* cleanup */
    free(buf_hidden);
    free(buf_tmp);
    free(buf_ctx);
    free(buf_qkv);
    free(buf_large);
    free(buf_scores);
    free(buf_logits);

    return 0;
}
