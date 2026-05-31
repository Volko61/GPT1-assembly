// ── Model dimensions ──
.equ N_EMBD,       768
.equ N_HEADS,       12
.equ HEAD_DIM,      64
.equ N_LAYERS,      12
.equ N_POSITIONS,  512
.equ N_CTX,        512
.equ VOCAB_SIZE,  40478
.equ N_FFN,       3072
.equ QKV_DIM,     2304

// ── Tokenizer / packing ──
.equ UNK_ID,       0
.equ MERGES_COUNT, 40000
.equ TOKENIZER_HEADER_SIZE, 64
.equ MERGES_HEADER_SIZE,    64
.equ WEIGHTS_HEADER_SIZE,   64
.equ MAGIC_TKR1,  0x31524b54
.equ MAGIC_MRG1,  0x3147524d
.equ MAGIC_WGT1,  0x31544757
.equ HASH_EMPTY,  0xFFFFFFFF

// ── Embedding byte sizes ──
.equ TOK_EMBED_BYTES,  VOCAB_SIZE * N_EMBD * 4
.equ POS_EMBED_BYTES,  N_POSITIONS * N_EMBD * 4

// ── Per-layer weight byte sizes ──
.equ CATTN_W_BYTES,  N_EMBD * QKV_DIM * 4
.equ CATTN_B_BYTES,  QKV_DIM * 4
.equ CPROJ_W_BYTES,  N_EMBD * N_EMBD * 4
.equ VEC768_BYTES,   N_EMBD * 4
.equ CFC_W_BYTES,    N_EMBD * N_FFN * 4
.equ CFC_B_BYTES,    N_FFN * 4
.equ CPROJ2_W_BYTES, N_FFN * N_EMBD * 4

// ── Within-layer byte offsets ──
.equ OFF_CATTN_W,  0
.equ OFF_CATTN_B,  OFF_CATTN_W  + CATTN_W_BYTES
.equ OFF_CPROJ_W,  OFF_CATTN_B  + CATTN_B_BYTES
.equ OFF_CPROJ_B,  OFF_CPROJ_W  + CPROJ_W_BYTES
.equ OFF_LN1_W,    OFF_CPROJ_B  + VEC768_BYTES
.equ OFF_LN1_B,    OFF_LN1_W    + VEC768_BYTES
.equ OFF_CFC_W,    OFF_LN1_B    + VEC768_BYTES
.equ OFF_CFC_B,    OFF_CFC_W    + CFC_W_BYTES
.equ OFF_CPROJ2_W, OFF_CFC_B    + CFC_B_BYTES
.equ OFF_CPROJ2_B, OFF_CPROJ2_W + CPROJ2_W_BYTES
.equ OFF_LN2_W,    OFF_CPROJ2_B + VEC768_BYTES
.equ OFF_LN2_B,    OFF_LN2_W    + VEC768_BYTES
.equ LAYER_BYTES,  OFF_LN2_B    + VEC768_BYTES

// ── Generation ──
.equ GEN_TOKENS, 10
