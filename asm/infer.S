/* ================================================================
   GPT-1 Inference in pure ARM64 Assembly (AArch64, Linux)
   ================================================================ */
.text
.include "model.inc"
.global main

.extern open, lseek, mmap, close
.extern printf, puts, exit, malloc, free, strlen, write
.extern expf, tanhf

.equ O_RDONLY,   0
.equ SEEK_SET,   0
.equ SEEK_END,   2
.equ PROT_READ,  1
.equ MAP_PRIVATE,2
.equ STDOUT_FD,  1

.equ FNV_OFFSET_BASIS_64, 0xcbf29ce484222325
.equ FNV_PRIME_64,        0x100000001b3

/* ────────────────────── BSS ────────────────────── */
.section .bss
.align 3

/* tokenizer globals (existing) */
g_tok_base:     .skip 8
g_merges_base:  .skip 8
g_vocab_size:   .skip 4
g_unk_id:       .skip 4
g_lowercase:    .skip 4
g_suffix_len:   .skip 4
g_suffix_ptr:   .skip 8
g_offsets_ptr:  .skip 8
g_strings_ptr:  .skip 8
g_hash_ptr:     .skip 8
g_hash_len:     .skip 4
g_ids_ptr:      .skip 8
g_merges_count: .skip 4
g_merges_ptr:   .skip 8

scratch_pieces: .skip N_CTX*4
scratch_tmp:    .skip 16
out_ids:        .skip N_CTX*4

/* weight pointers */
g_w_tok_embed:   .skip 8
g_w_pos_embed:   .skip 8
g_w_layers_base: .skip 8

/* scratch buffer pointers (malloc'd) */
g_buf_hidden:  .skip 8
g_buf_tmp:     .skip 8
g_buf_tmp2:    .skip 8
g_buf_large:   .skip 8
g_buf_scores:  .skip 8
g_buf_logits:  .skip 8

/* token id array for generation */
g_token_ids:   .skip N_CTX*4
g_num_tokens:  .skip 4
g_pending_space:.skip 4

/* ────────────────────── TEXT ────────────────────── */
.section .text

/* ================================================================
   load_file  –  mmap a file read-only
   x0 = path (char*)
   Returns x0 = base ptr, x1 = length
   ================================================================ */
.type load_file, %function
load_file:
    stp x29,x30,[sp,-64]!
    mov x29,sp
    stp x19,x20,[sp,16]
    stp x21,x22,[sp,32]

    mov x1,#O_RDONLY
    bl  open
    cmp x0,#0
    blt load_file_fail
    mov x19,x0

    mov x0,x19; mov x1,#0; mov x2,#SEEK_END
    bl  lseek
    mov x21,x0

    mov x0,x19; mov x1,#0; mov x2,#SEEK_SET
    bl  lseek

    mov x0,#0; mov x1,x21; mov x2,#PROT_READ
    mov x3,#MAP_PRIVATE; mov x4,x19; mov x5,#0
    bl  mmap
    cmp x0,#-1
    beq load_file_fail
    mov x20,x0

    mov x0,x19; bl close

    mov x0,x20; mov x1,x21
    ldp x21,x22,[sp,32]
    ldp x19,x20,[sp,16]
    ldp x29,x30,[sp],64
    ret
load_file_fail:
    adrp x0,msg_load_fail; add x0,x0,:lo12:msg_load_fail
    bl puts; mov w0,#1; bl exit

/* ================================================================
   check_magic  –  x0=base, w1=expected, x3=errmsg
   ================================================================ */
.type check_magic, %function
check_magic:
    stp x29,x30,[sp,-16]!; mov x29,sp
    ldr w2,[x0]
    cmp w2,w1
    beq 1f
    mov x0,x3; bl puts; mov w0,#1; bl exit
1:  ldp x29,x30,[sp],16; ret

/* ================================================================
   tokenizer_setup  –  x0=tok_base, x1=merges_base
   ================================================================ */
.type tokenizer_setup, %function
tokenizer_setup:
    stp x29,x30,[sp,-16]!; mov x29,sp

    adrp x2,g_tok_base; add x2,x2,:lo12:g_tok_base; str x0,[x2]
    adrp x2,g_merges_base; add x2,x2,:lo12:g_merges_base; str x1,[x2]

    ldr w3,[x0,#8];  adrp x2,g_vocab_size; add x2,x2,:lo12:g_vocab_size; str w3,[x2]
    ldr w3,[x0,#12]; adrp x2,g_unk_id;     add x2,x2,:lo12:g_unk_id;     str w3,[x2]
    ldr w3,[x0,#16]; adrp x2,g_lowercase;  add x2,x2,:lo12:g_lowercase;  str w3,[x2]
    ldr w3,[x0,#20]; adrp x2,g_suffix_len; add x2,x2,:lo12:g_suffix_len; str w3,[x2]

    ldr w4,[x0,#24]; add x4,x0,x4
    adrp x2,g_suffix_ptr; add x2,x2,:lo12:g_suffix_ptr; str x4,[x2]

    ldr w4,[x0,#28]; add x4,x0,x4
    adrp x2,g_offsets_ptr; add x2,x2,:lo12:g_offsets_ptr; str x4,[x2]

    ldr w4,[x0,#32]; add x4,x0,x4
    adrp x2,g_strings_ptr; add x2,x2,:lo12:g_strings_ptr; str x4,[x2]

    ldr w4,[x0,#36]; add x4,x0,x4
    adrp x2,g_hash_ptr; add x2,x2,:lo12:g_hash_ptr; str x4,[x2]

    ldr w3,[x0,#40]; adrp x2,g_hash_len; add x2,x2,:lo12:g_hash_len; str w3,[x2]

    ldr w4,[x0,#44]; add x4,x0,x4
    adrp x2,g_ids_ptr; add x2,x2,:lo12:g_ids_ptr; str x4,[x2]

    ldr w3,[x1,#8];  adrp x2,g_merges_count; add x2,x2,:lo12:g_merges_count; str w3,[x2]

    ldr w4,[x1,#12]; add x4,x1,x4
    adrp x2,g_merges_ptr; add x2,x2,:lo12:g_merges_ptr; str x4,[x2]

    ldp x29,x30,[sp],16; ret

/* ================================================================
   is_space / is_punct
   ================================================================ */
.type is_space, %function
is_space:
    cmp w0,#32; beq 1f; cmp w0,#9; beq 1f; cmp w0,#10; beq 1f
    cmp w0,#13; beq 1f; cmp w0,#11; beq 1f; cmp w0,#12; beq 1f
    mov w0,#0; ret
1:  mov w0,#1; ret

.type is_punct, %function
is_punct:
    adrp x1,punct_str; add x1,x1,:lo12:punct_str
2:  ldrb w2,[x1],#1; cbz w2,3f; cmp w2,w0; beq 4f; b 2b
4:  mov w0,#1; ret
3:  mov w0,#0; ret

/* ================================================================
   lookup_token  –  FNV-1a hash lookup
   x0=str, w1=len  →  w0=token_id
   ================================================================ */
.type lookup_token, %function
lookup_token:
    stp x29,x30,[sp,-16]!; mov x29,sp
    mov x9,x0; mov w10,w1

    ldr x11,=FNV_OFFSET_BASIS_64
    ldr x12,=FNV_PRIME_64
    mov w13,#0
.Lhash_loop:
    cmp w13,w10; bge .Lhash_done
    ldrb w14,[x9,x13]; eor x11,x11,x14; mul x11,x11,x12
    add w13,w13,#1; b .Lhash_loop
.Lhash_done:
    adrp x14,g_hash_ptr; add x14,x14,:lo12:g_hash_ptr; ldr x14,[x14]
    adrp x15,g_ids_ptr;  add x15,x15,:lo12:g_ids_ptr;  ldr x15,[x15]
    adrp x16,g_offsets_ptr; add x16,x16,:lo12:g_offsets_ptr; ldr x16,[x16]
    adrp x17,g_strings_ptr; add x17,x17,:lo12:g_strings_ptr; ldr x17,[x17]
    adrp x18,g_hash_len; add x18,x18,:lo12:g_hash_len; ldr w18,[x18]; uxtw x18,w18

    sub x2,x18,#1; and x3,x11,x2
.Lprobe:
    ldr w5,[x15,x3,lsl #2]; cmn w5,#1; beq .Lnotfound
    ldr x4,[x14,x3,lsl #3]; cmp x4,x11; bne .Lnext
    uxtw x6,w5; ldr w7,[x16,x6,lsl #2]
    add x8,x6,#1; ldr w0,[x16,x8,lsl #2]; sub w0,w0,w7
    cmp w0,w10; bne .Lnext
    add x8,x17,x7; mov w6,#0
.Lcmp:  cmp w6,w10; bge .Lfound
    ldrb w0,[x8,x6]; ldrb w1,[x9,x6]
    cmp w0,w1; bne .Lnext; add w6,w6,#1; b .Lcmp
.Lfound: mov w0,w5; ldp x29,x30,[sp],16; ret
.Lnext:  add x3,x3,#1; and x3,x3,x2; b .Lprobe
.Lnotfound:
    adrp x0,g_unk_id; add x0,x0,:lo12:g_unk_id; ldr w0,[x0]
    ldp x29,x30,[sp],16; ret

/* ================================================================
   merge_word  –  BPE merge on a single word
   x0=str, w1=len, x2=out_ids, w3=max  →  w0=count
   ================================================================ */
.type merge_word, %function
merge_word:
    stp x29,x30,[sp,-80]!
    mov x29,sp
    stp x19,x20,[sp,16]
    stp x21,x22,[sp,32]
    stp x23,x24,[sp,48]
    stp x25,x26,[sp,64]

    mov x19,x0; mov w20,w1; mov x21,x2; mov w22,w3

    adrp x0,g_suffix_ptr; add x0,x0,:lo12:g_suffix_ptr; ldr x25,[x0]
    adrp x0,g_suffix_len; add x0,x0,:lo12:g_suffix_len; ldr w26,[x0]

    mov w23,#0; mov w24,#0
.Lmi_loop:
    cmp w24,w20; bge .Lmi_done
    ldrb w0,[x19,x24]
    adrp x2,scratch_tmp; add x2,x2,:lo12:scratch_tmp
    strb w0,[x2]
    mov w3,#1
    add w4,w24,#1; cmp w4,w20; bne .Lmi_nosuf
    cbz w26,.Lmi_nosuf
    mov w6,#0
.Lmi_sufloop:
    cmp w6,w26; bge .Lmi_sufdone
    ldrb w7,[x25,x6]; strb w7,[x2,x3]
    add w6,w6,#1; add w3,w3,#1; b .Lmi_sufloop
.Lmi_sufdone:
.Lmi_nosuf:
    mov x0,x2; uxtw x1,w3; bl lookup_token
    cmp w23,w22; bge .Lmi_next
    str w0,[x21,x23,lsl #2]; add w23,w23,#1
.Lmi_next:
    add w24,w24,#1; b .Lmi_loop
.Lmi_done:
    adrp x9,g_merges_ptr; add x9,x9,:lo12:g_merges_ptr; ldr x9,[x9]
    adrp x10,g_merges_count; add x10,x10,:lo12:g_merges_count; ldr w10,[x10]
    mov w11,#0
.Lmr_loop:
    cmp w11,w10; bge .Lm_done
    uxtw x12,w11; mov x13,#12; madd x14,x12,x13,x9
    ldr w15,[x14]; ldr w16,[x14,#4]; ldr w17,[x14,#8]
    mov w12,#0
.Lms_loop:
    subs w18,w23,#1; cmp w12,w18; bge .Lmr_next
    uxtw x12,w12; ldr w0,[x21,x12,lsl #2]; add w1,w12,#1; uxtw x1,w1; ldr w2,[x21,x1,lsl #2]
    cmp w0,w15; bne .Lms_next; cmp w2,w16; bne .Lms_next
    str w17,[x21,x12,lsl #2]; mov w3,w1
.Lmsh:  add w4,w3,#1; cmp w4,w23; bge .Lmshd
    uxtw x4,w4; ldr w5,[x21,x4,lsl #2]; uxtw x3,w3; str w5,[x21,x3,lsl #2]; add w3,w4,#0; b .Lmsh
.Lmshd: sub w23,w23,#1; b .Lms_loop
.Lms_next: add w12,w12,#1; b .Lms_loop
.Lmr_next: add w11,w11,#1; b .Lmr_loop
.Lm_done:
    mov w0,w23
    ldp x25,x26,[sp,64]; ldp x23,x24,[sp,48]
    ldp x21,x22,[sp,32]; ldp x19,x20,[sp,16]
    ldp x29,x30,[sp],80; ret

/* ================================================================
   append_pieces / process_word
   ================================================================ */
.type append_pieces, %function
append_pieces:
    stp x29,x30,[sp,-64]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]; stp x23,x24,[sp,48]
    mov x19,x0; mov w20,w1; mov x21,x2; mov w22,w3; mov w23,#N_CTX
    mov w24,#0
.Lap: cmp w24,w20; bge .Lapd; cmp w22,w23; bge .Lapd
    uxtw x24,w24; ldr w0,[x19,x24,lsl #2]
    uxtw x22,w22; str w0,[x21,x22,lsl #2]
    add w22,w22,#1; add w24,w24,#1; b .Lap
.Lapd: mov w0,w22
    ldp x23,x24,[sp,48]; ldp x21,x22,[sp,32]
    ldp x19,x20,[sp,16]; ldp x29,x30,[sp],64; ret

.type process_word, %function
process_word:
    stp x29,x30,[sp,-32]!; mov x29,sp
    stp x19,x20,[sp,16]
    mov x19,x2; mov w20,w3
    adrp x2,scratch_pieces; add x2,x2,:lo12:scratch_pieces
    mov w3,#N_CTX; bl merge_word
    mov w1,w0
    adrp x0,scratch_pieces; add x0,x0,:lo12:scratch_pieces
    mov x2,x19; mov w3,w20
    bl append_pieces
    ldp x19,x20,[sp,16]
    ldp x29,x30,[sp],32; ret

/* ================================================================
   tokenize_prompt  –  x0=text, x1=out_ids, w2=max  →  w0=count
   ================================================================ */
.type tokenize_prompt, %function
tokenize_prompt:
    stp x29,x30,[sp,-80]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]
    stp x23,x24,[sp,48]; stp x25,x26,[sp,64]

    mov x19,x0; mov x20,x1; mov w21,w2; mov x22,#0
    adrp x0,g_lowercase; add x0,x0,:lo12:g_lowercase; ldr w0,[x0]
    cbz w0,.Ltnl
    mov x0,x19; bl strlen; add x1,x0,#1; mov x0,x1; bl malloc
    cbz x0,.Ltnl; mov x22,x0; mov x23,x19; mov x24,x22
.Ltcl: ldrb w0,[x23],#1; cbz w0,.Ltcd
    cmp w0,#'A'; blt .Ltcs; cmp w0,#'Z'; bgt .Ltcs; add w0,w0,#32
.Ltcs: strb w0,[x24],#1; b .Ltcl
.Ltcd: mov w0,#0; strb w0,[x24]; mov x19,x22
.Ltnl:
    mov w23,#0; mov x24,#0; mov w25,#0; mov x26,x19
.Ltlp: ldrb w0,[x26]; cbz w0,.Ltdn; mov w1,w0
    mov w0,w1; bl is_space;  cbnz w0,.Ltsp
    mov w0,w1; bl is_punct;  cbnz w0,.Ltpu
    cbnz w25,.Ltcn; mov x24,x26
.Ltcn: add w25,w25,#1; add x26,x26,#1; b .Ltlp
.Ltsp: cbz w25,.Ltss
    mov x0,x24; mov w1,w25; mov x2,x20; mov w3,w23; mov w4,w21
    bl process_word; mov w23,w0; mov w25,#0; mov x24,#0
.Ltss: add x26,x26,#1; b .Ltlp
.Ltpu: cbz w25,.Ltpo
    mov x0,x24; mov w1,w25; mov x2,x20; mov w3,w23; mov w4,w21
    bl process_word; mov w23,w0; mov w25,#0; mov x24,#0
.Ltpo: mov x0,x26; mov w1,#1; mov x2,x20; mov w3,w23; mov w4,w21
    bl process_word; mov w23,w0; mov w25,#0; mov x24,#0
    add x26,x26,#1; b .Ltlp
.Ltdn: cbz w25,.Lten
    mov x0,x24; mov w1,w25; mov x2,x20; mov w3,w23; mov w4,w21
    bl process_word; mov w23,w0
.Lten: cbz x22,.Ltrt; mov x0,x22; bl free
.Ltrt: mov w0,w23
    ldp x25,x26,[sp,64]; ldp x23,x24,[sp,48]
    ldp x21,x22,[sp,32]; ldp x19,x20,[sp,16]
    ldp x29,x30,[sp],80; ret

/* ================================================================
   print_tokens  –  debug helper
   ================================================================ */
.type print_tokens, %function
print_tokens:
    stp x29,x30,[sp,-48]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]
    mov x19,x0; mov w20,w1; mov w21,#0
.Lpt: cmp w21,w20; bge .Lptd
    ldr w1,[x19,x21,lsl #2]
    adrp x0,fmt_token; add x0,x0,:lo12:fmt_token; bl printf
    add w21,w21,#1; b .Lpt
.Lptd: adrp x0,msg_newline; add x0,x0,:lo12:msg_newline; bl puts
    ldp x21,x22,[sp,32]; ldp x19,x20,[sp,16]
    ldp x29,x30,[sp],48; ret

/* ================================================================
   weights_setup  –  x0 = mmap'd weights.bin base
   Stores pointers to tok_embed, pos_embed, layers_base
   ================================================================ */
.type weights_setup, %function
weights_setup:
    stp x29,x30,[sp,-16]!; mov x29,sp
    add x1,x0,#WEIGHTS_HEADER_SIZE
    adrp x2,g_w_tok_embed; add x2,x2,:lo12:g_w_tok_embed; str x1,[x2]

    ldr x3,=TOK_EMBED_BYTES; add x1,x1,x3
    adrp x2,g_w_pos_embed; add x2,x2,:lo12:g_w_pos_embed; str x1,[x2]

    ldr x3,=POS_EMBED_BYTES; add x1,x1,x3
    adrp x2,g_w_layers_base; add x2,x2,:lo12:g_w_layers_base; str x1,[x2]

    ldp x29,x30,[sp],16; ret

/* ================================================================
   get_layer_ptr  –  helper: x0 = layer_base + w1*LAYER_BYTES + x2
   w0 = layer index, x1 = offset within layer
   Returns x0 = pointer
   ================================================================ */
.type get_layer_ptr, %function
get_layer_ptr:
    adrp x2,g_w_layers_base; add x2,x2,:lo12:g_w_layers_base; ldr x2,[x2]
    ldr x3,=LAYER_BYTES; uxtw x4,w0; mul x4,x4,x3; add x2,x2,x4
    add x0,x2,x1; ret

/* ================================================================
   GEMM  –  C[M,N] = A[M,K] @ B[K,N] + bias[N]
   x0=C  x1=A  x2=B  x3=bias(or 0)  w4=M  w5=K  w6=N
   N must be a multiple of 16.
   ================================================================ */
.type gemm, %function
gemm:
    stp x29,x30,[sp,-96]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]
    stp x23,x24,[sp,48]; stp x25,x26,[sp,64]
    stp x27,x28,[sp,80]

    mov x19,x0; mov x20,x1; mov x21,x2; mov x22,x3
    uxtw x23,w4; uxtw x24,w5; uxtw x25,w6
    lsl x27,x25,#2              // N_bytes
    lsl x28,x24,#2              // K_bytes

    mov x26,#0                  // i
.Lg_row:
    cmp x26,x23; bge .Lg_done
    mul x9,x26,x27;  add x9,x19,x9    // C_row
    mul x10,x26,x28; add x10,x20,x10  // A_row
    mov x11,#0                          // j (tile start)
.Lg_tile:
    cmp x11,x25; bge .Lg_nrow
    cbz x22,.Lg_zinit
    add x0,x22,x11,lsl #2
    ldp q20,q21,[x0]; ldp q22,q23,[x0,#32]; b .Lg_kstart
.Lg_zinit:
    movi v20.4s,#0; movi v21.4s,#0; movi v22.4s,#0; movi v23.4s,#0
.Lg_kstart:
    mov x12,x10                         // A ptr
    add x13,x21,x11,lsl #2             // B col ptr
    mov x14,#0                          // k
.Lg_k:
    cmp x14,x24; bge .Lg_store
    ldr s0,[x12],#4; dup v0.4s,v0.s[0]
    ldp q1,q2,[x13]; ldp q3,q4,[x13,#32]
    fmla v20.4s,v1.4s,v0.4s; fmla v21.4s,v2.4s,v0.4s
    fmla v22.4s,v3.4s,v0.4s; fmla v23.4s,v4.4s,v0.4s
    add x13,x13,x27                    // next B row
    add x14,x14,#1; b .Lg_k
.Lg_store:
    add x0,x9,x11,lsl #2
    stp q20,q21,[x0]; stp q22,q23,[x0,#32]
    add x11,x11,#16; b .Lg_tile
.Lg_nrow:
    add x26,x26,#1; b .Lg_row
.Lg_done:
    ldp x27,x28,[sp,80]; ldp x25,x26,[sp,64]
    ldp x23,x24,[sp,48]; ldp x21,x22,[sp,32]
    ldp x19,x20,[sp,16]; ldp x29,x30,[sp],96; ret

/* ================================================================
   vec_add_inplace  –  dst[i] += src[i]
   x0=dst  x1=src  w2=count (multiple of 4)
   ================================================================ */
.type vec_add_inplace, %function
vec_add_inplace:
    uxtw x2,w2; mov x3,#0
.Lva: cmp x3,x2; bge .Lvad
    add x4,x0,x3,lsl #2; add x5,x1,x3,lsl #2
    ldp q0,q1,[x4]; ldp q2,q3,[x5]
    fadd v0.4s,v0.4s,v2.4s; fadd v1.4s,v1.4s,v3.4s
    stp q0,q1,[x4]
    add x3,x3,#8; b .Lva
.Lvad: ret

/* ================================================================
   layer_norm_inplace
   x0=data[rows*cols]  x1=weight[cols]  x2=bias[cols]
   w3=rows  w4=cols (must be mult of 4)
   ================================================================ */
.type layer_norm_inplace, %function
layer_norm_inplace:
    stp x29,x30,[sp,-80]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]
    stp x23,x24,[sp,48]; stp d8,d9,[sp,64]

    mov x19,x0; mov x20,x1; mov x21,x2
    uxtw x22,w3; uxtw x23,w4
    lsl x24,x23,#2              // cols_bytes
    mov x9,#0                   // row idx
.Lln_row:
    cmp x9,x22; bge .Lln_done
    mul x10,x9,x24; add x10,x19,x10   // row ptr

    /* mean = sum / cols */
    movi v0.4s,#0; mov x11,#0
.Lln_sum:
    cmp x11,x23; bge .Lln_sumdn
    add x12,x10,x11,lsl #2; ldr q1,[x12]
    fadd v0.4s,v0.4s,v1.4s
    add x11,x11,#4; b .Lln_sum
.Lln_sumdn:
    faddp v0.4s,v0.4s,v0.4s; faddp s0,v0.2s
    ucvtf s1,w23; fdiv s8,s0,s1        // s8 = mean

    /* var = sum((x-mean)^2) / cols */
    dup v8.4s,v8.s[0]
    movi v0.4s,#0; mov x11,#0
.Lln_var:
    cmp x11,x23; bge .Lln_vardn
    add x12,x10,x11,lsl #2; ldr q1,[x12]
    fsub v1.4s,v1.4s,v8.4s; fmul v1.4s,v1.4s,v1.4s
    fadd v0.4s,v0.4s,v1.4s
    add x11,x11,#4; b .Lln_var
.Lln_vardn:
    faddp v0.4s,v0.4s,v0.4s; faddp s0,v0.2s
    ucvtf s1,w23; fdiv s0,s0,s1                       // var
    adrp x12,const_eps; add x12,x12,:lo12:const_eps; ldr s2,[x12]
    fadd s0,s0,s2; fsqrt s0,s0
    fmov s2,#1.0; fdiv s9,s2,s0        // s9 = 1/sqrt(var+eps)
    dup v9.4s,v9.s[0]

    /* normalize: x = weight * (x - mean) * inv_std + bias */
    mov x11,#0
.Lln_norm:
    cmp x11,x23; bge .Lln_nrow
    add x12,x10,x11,lsl #2
    ldr q0,[x12]                        // x
    fsub v0.4s,v0.4s,v8.4s             // x - mean
    fmul v0.4s,v0.4s,v9.4s             // * inv_std
    add x13,x20,x11,lsl #2; ldr q1,[x13]  // weight
    fmul v0.4s,v0.4s,v1.4s
    add x13,x21,x11,lsl #2; ldr q2,[x13]  // bias
    fadd v0.4s,v0.4s,v2.4s
    str q0,[x12]
    add x11,x11,#4; b .Lln_norm
.Lln_nrow:
    add x9,x9,#1; b .Lln_row
.Lln_done:
    ldp d8,d9,[sp,64]; ldp x23,x24,[sp,48]
    ldp x21,x22,[sp,32]; ldp x19,x20,[sp,16]
    ldp x29,x30,[sp],80; ret

/* ================================================================
   softmax_row  –  in-place softmax of a float vector
   x0=data  w1=len
   ================================================================ */
.type softmax_row, %function
softmax_row:
    stp x29,x30,[sp,-64]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]; stp d8,d9,[sp,48]
    mov x19,x0; uxtw x20,w1

    /* find max */
    ldr s8,[x19]
    mov x21,#1
.Lsm_mx:
    cmp x21,x20; bge .Lsm_mxd
    ldr s0,[x19,x21,lsl #2]; fmax s8,s8,s0
    add x21,x21,#1; b .Lsm_mx
.Lsm_mxd:
    /* exp(x-max) and sum */
    fmov s9,wzr                          // sum = 0
    mov x21,#0
.Lsm_exp:
    cmp x21,x20; bge .Lsm_div
    ldr s0,[x19,x21,lsl #2]; fsub s0,s0,s8
    bl expf                              // s0 = exp(x-max)
    str s0,[x19,x21,lsl #2]; fadd s9,s9,s0
    add x21,x21,#1; b .Lsm_exp
.Lsm_div:
    /* divide each by sum */
    mov x21,#0
.Lsm_d:
    cmp x21,x20; bge .Lsm_done
    ldr s0,[x19,x21,lsl #2]; fdiv s0,s0,s9
    str s0,[x19,x21,lsl #2]
    add x21,x21,#1; b .Lsm_d
.Lsm_done:
    ldp d8,d9,[sp,48]; ldp x21,x22,[sp,32]
    ldp x19,x20,[sp,16]; ldp x29,x30,[sp],64; ret

/* ================================================================
   gelu_inplace  –  x0=data  w1=count
   gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
   Uses tanhf from libm for correctness.
   ================================================================ */
.type gelu_inplace, %function
gelu_inplace:
    stp x29,x30,[sp,-48]!; mov x29,sp
    stp x19,x20,[sp,16]; stp d8,d9,[sp,32]
    mov x19,x0; uxtw x20,w1; mov x9,#0
    adrp x10,const_sqrt2pi; add x10,x10,:lo12:const_sqrt2pi; ldr s8,[x10]
    adrp x10,const_gelu_c;  add x10,x10,:lo12:const_gelu_c;  ldr s9,[x10]
.Lgelu:
    cmp x9,x20; bge .Lgelud
    ldr s0,[x19,x9,lsl #2]             // x
    fmul s1,s0,s0; fmul s1,s1,s0       // x^3
    fmul s1,s1,s9                       // 0.044715*x^3
    fadd s1,s0,s1                       // x + 0.044715*x^3
    fmul s0,s1,s8                       // sqrt(2/pi) * (...)
    /* save x on stack, call tanhf */
    str s0,[sp,#-16]!
    str x9,[sp,#8]                      // save loop counter
    bl tanhf                            // s0 = tanh(arg)
    ldr x9,[sp,#8]
    fmov s1,s0                          // tanh result
    ldr s0,[sp],#16                     // restore (not needed, discard)
    /* reload original x */
    ldr s2,[x19,x9,lsl #2]
    fmov s3,#1.0; fadd s1,s3,s1        // 1 + tanh(...)
    fmov s3,#0.5; fmul s1,s1,s3        // 0.5*(1+tanh)
    fmul s0,s2,s1                       // x * 0.5*(1+tanh)
    str s0,[x19,x9,lsl #2]
    add x9,x9,#1; b .Lgelu
.Lgelud:
    ldp d8,d9,[sp,32]; ldp x19,x20,[sp,16]
    ldp x29,x30,[sp],48; ret

/* ================================================================
   embedding  –  out[i] = tok_embed[ids[i]] + pos_embed[i]
   x0=out  x1=token_ids  w2=seq_len
   ================================================================ */
.type embedding, %function
embedding:
    stp x29,x30,[sp,-64]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]; stp x23,x24,[sp,48]
    mov x19,x0; mov x20,x1; uxtw x21,w2

    adrp x0,g_w_tok_embed; add x0,x0,:lo12:g_w_tok_embed; ldr x22,[x0]
    adrp x0,g_w_pos_embed; add x0,x0,:lo12:g_w_pos_embed; ldr x23,[x0]
    mov x24,#(N_EMBD*4)            // row bytes

    mov x9,#0
.Lemb:
    cmp x9,x21; bge .Lembd
    /* tok_embed row */
    ldr w10,[x20,x9,lsl #2]; uxtw x10,w10
    mul x10,x10,x24; add x11,x22,x10   // tok_embed[id]
    /* pos_embed row */
    mul x12,x9,x24; add x13,x23,x12    // pos_embed[i]
    /* out row */
    mul x14,x9,x24; add x15,x19,x14

    mov x16,#0
.Lemb_add:
    cmp x16,#N_EMBD; bge .Lemb_next
    lsl x17,x16,#2
    add x0,x11,x17; ldp q0,q1,[x0]
    add x0,x13,x17; ldp q2,q3,[x0]
    fadd v0.4s,v0.4s,v2.4s; fadd v1.4s,v1.4s,v3.4s
    add x0,x15,x17; stp q0,q1,[x0]
    add x16,x16,#8; b .Lemb_add
.Lemb_next:
    add x9,x9,#1; b .Lemb
.Lembd:
    ldp x23,x24,[sp,48]; ldp x21,x22,[sp,32]
    ldp x19,x20,[sp,16]; ldp x29,x30,[sp],64; ret

/* ================================================================
   dot64  –  dot product of two 64-float vectors
   x0=a  x1=b  →  s0 = result
   ================================================================ */
.type dot64, %function
dot64:
    movi v0.4s,#0; movi v1.4s,#0; movi v2.4s,#0; movi v3.4s,#0
    ldp q16,q17,[x0,#0];   ldp q18,q19,[x1,#0]
    fmla v0.4s,v16.4s,v18.4s; fmla v1.4s,v17.4s,v19.4s
    ldp q16,q17,[x0,#32];  ldp q18,q19,[x1,#32]
    fmla v2.4s,v16.4s,v18.4s; fmla v3.4s,v17.4s,v19.4s
    ldp q16,q17,[x0,#64];  ldp q18,q19,[x1,#64]
    fmla v0.4s,v16.4s,v18.4s; fmla v1.4s,v17.4s,v19.4s
    ldp q16,q17,[x0,#96];  ldp q18,q19,[x1,#96]
    fmla v2.4s,v16.4s,v18.4s; fmla v3.4s,v17.4s,v19.4s
    ldp q16,q17,[x0,#128]; ldp q18,q19,[x1,#128]
    fmla v0.4s,v16.4s,v18.4s; fmla v1.4s,v17.4s,v19.4s
    ldp q16,q17,[x0,#160]; ldp q18,q19,[x1,#160]
    fmla v2.4s,v16.4s,v18.4s; fmla v3.4s,v17.4s,v19.4s
    ldp q16,q17,[x0,#192]; ldp q18,q19,[x1,#192]
    fmla v0.4s,v16.4s,v18.4s; fmla v1.4s,v17.4s,v19.4s
    ldp q16,q17,[x0,#224]; ldp q18,q19,[x1,#224]
    fmla v2.4s,v16.4s,v18.4s; fmla v3.4s,v17.4s,v19.4s
    fadd v0.4s,v0.4s,v2.4s; fadd v1.4s,v1.4s,v3.4s
    fadd v0.4s,v0.4s,v1.4s
    faddp v0.4s,v0.4s,v0.4s; faddp s0,v0.2s
    ret

/* ================================================================
   mha_block  –  multi-head self-attention
   x0=out[seq*768]  x1=hidden[seq*768]  w2=seq_len  w3=layer
   Uses: g_buf_large (QKV), g_buf_scores, g_buf_tmp2 (context)
   ================================================================ */
.type mha_block, %function
mha_block:
    stp x29,x30,[sp,-112]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]
    stp x23,x24,[sp,48]; stp x25,x26,[sp,64]
    stp x27,x28,[sp,80]; stp d8,d9,[sp,96]

    mov x19,x0; mov x20,x1; uxtw x21,w2; uxtw x22,w3

    /* compute layer_base */
    adrp x0,g_w_layers_base; add x0,x0,:lo12:g_w_layers_base; ldr x0,[x0]
    ldr x1,=LAYER_BYTES; mul x1,x22,x1; add x23,x0,x1

    /* QKV gemm: g_buf_large = hidden @ cattn_w + cattn_b */
    adrp x0,g_buf_large; add x0,x0,:lo12:g_buf_large; ldr x0,[x0]
    mov x1,x20
    mov x2,x23                          // cattn_w at layer_base+0
    ldr x4,=OFF_CATTN_B; add x3,x23,x4 // cattn_b
    mov w4,w21; mov w5,#N_EMBD; mov w6,#QKV_DIM
    bl gemm

    /* load buffer pointers into callee-saved regs */
    adrp x0,g_buf_large;  add x0,x0,:lo12:g_buf_large;  ldr x25,[x0]
    adrp x0,g_buf_scores; add x0,x0,:lo12:g_buf_scores; ldr x26,[x0]
    adrp x0,g_buf_tmp2;   add x0,x0,:lo12:g_buf_tmp2;   ldr x27,[x0]

    /* zero context buffer */
    mov x0,x27; mul x1,x21,x21          // reuse x1 for size... no
    mov x1,#N_EMBD; mul x1,x21,x1; lsl x1,x1,#2
    mov x2,#0
.Lmha_z: cmp x2,x1; bge .Lmha_zd
    str wzr,[x0,x2]; add x2,x2,#4; b .Lmha_z
.Lmha_zd:

    mov x24,#0                          // h = head index
.Lmha_head:
    cmp x24,#N_HEADS; bge .Lmha_proj
    mov x20,#0                          // i = query pos (reuse x20, hidden no longer needed)
.Lmha_qi:
    cmp x20,x21; bge .Lmha_ctx

    /* ── compute attention scores for row i ── */
    mov x28,#0                          // j = key pos
.Lmha_kj:
    cmp x28,x21; bge .Lmha_sfmx
    cmp x28,x20; bgt .Lmha_mask

    /* Q ptr = qkv + (i*QKV_DIM + h*HEAD_DIM)*4 */
    mov x9,#QKV_DIM; mul x9,x20,x9
    mov x10,#HEAD_DIM; madd x9,x24,x10,x9
    add x0,x25,x9,lsl #2

    /* K ptr = qkv + (j*QKV_DIM + N_EMBD + h*HEAD_DIM)*4 */
    mov x9,#QKV_DIM; mul x9,x28,x9
    madd x9,x24,x10,x9; add x9,x9,#N_EMBD
    add x1,x25,x9,lsl #2

    bl dot64
    /* scale by 1/sqrt(64) = 0.125 */
    fmov s1,#0.125; fmul s0,s0,s1
    b .Lmha_stsc

.Lmha_mask:
    adrp x9,const_neg_inf; add x9,x9,:lo12:const_neg_inf; ldr s0,[x9]

.Lmha_stsc:
    mul x9,x20,x21; add x9,x9,x28
    str s0,[x26,x9,lsl #2]
    add x28,x28,#1; b .Lmha_kj

.Lmha_sfmx:
    /* softmax row i */
    mul x9,x20,x21; add x0,x26,x9,lsl #2; mov w1,w21
    bl softmax_row

    add x20,x20,#1; b .Lmha_qi

.Lmha_ctx:
    /* ── compute context: ctx[i,h*64+d] = Σ_j probs[i,j]*V[j,h,d] ── */
    mov x20,#0                          // i
.Lmha_ci:
    cmp x20,x21; bge .Lmha_nh
    mov x9,#0                           // d (step 4)
.Lmha_cd:
    cmp x9,#HEAD_DIM; bge .Lmha_ni
    movi v0.4s,#0                       // accumulator
    mov x10,#0                          // j
.Lmha_cj:
    cmp x10,x21; bge .Lmha_cst
    /* probs[i,j] */
    mul x11,x20,x21; add x11,x11,x10
    ldr s1,[x26,x11,lsl #2]; dup v1.4s,v1.s[0]
    /* V[j,h,d:d+3] = qkv + (j*QKV_DIM + 2*N_EMBD + h*HEAD_DIM + d)*4 */
    mov x11,#QKV_DIM; mul x11,x10,x11
    mov x12,#HEAD_DIM; madd x11,x24,x12,x11
    add x11,x11,x9; add x11,x11,#(2*N_EMBD)
    add x12,x25,x11,lsl #2
    ld1 {v2.4s},[x12]
    fmla v0.4s,v2.4s,v1.4s
    add x10,x10,#1; b .Lmha_cj
.Lmha_cst:
    /* store to ctx[i*N_EMBD + h*HEAD_DIM + d] */
    mov x11,#N_EMBD; mul x11,x20,x11
    mov x12,#HEAD_DIM; madd x11,x24,x12,x11
    add x11,x11,x9
    add x12,x27,x11,lsl #2
    st1 {v0.4s},[x12]
    add x9,x9,#4; b .Lmha_cd
.Lmha_ni:
    add x20,x20,#1; b .Lmha_ci

.Lmha_nh:
    add x24,x24,#1; b .Lmha_head

.Lmha_proj:
    /* out = context @ c_proj_w + c_proj_b */
    mov x0,x19; mov x1,x27
    ldr x4,=OFF_CPROJ_W; add x2,x23,x4
    ldr x4,=OFF_CPROJ_B; add x3,x23,x4
    mov w4,w21; mov w5,#N_EMBD; mov w6,#N_EMBD
    bl gemm

    ldp d8,d9,[sp,96]; ldp x27,x28,[sp,80]
    ldp x25,x26,[sp,64]; ldp x23,x24,[sp,48]
    ldp x21,x22,[sp,32]; ldp x19,x20,[sp,16]
    ldp x29,x30,[sp],112; ret

/* ================================================================
   mlp_block  –  feed-forward network
   x0=out[seq*768]  x1=hidden[seq*768]  w2=seq_len  w3=layer
   ================================================================ */
.type mlp_block, %function
mlp_block:
    stp x29,x30,[sp,-64]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]; stp x23,x24,[sp,48]

    mov x19,x0; mov x20,x1; uxtw x21,w2; uxtw x22,w3

    adrp x0,g_w_layers_base; add x0,x0,:lo12:g_w_layers_base; ldr x0,[x0]
    ldr x1,=LAYER_BYTES; mul x1,x22,x1; add x23,x0,x1

    /* h1 = gelu(hidden @ c_fc_w + c_fc_b) */
    adrp x0,g_buf_large; add x0,x0,:lo12:g_buf_large; ldr x0,[x0]
    mov x1,x20
    ldr x4,=OFF_CFC_W; add x2,x23,x4
    ldr x4,=OFF_CFC_B; add x3,x23,x4
    mov w4,w21; mov w5,#N_EMBD; mov w6,#N_FFN
    bl gemm

    adrp x0,g_buf_large; add x0,x0,:lo12:g_buf_large; ldr x0,[x0]
    mov w1,w21; mov w2,#N_FFN; mul w1,w1,w2
    bl gelu_inplace

    /* out = h1 @ c_proj2_w + c_proj2_b */
    mov x0,x19
    adrp x1,g_buf_large; add x1,x1,:lo12:g_buf_large; ldr x1,[x1]
    ldr x4,=OFF_CPROJ2_W; add x2,x23,x4
    ldr x4,=OFF_CPROJ2_B; add x3,x23,x4
    mov w4,w21; mov w5,#N_FFN; mov w6,#N_EMBD
    bl gemm

    ldp x23,x24,[sp,48]; ldp x21,x22,[sp,32]
    ldp x19,x20,[sp,16]; ldp x29,x30,[sp],64; ret

/* ================================================================
   transformer_layer  –  one transformer block
   x0=hidden[seq*768]  w1=seq_len  w2=layer
   ================================================================ */
.type transformer_layer, %function
transformer_layer:
    stp x29,x30,[sp,-64]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]; stp x23,x24,[sp,48]

    mov x19,x0; uxtw x20,w1; uxtw x21,w2

    /* layer_base for LN weights */
    adrp x0,g_w_layers_base; add x0,x0,:lo12:g_w_layers_base; ldr x0,[x0]
    ldr x1,=LAYER_BYTES; mul x1,x21,x1; add x23,x0,x1

    /* attn_out = mha_block(hidden) */
    adrp x0,g_buf_tmp; add x0,x0,:lo12:g_buf_tmp; ldr x0,[x0]
    mov x1,x19; mov w2,w20; mov w3,w21
    bl mha_block

    /* hidden += attn_out */
    mov x0,x19
    adrp x1,g_buf_tmp; add x1,x1,:lo12:g_buf_tmp; ldr x1,[x1]
    mov w2,w20; mov w3,#N_EMBD; mul w2,w2,w3
    bl vec_add_inplace

    /* hidden = LN(hidden, ln1_w, ln1_b) */
    mov x0,x19
    ldr x4,=OFF_LN1_W; add x1,x23,x4
    ldr x4,=OFF_LN1_B; add x2,x23,x4
    mov w3,w20; mov w4,#N_EMBD
    bl layer_norm_inplace

    /* mlp_out = mlp_block(hidden) */
    adrp x0,g_buf_tmp; add x0,x0,:lo12:g_buf_tmp; ldr x0,[x0]
    mov x1,x19; mov w2,w20; mov w3,w21
    bl mlp_block

    /* hidden += mlp_out */
    mov x0,x19
    adrp x1,g_buf_tmp; add x1,x1,:lo12:g_buf_tmp; ldr x1,[x1]
    mov w2,w20; mov w3,#N_EMBD; mul w2,w2,w3
    bl vec_add_inplace

    /* hidden = LN(hidden, ln2_w, ln2_b) */
    mov x0,x19
    ldr x4,=OFF_LN2_W; add x1,x23,x4
    ldr x4,=OFF_LN2_B; add x2,x23,x4
    mov w3,w20; mov w4,#N_EMBD
    bl layer_norm_inplace

    ldp x23,x24,[sp,48]; ldp x21,x22,[sp,32]
    ldp x19,x20,[sp,16]; ldp x29,x30,[sp],64; ret

/* ================================================================
   lm_head  –  logits = hidden_last @ tok_embed^T
   x0=logits[VOCAB_SIZE]  x1=hidden_last[N_EMBD]
   ================================================================ */
.type lm_head, %function
lm_head:
    stp x29,x30,[sp,-64]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]; stp x23,x24,[sp,48]

    mov x19,x0; mov x20,x1
    adrp x0,g_w_tok_embed; add x0,x0,:lo12:g_w_tok_embed; ldr x21,[x0]
    mov x22,#(N_EMBD*4)
    mov x23,#0                          // j
    mov w9,#VOCAB_SIZE
.Llm:
    cmp x23,x9; bge .Llmd
    mul x24,x23,x22; add x10,x21,x24   // tok_embed[j]
    mov x11,x20                         // hidden_last
    /* inline dot768 */
    movi v0.4s,#0; movi v1.4s,#0; movi v2.4s,#0; movi v3.4s,#0
    mov x12,#(N_EMBD*4)
.Llm_dot:
    ldp q16,q17,[x10],#32; ldp q18,q19,[x11],#32
    fmla v0.4s,v16.4s,v18.4s; fmla v1.4s,v17.4s,v19.4s
    ldp q16,q17,[x10],#32; ldp q18,q19,[x11],#32
    fmla v2.4s,v16.4s,v18.4s; fmla v3.4s,v17.4s,v19.4s
    subs x12,x12,#64; bgt .Llm_dot
    fadd v0.4s,v0.4s,v1.4s; fadd v2.4s,v2.4s,v3.4s
    fadd v0.4s,v0.4s,v2.4s
    faddp v0.4s,v0.4s,v0.4s; faddp s0,v0.2s
    str s0,[x19,x23,lsl #2]
    add x23,x23,#1; b .Llm
.Llmd:
    ldp x23,x24,[sp,48]; ldp x21,x22,[sp,32]
    ldp x19,x20,[sp,16]; ldp x29,x30,[sp],64; ret

/* ================================================================
   argmax  –  x0=data  w1=len  →  w0=index of max
   ================================================================ */
.type argmax, %function
argmax:
    ldr s0,[x0]; mov w2,#0; mov w3,#1
    uxtw x1,w1
.Lam: cmp x3,x1; bge .Lamd
    ldr s1,[x0,x3,lsl #2]; fcmp s1,s0; ble .Lamn
    fmov s0,s1; mov w2,w3
.Lamn: add x3,x3,#1; b .Lam
.Lamd: mov w0,w2; ret

/* ================================================================
   detok_print  –  print the string for a single token id
   w0=token_id
   ================================================================ */
.type detok_print, %function
detok_print:
    stp x29,x30,[sp,-48]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]
    uxtw x19,w0

    adrp x0,g_offsets_ptr; add x0,x0,:lo12:g_offsets_ptr; ldr x0,[x0]
    ldr w20,[x0,x19,lsl #2]            // offset_start
    add x1,x19,#1; ldr w21,[x0,x1,lsl #2] // offset_end
    sub w22,w21,w20                     // str_len

    adrp x0,g_strings_ptr; add x0,x0,:lo12:g_strings_ptr; ldr x0,[x0]
    add x1,x0,x20                       // str_ptr (x1)

    /* check if first char is punctuation */
    cbz w22, .Lcheck_space
    ldrb w9, [x1]
    cmp w9, #','
    beq .Lclear_space
    cmp w9, #'.'
    beq .Lclear_space
    cmp w9, #';'
    beq .Lclear_space
    cmp w9, #':'
    beq .Lclear_space
    cmp w9, #'!'
    beq .Lclear_space
    cmp w9, #'?'
    beq .Lclear_space
    b .Lcheck_space

.Lclear_space:
    adrp x0, g_pending_space; add x0, x0, :lo12:g_pending_space
    str wzr, [x0]
    b .Lprint_token

.Lcheck_space:
    adrp x0, g_pending_space; add x0, x0, :lo12:g_pending_space
    ldr w8, [x0]
    cbz w8, .Lprint_token
    str wzr, [x0]                    // clear g_pending_space
    str x1, [sp, #-16]!               // save str_ptr
    adrp x1, char_space; add x1, x1, :lo12:char_space
    mov x0, #STDOUT_FD; mov x2, #1; bl write
    ldr x1, [sp], #16                 // restore str_ptr

.Lprint_token:
    /* check suffix */
    adrp x0,g_suffix_len; add x0,x0,:lo12:g_suffix_len; ldr w3,[x0]
    cbz w3,.Ldt_raw
    cmp w22,w3; blt .Ldt_raw

    /* compare tail with suffix */
    sub w4,w22,w3; add x5,x1,x4        // tail ptr
    adrp x0,g_suffix_ptr; add x0,x0,:lo12:g_suffix_ptr; ldr x6,[x0]
    mov w7,#0
.Ldt_cmp:
    cmp w7,w3; bge .Ldt_suf
    ldrb w8,[x5,x7]; ldrb w9,[x6,x7]
    cmp w8,w9; bne .Ldt_raw
    add w7,w7,#1; b .Ldt_cmp

.Ldt_suf:
    /* print without suffix then set pending space */
    mov x0,#STDOUT_FD; /* x1 already str_ptr */ uxtw x2,w4
    bl write
    adrp x0,g_pending_space; add x0,x0,:lo12:g_pending_space
    mov w3,#1; str w3,[x0]
    b .Ldt_done

.Ldt_raw:
    mov x0,#STDOUT_FD; /* x1=str_ptr */ uxtw x2,w22
    bl write
    adrp x0,g_pending_space; add x0,x0,:lo12:g_pending_space
    str wzr,[x0]

.Ldt_done:
    ldp x21,x22,[sp,32]; ldp x19,x20,[sp,16]
    ldp x29,x30,[sp],48; ret

/* ================================================================
   alloc_buf  –  helper: malloc and store pointer
   x0=bss_addr  x1=size  →  stores result at [x0]
   ================================================================ */
.type alloc_buf, %function
alloc_buf:
    stp x29,x30,[sp,-32]!; mov x29,sp; str x19,[sp,16]
    mov x19,x0; mov x0,x1; bl malloc
    cbz x0,.Loom; str x0,[x19]
    ldr x19,[sp,16]; ldp x29,x30,[sp],32; ret
.Loom:
    adrp x0,msg_oom; add x0,x0,:lo12:msg_oom
    bl puts; mov w0,#1; bl exit

/* ================================================================
   MAIN
   ================================================================ */
.type main, %function
main:
    stp x29,x30,[sp,-80]!; mov x29,sp
    stp x19,x20,[sp,16]; stp x21,x22,[sp,32]
    stp x23,x24,[sp,48]; stp x25,x26,[sp,64]

    /* ── load packed files ── */
    adrp x0,path_tokenizer; add x0,x0,:lo12:path_tokenizer
    bl load_file; mov x19,x0

    mov x0,x19; ldr w1,=MAGIC_TKR1
    adrp x3,msg_bad_tokenizer; add x3,x3,:lo12:msg_bad_tokenizer; bl check_magic

    adrp x0,path_merges; add x0,x0,:lo12:path_merges
    bl load_file; mov x20,x0

    mov x0,x20; ldr w1,=MAGIC_MRG1
    adrp x3,msg_bad_merges; add x3,x3,:lo12:msg_bad_merges; bl check_magic

    adrp x0,path_weights; add x0,x0,:lo12:path_weights
    bl load_file; mov x21,x0

    mov x0,x21; ldr w1,=MAGIC_WGT1
    adrp x3,msg_bad_weights; add x3,x3,:lo12:msg_bad_weights; bl check_magic

    /* ── setup ── */
    mov x0,x19; mov x1,x20; bl tokenizer_setup
    mov x0,x21; bl weights_setup
    
    adrp x0,msg_loaded; add x0,x0,:lo12:msg_loaded; bl puts

    /* ── allocate scratch buffers ── */
    adrp x0,g_buf_hidden;  add x0,x0,:lo12:g_buf_hidden
    mov x1,#(N_CTX*N_EMBD*4); bl alloc_buf

    adrp x0,g_buf_tmp;     add x0,x0,:lo12:g_buf_tmp
    mov x1,#(N_CTX*N_EMBD*4); bl alloc_buf

    adrp x0,g_buf_tmp2;    add x0,x0,:lo12:g_buf_tmp2
    mov x1,#(N_CTX*N_EMBD*4); bl alloc_buf

    adrp x0,g_buf_large;   add x0,x0,:lo12:g_buf_large
    ldr x1,=(N_CTX*N_FFN*4); bl alloc_buf

    adrp x0,g_buf_scores;  add x0,x0,:lo12:g_buf_scores
    ldr x1,=(N_CTX*N_CTX*4); bl alloc_buf

    adrp x0,g_buf_logits;  add x0,x0,:lo12:g_buf_logits
    ldr x1,=(VOCAB_SIZE*4); bl alloc_buf

    /* ── tokenize prompt ── */
    adrp x0,prompt; add x0,x0,:lo12:prompt
    adrp x1,g_token_ids; add x1,x1,:lo12:g_token_ids
    mov w2,#N_CTX
    bl tokenize_prompt
    adrp x1,g_num_tokens; add x1,x1,:lo12:g_num_tokens; str w0,[x1]
    mov w22,w0                          // seq_len


    /* print prompt tokens (decoded) */
    adrp x23,g_token_ids; add x23,x23,:lo12:g_token_ids
    mov w9,#0
.Lmain_pp:
    cmp w9,w22; bge .Lmain_gen
    ldr w0,[x23,x9,lsl #2]; str x9,[sp,#-16]!; bl detok_print; ldr x9,[sp],#16
    add w9,w9,#1; b .Lmain_pp

    /* ── generation loop ── */
.Lmain_gen:
    mov w24,#0                          // gen step
.Lmain_step:
    cmp w24,#GEN_TOKENS; bge .Lmain_end

    /* embedding */
    adrp x0,g_buf_hidden; add x0,x0,:lo12:g_buf_hidden; ldr x0,[x0]
    mov x1,x23; mov w2,w22
    bl embedding


    /* transformer layers */
    mov w25,#0
.Lmain_layer:
    cmp w25,#N_LAYERS; bge .Lmain_lm
    adrp x0,g_buf_hidden; add x0,x0,:lo12:g_buf_hidden; ldr x0,[x0]
    mov w1,w22; mov w2,w25
    bl transformer_layer

    add w25,w25,#1; b .Lmain_layer

.Lmain_lm:
    /* lm_head on last position */
    adrp x0,g_buf_logits; add x0,x0,:lo12:g_buf_logits; ldr x0,[x0]
    adrp x1,g_buf_hidden; add x1,x1,:lo12:g_buf_hidden; ldr x1,[x1]
    sub x2,x22,#1; mov x3,#(N_EMBD*4); mul x2,x2,x3; add x1,x1,x2
    bl lm_head

    /* argmax */
    adrp x0,g_buf_logits; add x0,x0,:lo12:g_buf_logits; ldr x0,[x0]
    mov w1,#VOCAB_SIZE; bl argmax
    mov w26,w0                          // next_id

    /* append to token_ids */
    str w26,[x23,x22,lsl #2]
    add w22,w22,#1

    /* print token */
    mov w0,w26; bl detok_print

    add w24,w24,#1; b .Lmain_step

.Lmain_end:
    /* newline */
    adrp x1,char_newline; add x1,x1,:lo12:char_newline
    mov x0,#STDOUT_FD; mov x2,#1; bl write

    mov w0,#0
    ldp x25,x26,[sp,64]; ldp x23,x24,[sp,48]
    ldp x21,x22,[sp,32]; ldp x19,x20,[sp,16]
    ldp x29,x30,[sp],80; ret

/* ────────────────────── RODATA ────────────────────── */
.section .rodata
.align 3

path_tokenizer: .asciz "../packed/tokenizer.bin"
path_merges:    .asciz "../packed/merges.bin"
path_weights:   .asciz "../packed/weights.bin"

prompt:         .asciz "The sky is"

msg_loaded:       .asciz "Packed files loaded"
msg_load_fail:    .asciz "Failed to load packed file"
msg_bad_tokenizer:.asciz "Invalid tokenizer.bin magic"
msg_bad_merges:   .asciz "Invalid merges.bin magic"
msg_bad_weights:  .asciz "Invalid weights.bin magic"
msg_oom:          .asciz "Out of memory"
msg_newline:      .asciz ""

fmt_token:        .asciz "%d "
fmt_debug_tok:    .asciz "Vocab size: %d, unk_id: %d, lowercase: %d, suffix_len: %d\n"
fmt_lookup:       .asciz "lookup: %.*s\n"
fmt_debug_probe:  .asciz "probe slot %ld: id=%d, hash=%016llx\n"
fmt_debug_offsets:.asciz "Pointers: tok_base=%p, suffix=%p, offsets=%p, strings=%p, hash=%p, ids=%p, hash_len=%d\n"
fmt_debug_hash:   .asciz "hash(%.*s) = %016llx\n"
fmt_debug_append: .asciz "append piece ID: %d\n"
fmt_debug_emb:    .asciz "Emb: %.6f, %.6f, %.6f, %.6f\n"
fmt_debug_l0:     .asciz "L0: %.6f, %.6f, %.6f, %.6f\n"
fmt_debug_attn:   .asciz "Attn: %.6f, %.6f, %.6f, %.6f\n"
fmt_debug_ln1:    .asciz "LN1: %.6f, %.6f, %.6f, %.6f\n"
fmt_debug_ln_stats:.asciz "LN stats: mean=%.6f, std=%.6f\n"
fmt_debug_x_attn: .asciz "X+Attn: %.6f, %.6f, %.6f, %.6f\n"

punct_str:        .asciz "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

.align 2
const_eps:        .float 1e-5
const_neg_inf:    .float -1e9
const_sqrt2pi:    .float 0.7978845608
const_gelu_c:     .float 0.044715
char_space:       .ascii " "
char_newline:     .ascii "\n"
