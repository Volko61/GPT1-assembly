Build steps

1) Pack the model assets:

python3 tools/pack_model.py --model-dir model --out-dir packed

2) Build the assembly runtime:

cd asm
make

3) Run:

./infer

Notes

- This is a scaffold that only loads and validates the packed files.
- Next steps are tokenizer, transformer math kernels, and sampling in pure ARM64 assembly.
