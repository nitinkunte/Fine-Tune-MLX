# 🏥 Fine-Tune-MLX  
**One-stop repo to fine-tune small LLMs on your Mac with Apple’s MLX framework**

> Zero cloud cost, full HIPAA control, and sub-second inference on your M-series Mac.

---

## 📦 What’s inside

| File / Folder | Purpose |
|---------------|---------|
| `README.md` | You’re reading it |
| `scripts/` | Ready-to-run shell snippets (see below) |
| `requirements.txt` | `mlx-lm` + optional GGUF converter |
| `data/` | Drop your JSONL training files here |
| `fused_model/` | Output directory after `mlx_lm.fuse` |
| `gguf/` | Optional GGUF binaries for llama.cpp |

---

## 🚀 Quick start (5 min)

### 1. Install MLX-LM
```bash
python -m pip install --upgrade pip
pip install mlx-lm
```
MLX-LM is the official Apple package that exposes `mlx_lm.lora`, `mlx_lm.fuse`, and `mlx_lm.generate`.

### 2. Prepare training data
Create a **JSONL** file (`data/train.jsonl`) with one JSON object per line:

```json
{"prompt": "Summarize the patient’s chief complaint.", "completion": "Chest pain for 2 hours."}
```

### 3. Fine-tune (LoRA)
```bash
mlx_lm.lora \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train \
  --data ./data \
  --adapter-path ./adapters.npz \
  --max-seq-length 8192 \
  --batch-size 1 \
  --iters 1200 \
  --learning-rate 3e-4 \
  --save-every 100
```
What it does  
- Loads the 1.5 B model into unified memory  
- Adds low-rank adapters (LoRA)  
- Trains only the adapters (fast & memory-friendly)  
- Saves checkpoints every 100 steps to `./adapters.npz`

### 4. Test the adapter
```bash
mlx_lm.generate \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path ./adapters.npz \
  --prompt "How many cases of viral infection?" \
  --max-tokens 64
```
You should see answers that reflect your training data style.

### 5. Merge adapter into base model
```bash
mlx_lm.fuse \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path ./adapters \
  --save-path ./fused_model
```
This produces a **stand-alone** model directory (`fused_model/`) that no longer needs the adapter file.

### 6. (Optional) Convert to GGUF for llama.cpp
```bash
pip install llama-cpp-python  # or grab convert_hf_to_gguf.py from llama.cpp repo
python convert_hf_to_gguf.py ./fused_model \
       --outfile ./gguf/Qwen2.5-1.5B-Instruct_FineTuned.gguf \
       --outtype f16
```
Now you can run the model with `llama-cli`, Ollama, or any GGUF-compatible stack.

---

## 🧪 Example scripts

```bash
chmod +x scripts/*.sh
./scripts/01_train.sh
./scripts/02_test.sh
./scripts/03_fuse.sh
```

---

## ⚠️ Caveats

1. **Data quality > model size**. Budget 80 % of your time cleaning and labeling JSONL.  
2. Use `--grad-checkpoint` and `--batch-size 1` on 8 GB+ RAM Macs; increase only if you have headroom.  
3. Always de-identify PHI before training.

---

## 🤝 Contributing

PRs welcome—especially data-prep utilities and new model configs!

---

**License**: MIT  
**Author**: Nitin Kunte
