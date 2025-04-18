# 🦭 SEA-LION Preprocessing Pipeline

This repository contains a scalable and efficient pipeline for preprocessing historical text data (e.g., OCR newspaper scans) for **causal language modeling** using the [`aisingapore/Llama-SEA-LION-v2-8B`](https://huggingface.co/aisingapore/Llama-SEA-LION-v2-8B) model or other LLaMA-style architectures.

---

## Installation

1. **Create and activate a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

2. **Install dependencies**

    ```bash
    pip install -r requirements.txt

---

## ⚙️ Preprocessing Options

### Run Locally

```bash
python pipeline.py
```

###  Run using SLURM

1. Make the SLURM script executable:

   ```bash
   chmod +x slurm_preprocess.sh
   ```

2. Submit it with:

   ```bash
   sbatch slurm_preprocess.sh
   ```

Make sure `pipeline.py` uses the environment variable `SLURM_ARRAY_TASK_ID` to determine the shard index for parallel processing.

---

## Output Files

All outputs will be written to the `output/` directory:

- `processed_baseline.jsonl`: Tokenized sequences with metadata
- `processed_baseline.pt`: Packed PyTorch tensors (for efficient training)
- `processed_texts.jsonl`: Cleaned and filtered text before tokenization

---

## Preprocessing Logic

Each article goes through the following steps:

1. **Initial Filtering**
   - Drop entries with fewer than 5 words or tokens
   - Remove entries with `ocr_quality_mean` < 0.5
   - Detect and remove non-English entries using `langdetect`
   - Flag (but retain) entries with `ocr_quality_sd` > 0.35 for optional inspection

2. **Text Cleaning**
   - Normalize Unicode characters (NFKC)
   - Fix hyphenated line breaks (e.g., `to-\\nday` → `today`)
   - Remove excessive character repetition (e.g., "loooool" → "lol")
   - Strip out chunks of junk unicode
   - Normalize spacing and remove junk punctuation (except `.` and `'`)

3. **Deduplication**
   - Clean + hash texts to remove exact and near-duplicates

4. **Tokenization & Chunking**
   - Split long texts by paragraphs to fit within model context window (e.g., 2048 tokens)
   - Shorter sequences are packed together using a separator (`<|end_of_text|>`)

5. **EOS Token**
   - Append the model's EOS token at the end of each sequence to signal completion

---

## 🛠 Configuration

Customize the pipeline via `config.py`:

```python
MODEL_NAME = "aisingapore/Llama-SEA-LION-v2-8B"
MAX_TOKENS = 
MIN_TOKENS = 32
MIN_WORDS = 5
OCRT_MEAN_THRESHOLD = 0.5
BATCH_SIZE = 128
SEPARATOR_TOKEN = "<|end_of_text|>"
```

---
