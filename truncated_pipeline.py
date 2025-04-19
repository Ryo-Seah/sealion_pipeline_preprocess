from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import os, re, json, hashlib, unicodedata
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from simhash import Simhash
from tqdm import tqdm
import torch
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
from config import *
import fasttext
lang_model = fasttext.load_model("models/lid.176.bin") 




def load_shard(shard_idx):
    ds = load_dataset("biglam/hmd_newspapers", split="train")
    ds = ds.shuffle(seed=42).select(range(30000))

    return ds.shard(num_shards=SHARDS, index=shard_idx, contiguous=True)


def safe_detect_lang(text):
    prediction = lang_model.predict(text.replace("\n", " "))[0][0]
    return prediction.replace("__label__", "")

def clean_text(text):
    #feplace newline character n surrounding white space with space
    text = re.sub(r'\s*\n\s*', ' ', text)
    #fix oddly hyphenated words
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    #if alphabets repeated >3 times, truncate to 1
    text = re.sub(r'([a-zA-Z])\1{3,}', r'\1', text)
    # Remove any character repeated >=4 times that is NOT a letter
    text = re.sub(r'([^a-zA-Z\s])\1{3,}', '', text)
    text = unicodedata.normalize("NFKC", text)
    
    # Keep normal sentence punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.\']", ' ', text)
    #normalize white space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def cached_tokenize(text, tokenizer, cache_dir="tokenizer_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{text_hash}.json")
    
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)["input_ids"]

    token_ids = tokenizer(text).input_ids
    with open(cache_path, "w") as f:
        json.dump({"input_ids": token_ids}, f)
    return token_ids

#Filter min words, nonenglish(garbled), ocr quality threshold
def filter_and_clean(example):
    text = example['text']
    if not text or len(text.split()) < MIN_WORDS:
        return None
    if float(example.get('ocr_quality_mean', 1)) < OCRT_MEAN_THRESHOLD:
        return None
    lang = safe_detect_lang(text)
    if lang not in ['en', 'undetermined', 'error']:
        return None
    #cleaning must come last
    cleaned = clean_text(text)
    return {
        'id': example['id'] if 'id' in example else hashlib.md5(text.encode()).hexdigest(),
        'text': cleaned,
        'ocr_quality_mean': example.get('ocr_quality_mean'),
        'ocr_quality_sd': example.get('ocr_quality_sd'),
        'word_count': example.get('word_count')
    }

# def deduplicate(examples):
#     if isinstance(examples, list):
#         examples = Dataset.from_list(examples)

#     df = examples.to_pandas()

#     # Clean and normalize text for deduplication
#     df['text_clean'] = df['text'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.strip()
#     df['text_hash'] = df['text_clean'].apply(hash)

#     df_dedup = df.drop_duplicates(subset='text_hash').drop(columns=['text_clean', 'text_hash'])

#     return Dataset.from_pandas(df_dedup.reset_index(drop=True))
def deduplicate(infile_path, outfile_path):
    seen_hashes = set()
    with open(outfile_path, 'w') as out:
        for chunk in pd.read_json(infile_path, lines=True, chunksize=5000):
            chunk['text_clean'] = chunk['text'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.strip()
            chunk['text_hash'] = chunk['text_clean'].apply(hash)
            chunk = chunk.drop_duplicates(subset='text_hash')
            chunk.drop(columns=['text_clean', 'text_hash'], inplace=True)
            for row in chunk.to_dict(orient='records'):
                if row['id'] not in seen_hashes:
                    seen_hashes.add(row['id'])
                    out.write(json.dumps(row) + '\n')

# def deduplicate(dataset):
#     seen_hashes = set()
#     simhashes = []
#     unique = []
#     for ex in dataset:
#         h = hashlib.md5(ex['text'].encode()).hexdigest()
#         if h in seen_hashes:
#             continue
#         sh = Simhash(ex['text'])
#         is_duplicate = any(sh.distance(other) < 5 for other in simhashes)
#         if not is_duplicate:
#             simhashes.append(sh)
#             seen_hashes.add(h)
#             unique.append(ex)
#     return unique

def tokenize_and_chunk(tokenizer, text, max_tokens=MAX_TOKENS):
    #use paragraph as chunk boundaries
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    buffer = ""
    for para in paragraphs:
        temp = f"{buffer} {para}".strip()
        # if buffer + para <= than max token, , add to bufer
        if len(tokenizer(temp).input_ids) <= max_tokens:
            buffer = temp
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para
    #flush buffer
    if buffer:
        chunks.append(buffer)
    return chunks
#explore packing in groups instead of chunks
def pack_sequences(tokenizer, chunks, max_tokens=MAX_TOKENS):
    packed = []
    buffer = []
    length = 0
    for chunk in chunks:
        tokens = tokenizer(chunk).input_ids
        
        #if more than max token + 1 f for seprator exceed max token, we do not append to buffer and flush it 
        if length + len(tokens) + 1 > max_tokens:
            if buffer:
                # when flushing, join chunks in buffer using seperator (EOT)
                combined = SEPARATOR_TOKEN.join(buffer)
                # token_ids = tokenizer(combined).input_ids
                token_ids = cached_tokenize(combined, tokenizer)
                token_ids.append(tokenizer.eos_token_id)  # append EOS token explicitly
                packed.append(token_ids)
                
            buffer, length = [], 0
        buffer.append(chunk)
        length += len(tokens)
    if buffer:
        combined = SEPARATOR_TOKEN.join(buffer)
        # token_ids = tokenizer(combined).input_ids
        token_ids = cached_tokenize(combined, tokenizer)
        token_ids.append(tokenizer.eos_token_id)  # append EOS token
        packed.append(token_ids)
    return packed

def process_batch(batch, tokenizer):
    outputs = []
    for ex in batch:
        chunks = tokenize_and_chunk(tokenizer, ex['text'])
        sequences = pack_sequences(tokenizer, chunks)
        for seq in sequences:
            # might be unecessary
            if len(seq) >= MIN_TOKENS:
                outputs.append({
                    'article_id': ex['id'],
                    'tokens': seq,
                    'token_count': len(seq),
                    'ocr_quality_mean': ex['ocr_quality_mean'],
                    'ocr_quality_sd': ex['ocr_quality_sd'],
                })
    return outputs

def save_as_pt(file_path, data):
    token_tensors = [torch.tensor(d['tokens'], dtype=torch.long) for d in data]
    torch.save(token_tensors, file_path.replace('.jsonl', '.pt'))
    
def batch_to_rows(batch):
    if isinstance(batch, dict):
        return [dict(row) for row in zip(*batch.values())]
    else:
        return [dict(row) for row in zip(*batch.to_dict().values())]
    
def main():
    shard_idx = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    ds = load_shard(shard_idx)
    # dataset = load_dataset("biglam/hmd_newspapers", split="train")
    # ds = dataset.shuffle(seed=42).select(range(30000))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    os.makedirs("output", exist_ok=True)
    filtered_path = f"output/filtered_shard_{shard_idx}.jsonl"
    deduped_path = f"output/deduped_shard_{shard_idx}.jsonl"

    print("EOS token:", tokenizer.eos_token) #make sure is the same as my config


    os.makedirs("output", exist_ok=True)

    # Filter in batches
    with open(filtered_path, 'w') as fout:
        for i in tqdm(range(0, len(ds), BATCH_SIZE)):
            batch = ds.select(range(i, min(i + BATCH_SIZE, len(ds))))
            for example in batch:
                cleaned = filter_and_clean(example)
                if cleaned is not None:
                    fout.write(json.dumps(cleaned) + "\n")

    # dedup across all
    deduped = deduplicate(filtered_path, deduped_path)

    # Tokenize deduplicated data in batches
    all_outputs = []
    with open(deduped_path, 'r') as f:
        for chunk in pd.read_json(f, lines=True, chunksize=BATCH_SIZE):
            dataset = Dataset.from_pandas(chunk)
            batch = [row for row in dataset]
            tokenized = process_batch(batch, tokenizer)
            all_outputs.extend(tokenized)
    
    with open(f"output/processed_shard_{shard_idx}.jsonl", 'w') as f:
        for item in all_outputs:
            f.write(json.dumps(item) + '\n')


    save_as_pt("output/processed_baseline.jsonl", all_outputs)

if __name__ == "__main__":
    main()