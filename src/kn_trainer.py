# Builds en_kn3.pkl and hi_kn3.pkl for kn_measure.py
# Run:  uv run kn_trainer.py

import os
from datasets import load_dataset
from transformers import AutoTokenizer
from kn_measure import KneserNeyTrigram, KNSurprisalScorer

TOKENIZER_NAME = "bigscience/bloom-7b1"
MAX_LINES = 75_000      # lower this if your Mac runs out of memory


# English background text: WikiText-103 paragraphs
def english_lines():
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    for row in ds:
        line = row["text"].strip()
        if line and not line.startswith("=") and len(line) >= 40:
            yield line


# Hindi background text: IndicCorpV2.
# Not shuffled: its files are split mid-character, so it must be read from the start.
# Instead we keep 1 line in every 5, so the sample covers more of the corpus.
def hindi_lines():
    ds = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="hin_Deva", streaming=True)
    for i, row in enumerate(ds):
        if i % 5 != 0:
            continue
        line = row["text"].strip()
        if len(line) >= 40:
            yield line


def collect(lang, lines):
    # Checkpoint 1: the downloaded text is saved, so a rerun doesn't download it again
    text_file = f"bg_{lang}.txt"

    if os.path.exists(text_file):
        with open(text_file, encoding="utf-8") as f:
            texts = f.read().split("\n")
        print(f"[{lang}] text already downloaded, loaded {len(texts):,} lines from {text_file}")
        return texts

    print(f"[{lang}] downloading text...")
    texts = []
    try:
        for line in lines:
            texts.append(line.replace("\n", " "))
            if len(texts) % 5_000 == 0:
                print(f"    [{lang}] {len(texts):,} / {MAX_LINES:,} lines downloaded")
            if len(texts) >= MAX_LINES:
                break
    except Exception as e:
        # If the download breaks partway, keep what we already have
        if len(texts) < 10_000:
            raise
        print(f"    [{lang}] download stopped early ({type(e).__name__}), using the {len(texts):,} lines we have")

    with open(text_file, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    print(f"[{lang}] saved text to {text_file}")
    return texts


def build(lang, lines, out_path, tokenizer):
    # Checkpoint 2: skip a language whose model is already built
    if os.path.exists(out_path):
        print(f"\n[{lang}] {out_path} already exists, skipping")
        return

    print(f"\n[{lang}] step 1/3: text")
    texts = collect(lang, lines)

    print(f"[{lang}] step 2/3: tokenizing...")
    sequences = []
    for i in range(0, len(texts), 1000):
        sequences += tokenizer(texts[i:i + 1000], add_special_tokens=False)["input_ids"]
        if (i + 1000) % 10_000 == 0:
            print(f"    [{lang}] {i + 1000:,} / {len(texts):,} lines tokenized")
    print(f"[{lang}] {sum(len(s) for s in sequences):,} tokens")

    print(f"[{lang}] step 3/3: counting n-grams (about a minute)...")

    model = KneserNeyTrigram()
    model.train(sequences)
    print(f"[{lang}] {len(model.c3):,} different trigrams")

    model.save(out_path)
    print(f"[{lang}] DONE — saved {out_path}")


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

build("en", english_lines(), "en_kn3.pkl", tokenizer)
build("hi", hindi_lines(), "hi_kn3.pkl", tokenizer)

# Quick sanity check: common text should get lower surprisal than odd text
scorer = KNSurprisalScorer("en_kn3.pkl", TOKENIZER_NAME)
for cont in ["is one of the largest cities in the world.", "purple elephant quantum sandwich."]:
    tokens, bits = scorer.score_continuation("London", cont)
    print(f"{sum(bits) / len(bits):5.2f} bits/token  <-  London {cont}")