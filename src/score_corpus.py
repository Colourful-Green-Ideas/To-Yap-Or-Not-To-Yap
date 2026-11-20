# Scores the human continuations in corpus_prompts.tsv with three judges:
#   BLOOM-3b and mGPT (neural, external judges) and our Kneser-Ney trigram (lexical).
# Writes one JSON object per line to scores_human.jsonl and can be stopped and resumed.
#
# Documents are scored round-robin across the five subsets, so stopping early
# still gives a balanced sample.
#
# Run from src/:  uv run score_corpus.py
#                 uv run score_corpus.py --per-subset 100

import argparse
import csv
import json
import os
import time
from collections import defaultdict

from kn import KNSurprisalScorer
from neural import NeuralSurprisalScorer

INPUT_TSV = "corpus_prompts.tsv"
OUTPUT_JSONL = "scores_human.jsonl"
TOKENIZER_NAME = "bigscience/bloom-3b"


def round_robin(rows):
    """Reorder rows so the subsets take turns: 1st of each subset, then 2nd of each, ..."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r[1], r[2])].append(r)
    ordered = []
    for i in range(max(len(g) for g in groups.values())):
        for key in sorted(groups):
            if i < len(groups[key]):
                ordered.append(groups[key][i])
    return ordered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-subset", type=int, default=None,
                        help="only score the first N documents of each subset")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    bloom_scorer = NeuralSurprisalScorer("bigscience/bloom-3b", device=args.device)
    mgpt_scorer = NeuralSurprisalScorer("ai-forever/mGPT", device=args.device)

    print("Loading Kneser-Ney models...")
    kn_scorers = {
        "en": KNSurprisalScorer("en_kn3.pkl", tokenizer_name=TOKENIZER_NAME),
        "hi": KNSurprisalScorer("hi_kn3.pkl", tokenizer_name=TOKENIZER_NAME),
    }

    done = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["doc_id"])
        print(f"Already scored: {len(done)} documents, skipping them")

    with open(INPUT_TSV, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]

    if args.per_subset is not None:
        counts = defaultdict(int)
        kept = []
        for r in rows:
            if counts[(r[1], r[2])] < args.per_subset:
                kept.append(r)
                counts[(r[1], r[2])] += 1
        rows = kept

    todo = [r for r in round_robin(rows) if r[0] not in done]
    print(f"To score: {len(todo)} documents")

    start = time.time()
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out:
        for i, (doc_id, lang, genre, author, prompt, cont) in enumerate(todo, start=1):
            b_tokens, b_bits = bloom_scorer.score_continuation(prompt, cont)
            m_tokens, m_bits = mgpt_scorer.score_continuation(prompt, cont)
            k_tokens, k_bits = kn_scorers[lang].score_continuation(prompt, cont)

            entry = {
                "doc_id": doc_id,
                "lang": lang,
                "genre": genre,
                "author": author,
                "bloom_pairs": [[t, round(b, 4)] for t, b in zip(b_tokens, b_bits)],
                "mgpt_pairs": [[t, round(b, 4)] for t, b in zip(m_tokens, m_bits)],
                "kn_pairs": [[t, round(b, 4)] for t, b in zip(k_tokens, k_bits)],
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out.flush()

            per_doc = (time.time() - start) / i
            mins_left = per_doc * (len(todo) - i) / 60
            print(f"{len(done) + i}/{len(done) + len(todo)} ({doc_id})  |  "
                  f"{per_doc:.2f} s/doc  |  ~{mins_left:.0f} min left")

    print("Finished. Results in", OUTPUT_JSONL)


if __name__ == "__main__":
    main()
