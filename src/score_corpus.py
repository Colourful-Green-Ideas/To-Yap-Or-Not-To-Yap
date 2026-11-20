import csv
import json
import os
import time
from kn import KNSurprisalScorer
from neural import NeuralSurprisalScorer

INPUT_TSV = "corpus_prompts.tsv"
OUTPUT_JSONL = "output.jsonl"
TOKENIZER_NAME = "bigscience/bloom-3b"


def main():
    # External bilingual judges (shared across English & Hindi)
    bloom_scorer = NeuralSurprisalScorer("bigscience/bloom-3b", device="mps")
    mgpt_scorer = NeuralSurprisalScorer("ai-forever/mGPT", device="mps")

    print("Loading Kneser-Ney models...")
    kn_scorers = {
        "en": KNSurprisalScorer("en_kn3.pkl", tokenizer_name=TOKENIZER_NAME),
        "hi": KNSurprisalScorer("hi_kn3.pkl", tokenizer_name=TOKENIZER_NAME),
    }

    # Resume by (doc_id, author) pair so multiple authors per doc_id don't collide
    done: set[tuple[str, str]] = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add((rec["doc_id"], rec["author"]))
        print(f"Already scored: {len(done)} entries, skipping them")

    with open(INPUT_TSV, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]

    todo = [r for r in rows if (r[0], r[3]) not in done]
    print(f"To score: {len(todo)} entries")

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

            #if i == 1 or i % 25 == 0 or i == len(todo):
            per_doc = (time.time() - start) / i
            mins_left = per_doc * (len(todo) - i) / 60
            total_done = len(done) + i
            print(
                f"{total_done}/{len(rows)} done ({doc_id} | {author})  |  "
                f"{per_doc:.2f} s/entry  |  ~{mins_left:.1f} min left"
            )

    print("Finished. Results in", OUTPUT_JSONL)


if __name__ == "__main__":
    main()

