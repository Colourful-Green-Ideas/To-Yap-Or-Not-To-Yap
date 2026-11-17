from kn_measure import KNSurprisalScorer
from neural_measure import NeuralSurprisalScorer
import csv
import json
import os
import time

OUTPUT = "output.jsonl"  # one JSON object per line


def main():
    neural_scorer = NeuralSurprisalScorer("unsloth/Llama-3.2-1B")

    print("Loading Kneser-Ney models...")
    kn_scorers = {
        "en": KNSurprisalScorer("en_kn3.pkl"),
        "hi": KNSurprisalScorer("hi_kn3.pkl"),
    }

    # Resume: skip documents already scored on an earlier run
    done = set()
    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["doc_id"])
        print("Already scored:", len(done), "documents, skipping them")

    with open("corpus_prompts.tsv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]

    todo = [r for r in rows if r[0] not in done]
    print("To score:", len(todo), "documents")

    out = open(OUTPUT, "a", encoding="utf-8")
    start = time.time()

    for i, (doc_id, lang, genre, author, prompt, human_cont) in enumerate(
        todo, start=1
    ):
        n_tokens, n_bits = neural_scorer.score_continuation(prompt, human_cont)
        k_tokens, k_bits = kn_scorers[lang].score_continuation(prompt, human_cont)

        entry = {
            "doc_id": doc_id,
            "lang": lang,
            "genre": genre,
            "author": author,
            "neural_pairs": [[t, round(b, 4)] for t, b in zip(n_tokens, n_bits)],
            "kn_pairs": [[t, round(b, 4)] for t, b in zip(k_tokens, k_bits)],
        }

        out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        out.flush()  # saved immediately, so a crash loses nothing

        # Progress after the 1st document, then every 25
        if i == 1 or i % 25 == 0 or i == len(todo):
            per_doc = (time.time() - start) / i
            mins_left = per_doc * (len(todo) - i) / 60
            total_done = len(done) + i
            print(
                f"{total_done}/{len(rows)} done ({doc_id})  |  {per_doc:.2f} s/doc  |  ~{mins_left:.0f} min left"
            )

    out.close()
    print("Finished. Results in", OUTPUT)


if __name__ == "__main__":
    main()

