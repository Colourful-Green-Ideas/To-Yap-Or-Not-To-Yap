from datasets import load_dataset
from kneyser_ney_measure import NLTKLexicalSurprisalScorer


def fetch_wikitext103_lines(
    max_lines: int = 75_000, exclude_texts: set[str] | None = None
) -> list[str]:
    """Streams clean, non-header paragraphs from WikiText-103."""
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    exclude_texts = exclude_texts or set()

    clean_lines: list[str] = []
    for row in ds:
        line = row["text"].strip()
        # Skip blank lines, WikiText section headings (= = Title = =), and any overlapping eval passages
        if not line or line.startswith("=") or len(line) < 40:
            continue
        if line in exclude_texts:
            continue

        clean_lines.append(line)
        if len(clean_lines) >= max_lines:
            break

    return clean_lines


if __name__ == "__main__":
    TOKENIZER_NAME = "bigscience/bloom-7b1"
    OUTPUT_PATH = "nltk_3gram_en.pkl"

    print("Streaming WikiText-103 lines...")
    train_texts = fetch_wikitext103_lines(max_lines=75_000)
    print(f"Collected {len(train_texts)} clean passages.")

    print("Training 3-gram Kneser-Ney model...")
    scorer = NLTKLexicalSurprisalScorer(
        tokenizer_name=TOKENIZER_NAME, order=3, discount=0.75
    )
    scorer.train(train_texts)

    scorer.save(OUTPUT_PATH)
    print(f"Saved trained model to {OUTPUT_PATH}")

    # Quick sanity check
    tokens, surprisals = scorer.score_text(
        "The scientific method relies on empirical observation and hypothesis testing."
    )
    print("Sample tokens:", tokens[:8])
    print("Sample surprisals (bits):", [round(s, 2) for s in surprisals[:8]])
