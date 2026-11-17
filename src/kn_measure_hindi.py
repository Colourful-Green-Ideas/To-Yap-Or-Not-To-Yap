import math
import pickle
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from transformers import AutoTokenizer
import sys


class NLTKLexicalSurprisalScorer:
    def __init__(self, tokenizer_name: str, order: int = 3, discount: float = 0.75):
        self.order = order
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.model = KneserNeyInterpolated(order=order, discount=discount)

    def train(self, texts: list[str]) -> None:
        tokenized_sents = [self.tokenizer.tokenize(t) for t in texts if t.strip()]
        train_data, vocab = padded_everygram_pipeline(self.order, tokenized_sents)
        self.model.fit(train_data, vocab)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def score_text(self, text: str) -> tuple[list[str], list[float]]:
        tokens = self.tokenizer.tokenize(text)
        padded = ["<s>"] * (self.order - 1) + tokens

        surprisals = []
        for i, tok in enumerate(tokens):
            context = padded[i : i + self.order - 1]
            log2_prob = self.model.logscore(tok, context)
            bits = -log2_prob if math.isfinite(log2_prob) else 20.0
            surprisals.append(bits)

        return tokens, surprisals


def main():
    if len(sys.argv) > 1:
        background_texts = ""
        with open("big.txt", "r") as file:
            background_texts = file.read()

            scorer = NLTKLexicalSurprisalScorer("bigscience/bloom-7b1", order=3)
            scorer.train(background_texts)
            scorer.save("nltk_3gram_hi.pkl")

    else:
        scorer = NLTKLexicalSurprisalScorer("bigscience/bloom-7b1", order=3)
        scorer.load("nltk_3gram_hi.pkl")

        token_pairs = zip(tokens, surprisals)

        for token, surprisal in token_pairs:
            print(token, "\t", surprisal)


if __name__ == "__main__":
    main()
