import math
import kenlm
from transformers import AutoTokenizer

LOG10_TO_BITS = -1.0 / math.log10(2.0)


class LexicalSurprisalScorer:
    def __init__(self, binary_path: str, tokenizer_name: str):
        self.model = kenlm.Model(binary_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def score_text(self, text: str) -> tuple[list[str], list[float]]:
        """Returns (tokens, token_surprisals_in_bits) for a raw text string."""
        tokens = self.tokenizer.tokenize(text)

        state_in = kenlm.State()
        state_out = kenlm.State()
        self.model.BeginSentenceWrite(state_in)

        surprisals = []
        for tok in tokens:
            log10_prob = self.model.BaseScore(state_in, tok, state_out)
            surprisals.append(log10_prob * LOG10_TO_BITS)
            state_in, state_out = state_out, state_in

        return tokens, surprisals
