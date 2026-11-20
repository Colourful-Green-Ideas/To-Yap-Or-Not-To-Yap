import math
import kenlm
from transformers import AutoTokenizer

LOG10_TO_BITS = -1.0 / math.log10(2.0)

class KenLMSurprisalScorer:
    def __init__(self, binary_path: str, tokenizer_name: str):
        self.model = kenlm.Model(binary_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def score_text(self, text: str) -> tuple[list[str], list[float]]:

        tokens = self.tokenizer.tokenizer(text)

        state_in = kenlm.State()
        state_out = kenlm.State()
        self.model.BeginSentenceWrite(state_in)

        surprisals = []
        for tok in tokens:
            log10_prob = self.model.BaseScore(state_in, tok, state_out)
            surprisals.append(log10_prob * LOG10_TO_BITS)
            state_in, state_out = state_out, state_in

        return tokens, surprisals

def main():
    kenlm_scorer = KenLMSurprisalScorer("./model_3gram.binary")
    tokens, surprisals = kenlm_scorer.score_text("Hi, how are you?")

    token_pairs = zip(tokens, surprisals)

    for token, surprisal in token_pairs:
        print(token, '\t', surprisal)

if __name__ == "__main__":
    main()
