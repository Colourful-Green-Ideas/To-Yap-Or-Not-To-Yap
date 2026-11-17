import math
import pickle
import unicodedata
from collections import defaultdict
from transformers import AutoTokenizer

DISCOUNT = 0.75
B = 1 << 18
BOS = B - 2
EOS = B - 1


class KneserNeyTrigram:
    def train(self, sequences):
        """sequences: list of token-id lists (one per paragraph)"""
        c3 = defaultdict(int)
        for seq in sequences:
            s = [BOS, BOS] + seq + [EOS]
            for i in range(2, len(s)):
                c3[(s[i - 2] * B + s[i - 1]) * B + s[i]] += 1

        ctx3_count = defaultdict(int)
        ctx3_types = defaultdict(int)
        cont2 = defaultdict(int)
        for key, count in c3.items():
            uv, w = divmod(key, B)
            ctx3_count[uv] += count
            ctx3_types[uv] += 1
            cont2[(uv % B) * B + w] += 1

        ctx2_count = defaultdict(int)
        ctx2_types = defaultdict(int)
        cont1 = defaultdict(int)
        for key, n in cont2.items():
            v, w = divmod(key, B)
            ctx2_count[v] += n
            ctx2_types[v] += 1
            cont1[w] += 1

        self.c3, self.ctx3_count, self.ctx3_types = c3, ctx3_count, ctx3_types
        self.cont2, self.ctx2_count, self.ctx2_types = cont2, ctx2_count, ctx2_types
        self.cont1 = cont1
        self.total1 = sum(cont1.values())
        self.types1 = len(cont1)

    def p1(self, w):
        n = self.cont1.get(w, 0)
        return (max(n - DISCOUNT, 0) + DISCOUNT * self.types1 / B) / self.total1

    def p2(self, v, w):
        den = self.ctx2_count.get(v, 0)
        if den == 0:
            return self.p1(w)
        n = self.cont2.get(v * B + w, 0)
        return (max(n - DISCOUNT, 0) + DISCOUNT * self.ctx2_types[v] * self.p1(w)) / den

    def p3(self, u, v, w):
        uv = u * B + v
        den = self.ctx3_count.get(uv, 0)
        if den == 0:
            return self.p2(v, w)
        c = self.c3.get(uv * B + w, 0)
        return (
            max(c - DISCOUNT, 0) + DISCOUNT * self.ctx3_types[uv] * self.p2(v, w)
        ) / den

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))


class KNSurprisalScorer:
    def __init__(self, model_path: str, tokenizer_name: str = "bigscience/bloom-3b"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.model = KneserNeyTrigram()
        self.model.load(model_path)

    def ids(self, text: str) -> list[int]:
        norm_text = unicodedata.normalize("NFC", text)
        return self.tokenizer(norm_text, add_special_tokens=False)["input_ids"]

    def score_continuation(
        self, prompt: str, cont: str
    ) -> tuple[list[str], list[float]]:
        history = [BOS, BOS] + self.ids(prompt)
        cont_ids = self.ids(" " + cont.lstrip())

        surprisals = []
        for w in cont_ids:
            prob = self.model.p3(history[-2], history[-1], w)
            surprisals.append(-math.log2(prob))
            history.append(w)

        tokens = self.tokenizer.convert_ids_to_tokens(cont_ids)
        return tokens, surprisals