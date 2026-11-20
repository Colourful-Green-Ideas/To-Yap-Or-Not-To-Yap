import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

NAT_TO_BITS = -1.0 / math.log(2.0)


class NeuralSurprisalScorer:
    def __init__(self, model_name: str, device: str = "mps"):
        self.model_name = model_name
        self.device = device
        print(f"Loading tokenizer & model: {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()

    @torch.no_grad()
    def score_text(self, text: str) -> tuple[list[str], list[float]]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        logits = self.model(input_ids).logits[0]

        log_probs = F.log_softmax(logits[:-1].float(), dim=-1)
        target_ids = input_ids[0, 1:]

        token_nats = (
            log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).cpu().tolist()
        )
        token_bits = [lp * NAT_TO_BITS for lp in token_nats]
        tokens = self.tokenizer.convert_ids_to_tokens(target_ids.cpu().tolist())
        return tokens, token_bits

    @torch.no_grad()
    def score_continuation(
        self, prompt: str, cont: str, max_length: int = 1024
    ) -> tuple[list[str], list[float]]:
        """Scores continuation tokens conditioned on the full prompt prefix."""
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True, truncation=True, max_length=max_length // 2
        )["input_ids"]
        cont_ids = self.tokenizer(
            " " + cont.lstrip(),
            add_special_tokens=False,
            truncation=True,
            max_length=max_length // 2,
        )["input_ids"]

        if not cont_ids:
            return [], []

        full_ids = torch.tensor([prompt_ids + cont_ids], device=self.device)
        logits = self.model(full_ids).logits[0]

        # Shift by 1: logit at index t-1 predicts token at index t
        start_idx = len(prompt_ids) - 1
        end_idx = len(prompt_ids) + len(cont_ids) - 1

        cont_logits = logits[start_idx:end_idx].float()
        log_probs = F.log_softmax(cont_logits, dim=-1)
        target_tensor = torch.tensor(cont_ids, device=self.device)

        token_nats = (
            log_probs.gather(-1, target_tensor.unsqueeze(-1)).squeeze(-1).cpu().tolist()
        )
        token_bits = [lp * NAT_TO_BITS for lp in token_nats]
        tokens = self.tokenizer.convert_ids_to_tokens(cont_ids)
        return tokens, token_bits


if __name__ == "__main__":
    scorer = NeuralSurprisalScorer("bigscience/bloom-3b")
    tokens, bits = scorer.score_continuation(
        "The scientific method relies on",
        "empirical observation and hypothesis testing.",
    )
    for t, b in zip(tokens, bits):
        print(f"{t}\t{b:.4f}")