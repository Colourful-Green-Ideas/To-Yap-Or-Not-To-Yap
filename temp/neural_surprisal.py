import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

NAT_TO_BITS = -1.0 / math.log(2.0)


class NeuralSurprisalScorer:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()

    @torch.no_grad()
    def score_text(self, text: str) -> tuple[list[str], list[float]]:
        """Returns (tokens, token_surprisals_in_bits) directly at the token level."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)

        logits = self.model(input_ids).logits[0]  # [seq_len, vocab_size]
        log_probs = F.log_softmax(logits[:-1].float(), dim=-1)
        target_ids = input_ids[0, 1:]

        token_nats = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).cpu().tolist()
        token_bits = [lp * NAT_TO_BITS for lp in token_nats]
        tokens = self.tokenizer.convert_ids_to_tokens(target_ids.cpu().tolist())

        return tokens, token_bits
