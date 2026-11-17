import torch
import math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

NAT_TO_BITS = -1.0 / math.log(2.0)


class NeuralSurprisalScorer:
    def __init__(self, model_name: str, device: str = "mps"):
        self.device = device
        print("Initializing Causal Language Model Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print("Initializing the Causal Language Model itself...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()

        print("Model initialized!")

    @torch.no_grad()
    def score_text(self, text: str) -> tuple[list[str], list[float]]:
        print("Encoding text...")
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = enc["input_ids"].to(self.device)

        print("Getting output logits...")
        logits = self.model(input_ids).logits[0]

        log_probs = F.log_softmax(logits[:-1].float(), dim=-1)
        target_ids = input_ids[0, 1:]

        token_nats = (
            log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).cpu().tolist()
        )
        token_bits = [lp * NAT_TO_BITS for lp in token_nats]
        tokens = self.tokenizer.convert_ids_to_tokens(target_ids.cpu().tolist())

        return tokens, token_bits


def main():
    llama_nss = NeuralSurprisalScorer("unsloth/Llama-3.2-1B")

    tokens, token_bits = llama_nss.score_text(
        "Hello, how are you? Do you know who is Xi Jin Ping?"
    )

    token_pairs = zip(tokens, token_bits)

    for k, v in token_pairs:
        print(k, "\t", v)


if __name__ == "__main__":
    main()
