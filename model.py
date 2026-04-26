"""
model.py — Model loading and prompt building.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import MODEL_NAME, NEUTRAL_PROMPT


_tokenizer = None
_model = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        raise RuntimeError("Call load_model() before using the tokenizer.")
    return _tokenizer


def get_model():
    global _model
    if _model is None:
        raise RuntimeError("Call load_model() before using the model.")
    return _model


def load_model():
    """Load tokenizer and model. Call once at the start of the experiment."""
    global _tokenizer, _model

    print(f"Loading tokenizer: {MODEL_NAME}", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _tokenizer.pad_token = _tokenizer.eos_token

    print(f"Loading model: {MODEL_NAME}", flush=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    ).eval()
    _model.config.pad_token_id = _tokenizer.eos_token_id

    print("Model loaded.", flush=True)
    return _tokenizer, _model


def build_prompt(system_prompt: str, question_text: str) -> str:
    """
    Build a chat-formatted prompt with the given system prompt.
    Used for both neutral and discourage logits — user message is identical
    in both cases so that contrastive decoding isolates only the system
    prompt effect.
    """
    tokenizer = get_tokenizer()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"{question_text}\n\n"
            "Answer with only one letter without '.': e.g. A\nAnswer:"
        )},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def get_next_logits(prompt: str) -> torch.Tensor:
    """Return next-token logits (CPU tensor, shape [1, vocab_size])."""
    tokenizer = get_tokenizer()
    model = get_model()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model(input_ids)
    return out.logits[:, -1, :].cpu()
