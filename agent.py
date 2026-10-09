### COMPLETE THE CODE ###

import torch
from pathlib import Path
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = Path(__file__).resolve().parent / "models" / "qwen3-0.6b"


def load_model():

    if not (MODEL_PATH / "config.json").exists():
        snapshot_download(
            "Qwen/Qwen3-0.6B",
            local_dir=str(MODEL_PATH)
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float32,
        local_files_only=True
    )

    model.eval()

    return model, tokenizer


def generate_answer(question, context, model, tokenizer):

    system_prompt = (
        "You are a university IT support assistant. "
        "Answer using only the provided university IT policies. "
        "If the requested information is not present in the policies, "
        "explicitly say that the information is not available. "
        "Never invent phone numbers, contact details, or other facts. "
        "Do not provide unrelated advice. "
        "Keep your answer concise."
    )

    user_prompt = (
        "University IT policies:\n"
        + context
        + "\n\nQuestion:\n"
        + question
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()