import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import fire
from peft import PeftModel, PeftConfig


def generate(model, prompt, max_length=200):
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    sequences = pipeline(
        prompt,
        max_length=max_length,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


def test(
    base_model: str = "tiiuae/falcon-7b",
    checkpoint: str = "",
    prompt: str = "Write a blog on hiking in the PNW",
    max_length: int = 200,
):
    print("On baseline")
    generate(base_model)

    peft_model_id = checkpoint
    config = PeftConfig.from_pretrained(peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,
    #                                             device_map="auto",)

    pipeline = transformers.pipeline(
            "text-generation",
            model=base_model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
    pipeline.model = PeftModel.from_pretrained(pipeline.model, peft_model_id)

    sequences = pipeline(
        prompt,
        max_length=max_length,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


if __name__ == "__main__":
    fire.Fire(test)