import re
import transformers
from transformers import pipeline


def format_prompt(text: str) -> str:
    prompt = f"ユーザー: {text}\nシステム: "
    return prompt


def generate_answer(
    text: str, tokenizer: transformers.models, model: transformers.models
) -> str:
    prompt = format_prompt(text)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = generator(
        prompt,
        max_length=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.1,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
    )
    output_text = output[0]["generated_text"]
    output_text = re.sub(re.escape(prompt), "", output_text)
    return output_text
