import torch
import transformers
import peft
import accelerate

def format_prompt(text: str) -> str:
    prompt = [
        {
            "speaker": "ユーザー",
            "text": text
        }
    ]
    prompt = [
    f"{uttr['speaker']}: {uttr['text']}"
    for uttr in prompt
    ]
    prompt = "<NL>".join(prompt)
    prompt = (
        prompt
        + "<NL>"
        + "システム: "
    )
    return prompt


def generate_answer(text: str, tokenizer: transformers.models, model: transformers.models) -> str:
    prompt = format_prompt(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
    # generate prediction
    with torch.no_grad():
        output_ids = model.generate(
            # 先にmodelをGPUに送っておくこと
            token_ids.to(model.device),
            do_sample=True,
            max_new_tokens=1024,
            temperature=0.7,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    # format output
    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
    output = output.replace("<NL>", "\n")
    return output