from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16,
    )
    return model, tokenizer


if __name__ == '__main__':
    print(f'bro ran the model.py file')