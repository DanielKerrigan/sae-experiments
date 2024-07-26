from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")
dataset.save_to_disk("../datasets/roneneldan/TinyStories")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.save_pretrained("../models/EleutherAI/gpt-neo-125M")

model_names = [
    "roneneldan/TinyStories-1M",
    "roneneldan/TinyStories-3M",
    "roneneldan/TinyStories-8M",
    "roneneldan/TinyStories-28M",
    "roneneldan/TinyStories-33M",
]

for model_name in model_names:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(f"../models/{model_name}")
