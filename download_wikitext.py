from datasets import load_dataset
import os

try:
    print("Downloading wikitext...")
    # 'wikitext-2-raw-v1' is a standard config
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    dataset.save_to_disk('datasets/wikitext_hf')
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
