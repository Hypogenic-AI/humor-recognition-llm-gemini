import pandas as pd
from datasets import load_from_disk
from transformers import GPT2Tokenizer
import numpy as np
import os

def load_and_preprocess_data(joke_path, wiki_path, n_samples=2000, min_len=15, max_len=60, seed=42):
    """
    Loads jokes and wikitext, filters by length, and balances classes.
    Returns a DataFrame with 'text' and 'label' (1=Joke, 0=NonJoke).
    """
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large') # Use same tokenizer as model
    
    # 1. Load Jokes
    print(f"Loading jokes from {joke_path}...")
    df_jokes = pd.read_csv(joke_path)
    jokes = df_jokes['Joke'].dropna().tolist()
    
    # 2. Load WikiText
    print(f"Loading wikitext from {wiki_path}...")
    # Try loading as a dataset from disk, if fails, try loading as standard text
    try:
        wiki_ds = load_from_disk(wiki_path)
        # Combine train, test, val if available or just use train
        if 'train' in wiki_ds:
            texts = wiki_ds['train']['text']
        else:
            texts = wiki_ds['text']
    except Exception as e:
        print(f"Failed to load wikitext as HF dataset: {e}")
        return None

    non_jokes = [t for t in texts if len(t.strip()) > 0]
    
    # 3. Filter by token length
    print("Filtering by length...")
    
    def filter_text(text_list, label):
        filtered = []
        for t in text_list:
            # Quick length check by chars first (approx)
            if len(t) < min_len * 3 or len(t) > max_len * 6: 
                continue
                
            tokens = tokenizer.encode(t, add_special_tokens=False)
            if min_len <= len(tokens) <= max_len:
                filtered.append({'text': t, 'label': label})
                
            if len(filtered) >= n_samples * 2: # Optimize: stop if we have enough
                break
        return filtered

    processed_jokes = filter_text(jokes, 1)
    processed_non_jokes = filter_text(non_jokes, 0)
    
    print(f"Found {len(processed_jokes)} valid jokes and {len(processed_non_jokes)} valid non-jokes.")
    
    # 4. Sample and Balance
    np.random.seed(seed)
    
    final_jokes = np.random.choice(processed_jokes, min(n_samples, len(processed_jokes)), replace=False)
    final_non_jokes = np.random.choice(processed_non_jokes, min(n_samples, len(processed_non_jokes)), replace=False)
    
    combined = list(final_jokes) + list(final_non_jokes)
    np.random.shuffle(combined)
    
    df_final = pd.DataFrame(combined)
    print(f"Final dataset size: {len(df_final)}")
    return df_final

if __name__ == "__main__":
    df = load_and_preprocess_data(
        'datasets/short_jokes/shortjokes.csv',
        'datasets/wikitext_hf',
        n_samples=2000
    )
    if df is not None:
        df.to_csv('datasets/processed_dataset.csv', index=False)
        print("Saved processed dataset to datasets/processed_dataset.csv")