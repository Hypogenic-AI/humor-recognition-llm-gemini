import torch
from transformers import GPT2Model, GPT2Tokenizer
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def extract_activations(data_path, output_dir, model_name='gpt2-large', batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(model_name).to(device)
    model.eval()
    
    # Load Data
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Register Hooks
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            # output[0] is hidden state: (batch, seq, hidden)
            # We want the last token for each sequence in the batch
            # Note: GPT2 is causal, so last token sees everything.
            # However, if padding is used (left or right), we must be careful.
            # Here we assume right padding for simplicity in batching, 
            # but we need to select the actual last token index.
            pass # We will handle extraction in the loop to be safer with indices
        return hook

    # We will run forward pass and grab hidden_states output from the model directly
    # GPT2Model returns hidden_states if output_hidden_states=True
    
    all_activations = [] # List of dicts {layer: tensor}
    
    print("Extracting activations...")
    num_layers = model.config.n_layer
    # Pre-allocate storage to avoid memory fragmentation? 
    # Actually, collecting list of batch arrays is fine for 700MB.
    
    layer_storage = {i: [] for i in range(num_layers + 1)} # +1 for embeddings/final? Just layers 0-35 + final
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize with padding
        encoded = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
        # outputs.hidden_states is a tuple of (n_layer + 1) tensors
        # Index 0 is embeddings, Index 1 is first layer output, ..., Index -1 is last layer
        # We want layers 1 to N (outputs of blocks) and maybe embeddings?
        # Standard practice: Take outputs of each transformer block.
        # hidden_states[1] is output of block 0.
        
        # Find index of last real token for each item in batch
        # If padding is on the right (default), last token is at sum(mask) - 1
        last_token_indices = attention_mask.sum(1) - 1
        
        for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
            # layer_hidden: (batch, seq, dim)
            # Select last token
            batch_last_acts = layer_hidden[torch.arange(layer_hidden.size(0)), last_token_indices]
            layer_storage[layer_idx].append(batch_last_acts.cpu().numpy())
            
    # Concatenate
    print("Concatenating and saving...")
    final_activations = {}
    for layer_idx, batches in layer_storage.items():
        final_activations[layer_idx] = np.concatenate(batches, axis=0)
    
    # Save labels too
    np.savez(os.path.join(output_dir, 'activations.npz'), 
             labels=np.array(labels), 
             **{f'layer_{k}': v for k, v in final_activations.items()})
    
    print(f"Saved activations to {output_dir}/activations.npz")

if __name__ == "__main__":
    extract_activations(
        'datasets/processed_dataset.csv',
        'results/activations'
    )