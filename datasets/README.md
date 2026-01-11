# Downloaded Datasets

## 1. Short Jokes Dataset
- **Source**: GitHub (amoudgl/short-jokes-dataset)
- **Location**: `datasets/short_jokes/shortjokes.csv`
- **Format**: CSV
- **Content**: 200k+ short jokes.
- **Columns**: ID, Joke
- **Usage**: Positive examples for humor detection.

## 2. WikiText-2 (Raw)
- **Source**: HuggingFace Datasets (`wikitext`, `wikitext-2-raw-v1`)
- **Location**: `datasets/wikitext_hf/`
- **Format**: Arrow/HuggingFace Dataset (saved to disk)
- **Content**: Wikipedia articles (good quality English text).
- **Usage**: Negative examples (non-humor) for training/probing.

## Loading Instructions

```python
# Load Jokes
import pandas as pd
df = pd.read_csv("datasets/short_jokes/shortjokes.csv")
jokes = df['Joke'].tolist()

# Load WikiText
from datasets import load_from_disk
wiki = load_from_disk("datasets/wikitext_hf")
non_jokes = wiki['train']['text'] # List of strings
```
