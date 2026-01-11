# Humor Recognition in LLMs: A Low-Rank Analysis

## Project Overview
This project investigates the internal geometry of "humor" in Large Language Models (LLMs). Specifically, we test the hypothesis that humor recognition is a "low-rank" task—meaning the difference between a joke and a non-joke can be captured by a simple 1D direction or low-dimensional subspace in the model's activation space.

## Key Findings
-   **100% Separability**: GPT-2 Large can distinguish Jokes from WikiText with 100% accuracy using a linear probe.
-   **Rank 1 Representation**: A single direction (Difference-in-Means) captures >99% of the signal needed for classification.
-   **Early Emergence**: This distinction is present from Layer 0, suggesting a strong lexical/stylistic signature for the "Joke" genre.

## Repository Structure
-   `src/`: Python source code for data loading, extraction, and analysis.
-   `datasets/`: Contains the Short Jokes and WikiText datasets.
-   `results/`:
    -   `activations/`: Saved activation files (large, not in git).
    -   `plots/`: Visualizations of accuracy and rank.
    -   `metrics.json`: Raw experimental results.
-   `REPORT.md`: Full scientific report of the findings.

## Reproducibility
1.  **Environment Setup**:
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

2.  **Run Pipeline**:
    ```bash
    # 1. Prepare Data
    python src/data_loader.py
    
    # 2. Extract Activations (Requires GPU recommended, or edit batch size)
    python src/extract_activations.py
    
    # 3. Run Analysis
    python src/analysis.py
    ```

## Resources
-   **Model**: `gpt2-large`
-   **Datasets**: Kaggle Short Jokes, WikiText-2