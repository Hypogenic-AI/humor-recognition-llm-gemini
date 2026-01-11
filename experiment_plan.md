# Experiment Plan: Humor Rank Analysis

## Hypothesis
Humor recognition in LLMs (specifically binary distinction between Joke and Non-Joke) is represented by a low-rank subspace (possibly a single direction) in the model's activation space.

## Architecture
We will use a modular approach:
1.  **Data Loading**: Standardize "Short Jokes" and "WikiText" into a balanced dataset of (text, label) pairs.
2.  **Activation Extraction**: Run an open-source LLM (e.g., `gpt2-small` or `Pythia-410m`) on the data and save hidden states from selected layers.
3.  **Analysis**:
    *   **Linear Probing**: Train Logistic Regression on activations. High accuracy => Linearly Separable.
    *   **PCA**: Compute Principal Component Analysis on the difference vectors (or centered class means).
    *   **Explained Variance**: Plot cumulative explained variance. Fast saturation => Low Rank.

## Steps

### 1. Environment Setup
-   Ensure Python 3.x
-   Install `transformers`, `torch`, `datasets`, `scikit-learn`, `pandas`, `matplotlib`, `tqdm`.

### 2. Data Preparation (`src/data_loader.py`)
-   Load `datasets/short_jokes/shortjokes.csv`.
-   Load `datasets/wikitext_hf`.
-   Filter samples by length (e.g., 10-50 tokens) to minimize length as a confounding factor.
-   Create a balanced dataset: `N` Jokes + `N` Non-Jokes.
-   Split into Train/Test (e.g., 80/20).

### 3. Activation Extraction (`src/extract_activations.py`)
-   Load model (start with `gpt2-small` for speed).
-   Iterate through data batches.
-   For each sample, extract the hidden state of the **last token**.
-   Save activations to `results/activations.npz` (or similar).
-   *Optional*: Extract from multiple layers (early, middle, late).

### 4. Analysis (`src/analysis.py`)
-   **Probing**:
    -   Train LogReg on Train set.
    -   Report Accuracy on Test set.
-   **Rank Analysis**:
    -   Compute Mean_Joke - Mean_NonJoke.
    -   Run PCA on the union of centered data.
    -   Calculate Intrinsic Dimension (number of components to reach 90% variance).
-   **Visualization**:
    -   Generate 2D PCA scatter plots.
    -   Generate Explained Variance plot.

### 5. Execution
-   Run the full pipeline.
-   Save metrics to `results/metrics.json`.
-   Save plots to `results/plots/`.

## Target Model
-   `gpt2-small` (12 layers, 768d) - Fast, good baseline.
-   If resources permit: `gpt2-xl` or `Llama-2-7b`.

## Success Metrics
-   **Probing Accuracy**: > 85% suggests strong linear separability.
-   **Low Rank**: If top 1 component explains > 30% variance (or top 3 > 60%), hypothesis is supported.
