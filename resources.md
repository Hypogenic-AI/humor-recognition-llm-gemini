# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "How low rank is humor recognition in LLMs?".

## Papers
Total papers: 5

| Title | File | Key Info |
|-------|------|----------|
| Which LLMs Get the Joke? | papers/humorbench_...pdf | Benchmark for humor |
| Linear Representations of Sentiment | papers/linear_sentiment_...pdf | Methodology for 1D concepts |
| UR-FUNNY | papers/understanding_humor_...pdf | Humor dataset paper |
| LoRA | papers/lora_...pdf | Low-rank weight adaptation |
| Intrinsic Dimensionality | papers/intrinsic_dim_...pdf | Subspaces in LLMs |

## Datasets
Total datasets: 2 (Positive/Negative pair)

| Name | Location | Format | Count | Description |
|------|----------|--------|-------|-------------|
| Short Jokes | datasets/short_jokes/shortjokes.csv | CSV | 200k+ | Positive class (Jokes) |
| WikiText-2 | datasets/wikitext_hf/ | HF Dataset | ~40k | Negative class (Normal text) |

## Code Repositories
Total repos: 2

| Name | Location | Purpose |
|------|----------|---------|
| Geometry of Truth | code/geometry_of_truth/ | Code for finding linear directions |
| Eliciting Sentiment | code/eliciting_latent_sentiment/ | Code for sentiment analysis circuits |

## Recommendations for Experiment Design

1.  **Primary Task**: Binary classification (Joke vs Non-Joke) using internal activations.
2.  **Method**:
    -   Load `shortjokes.csv` and `wikitext`.
    -   Sample N=2000 from each.
    -   Use a model like `gpt2-xl` or `llama-2-7b` (via HuggingFace `transformers`).
    -   Collect activations from the last token (or mean pooling).
    -   Train Logistic Regression (Linear Probe).
    -   Perform PCA on the "Joke" activations centered by "Non-Joke" mean.
3.  **Evaluation**:
    -   Accuracy of Linear Probe (Is it linearly separable?).
    -   Explained Variance of top PCA components (Is it low rank?).
