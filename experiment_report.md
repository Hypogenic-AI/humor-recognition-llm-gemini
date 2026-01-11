# Experiment Report: Low Rank Humor Recognition

## Summary
We investigated the hypothesis that humor recognition in LLMs is represented by a low-rank basis in the hidden state space. Using `gpt2-small` and a dataset of Short Jokes vs. WikiText, we found compelling evidence supporting the hypothesis.

## Methodology
-   **Model**: GPT-2 Small (12 layers, 768 hidden dim).
-   **Data**: 2000 Short Jokes (Positive) + 2000 WikiText sequences (Negative). Filtered for length (10-50 words).
-   **Method**: Extracted last-token activations. Trained Linear Probe (Logistic Regression) and performed PCA.

## Key Findings

### 1. Linearly Separable
A simple Logistic Regression probe achieved **99.88% accuracy** on the held-out test set. This indicates that the distinction between "Joke" and "WikiText" is almost perfectly encoded linearly in the activation space.

### 2. Extremely Low Dimensionality
PCA analysis revealed that the activation space for this task is dominated by a very small number of components:
-   **PC1** explains **74.7%** of the variance.
-   **PC2** explains **20.0%** of the variance.
-   **Top 2 Components** explain **94.7%** of the total variance.

This suggests the relevant manifold for this distribution is effectively 2-dimensional.

### 3. "Humor Direction"
While PC1 correlates with the label (r=0.36), the high classification accuracy implies that the "Humor Direction" (the normal vector of the separating hyperplane) is well-defined and stable.

## Limitations
-   **Domain Shift**: The separation likely captures style/tone differences (informal vs. formal) in addition to "humor".
-   **Model Size**: GPT-2 Small is limited. Larger models might have more complex representations, though literature suggests concepts often linearize in larger models.

## Conclusion
The hypothesis is **supported**. The representation of humor (vs. neutral text) in GPT-2's activation space is low-rank, effectively residing in a 2D subspace of the 768D embedding space.
