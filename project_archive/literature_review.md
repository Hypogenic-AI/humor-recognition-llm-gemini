# Literature Review: Low Rank Hypothesis for Humor Recognition

## Research Area Overview
The research investigates the internal representations of Large Language Models (LLMs) regarding high-level concepts like humor. Recent work in "mechanistic interpretability" and "linear representations" suggests that many semantic concepts (truthfulness, sentiment, toxicity) are represented as linear directions (1D subspaces) or low-rank subspaces within the high-dimensional activation space of the model. This project aims to verify if **humor** follows this pattern.

## Key Papers

### 1. Which LLMs Get the Joke? Probing Non-STEM Reasoning Abilities with HumorBench
- **Key Contribution**: Establishes that LLMs have varying capabilities in recognizing and explaining humor. It provides a benchmark (HumorBench) but focuses more on *output generation* and *reasoning* rather than internal geometry.
- **Relevance**: Confirms that "humor" is a distinct capability that LLMs possess, making it a valid target for probing internal states.

### 2. Linear Representations of Sentiment in Large Language Models (Tigges et al.)
- **Methodology**: Used PCA on difference vectors (Positive - Negative) and trained linear probes (Logistic Regression) on activations. Found that a single direction captures sentiment.
- **Relevance**: Provides the exact methodology we should replicate: simple difference-in-means or PCA on contrastive pairs (Joke vs Non-Joke).

### 3. The Geometry of Truth (Marx et al.)
- **Methodology**: Demonstrated that "truth" is represented by a consistent linear direction across different topics. Used "mass mean" probes (vector difference of means).
- **Relevance**: Supports the "low rank" hypothesis for abstract concepts. If truth and sentiment are low rank, humor is likely low rank as well.

### 4. LoRA: Low-Rank Adaptation
- **Relevance**: While focused on weights, it highlights that the "change" required to adapt a model to a new task (like telling jokes) might be low rank, supporting the idea that the underlying feature space is compressible.

## Methodologies

### Probing Techniques
1.  **Linear Probes**: Train a logistic regression classifier on the residual stream activations of the LLM to distinguish Jokes from Non-Jokes.
2.  **PCA / SVD**: Perform PCA on the activations of a dataset of jokes. If the first few principal components explain a disproportionate amount of variance compared to random text, the representation is low rank.
3.  **Difference-in-Means**: Calculate $\mu_{joke} - \mu_{non\_joke}$. This vector defines a 1D subspace. Check if projecting onto this vector classifies held-out data well.

## Datasets
-   **Positive**: Short Jokes Dataset (200k one-liners).
-   **Negative**: WikiText (neutral, non-humorous facts).
-   **Contrastive**: ideally, we would want "failed jokes" or "serious versions of jokes", but WikiText serves as a baseline for "normal text".

## Gaps and Opportunities
-   Most work focuses on Sentiment and Truth. Humor is more subjective and context-dependent.
-   It is unknown if humor is a single direction (global "funny" vector) or a subspace (different types of humor: puns, sarcasm, slapstick).
-   **Research Question**: Is there a single "Humor Direction"?

## Recommendations for Experiment
1.  **Extract Activations**: Run a model (e.g., Llama-2-7b or GPT-J) on 1000 Jokes and 1000 WikiText samples.
2.  **Layer-wise Analysis**: Measure probing accuracy across layers. Humor might emerge in middle-to-late layers.
3.  **Dimensionality Analysis**:
    -   Compute PCA of the [Joke - NonJoke] difference vectors.
    -   Plot the explained variance ratio. If top 1-3 components explain >50% of variance, the hypothesis is supported.
