# Research Plan: Low Rank Hypothesis for Humor Recognition

## 1. Research Question
**Is humor recognition in Large Language Models (LLMs) represented by a low-rank subspace in the activation space?**

Specifically, we ask:
1.  Can a simple linear probe (1D subspace) distinguish between humorous and non-humorous text with high accuracy?
2.  What is the effective dimensionality of the "humor subspace"?
3.  Does this representation emerge in specific layers of the model?

## 2. Background and Motivation
Recent work (Tigges et al., Marx et al.) suggests that high-level concepts like "sentiment" and "truth" are represented linearly (1D) in LLMs. Humor is a more complex, subjective, and often context-dependent attribute. Determining if humor is also "low rank" would:
-   Validate the universality of the "linear representation" hypothesis.
-   Enable efficient "humor steering" (making models funnier or more serious via vector addition).
-   Provide insights into how LLMs semanticize abstract concepts.

## 3. Hypothesis Decomposition
-   **H1 (Separability)**: The activation space of LLMs contains a hyperplane that separates jokes from non-jokes with high accuracy (>85%).
-   **H2 (Low Rank)**: The variance of the difference vectors (Joke - NonJoke) is concentrated in a very small number of dimensions (Top 3 PCs explain >50% variance).
-   **H3 (Layer-specificity)**: This separability is not present in early layers (lexical processing) but emerges in middle/late layers (semantic processing).

## 4. Proposed Methodology

### Approach
We will use a **mechanistic interpretability** approach involving:
1.  **Activation Extraction**: Running a pre-trained LLM (`gpt2-large`) on a balanced dataset of Jokes and Non-Jokes.
2.  **Probing**: Training linear classifiers on these activations.
3.  **Dimensionality Reduction**: Using PCA to analyze the intrinsic rank of the humor representation.

### Experimental Steps
1.  **Data Preparation**:
    -   Load `datasets/short_jokes/shortjokes.csv` (Source: Kaggle Short Jokes).
    -   Load `datasets/wikitext_hf` (Source: WikiText-2, representing "neutral/factual" text).
    -   Sample N=2000 items from each, filtering for length (e.g., 10-50 tokens) to ensure comparability.
2.  **Model Loading**:
    -   Use `gpt2-large` (774M params). It is large enough to "get" jokes but small enough for efficient local inference and activation storage.
3.  **Activation Extraction**:
    -   Pass inputs through the model.
    -   Extract residual stream activations from the **last token** at each layer.
    -   Store as `(N_samples, N_layers, D_model)`.
4.  **Analysis 1: Linear Separability**:
    -   For each layer, train a Logistic Regression classifier (with 5-fold CV).
    -   Metric: Accuracy and F1-score.
5.  **Analysis 2: Intrinsic Dimensionality**:
    -   Compute the "Humor Direction" via Difference-in-Means ($\mu_{joke} - \mu_{normal}$).
    -   Perform PCA on the union of both classes (or just the difference vectors).
    -   Calculate the **Explained Variance Ratio** of the top components.

### Baselines
-   **Random Embeddings**: Probing on a randomly initialized model (control for architecture).
-   **Lexical Baseline**: TF-IDF + Logistic Regression (control for simple word usage like "laugh", "funny").

### Evaluation Metrics
-   **Classification Accuracy**: For the linear probe.
-   **Explained Variance Ratio**: For PCA (e.g., "PC1 explains 40% of variance").
-   **Cosine Similarity**: Between the "Humor Vector" found via means and the probe's weight vector.

### Statistical Analysis Plan
-   Use 5-fold Cross-Validation for probing to ensure robustness.
-   Report Mean ± Std Dev for accuracy.
-   Compare peak accuracy against baselines using a t-test.

## 5. Timeline and Milestones (Single Session)

-   **Phase 2: Setup (10 min)**
    -   Install `torch`, `transformers`, `scikit-learn`, `pandas`, `tqdm`.
    -   Verify data loading scripts.
-   **Phase 3: Implementation (45 min)**
    -   `data_loader.py`: Prepare balanced dataset.
    -   `extract_activations.py`: Pipeline for getting hidden states.
    -   `analysis.py`: Probing and PCA logic.
-   **Phase 4: Experimentation (45 min)**
    -   Run extraction on `gpt2-large`.
    -   Run analysis sweep across all layers.
-   **Phase 5: Analysis (30 min)**
    -   Generate "Accuracy vs Layer" plot.
    -   Generate "Explained Variance" scree plot.
    -   Interpret results.
-   **Phase 6: Reporting (20 min)**
    -   Write `REPORT.md`.

## 6. Potential Challenges
-   **Memory Limits**: Storing activations for 4000 samples x 36 layers x 1280 dims is large (~700MB). *Mitigation*: Process layer-by-layer or save to disk using `numpy.memmap`.
-   **Confounders**: Jokes might just be shorter or have different punctuation (e.g., "?"). *Mitigation*: Filter by length match.
-   **Polysemy**: "Humor" is broad. The probe might just pick up on "informal style". *Acceptance*: We acknowledge this limitation; distinguishing "informal" from "funny" requires a harder negative dataset (e.g., failed jokes), which we don't have. We test "Humorous vs Neutral".

## 7. Success Criteria
-   We successfully extract activations.
-   We find a layer where Linear Probe Accuracy > 85% (beating TF-IDF).
-   We determine if the representation is low-rank (Top 3 PCs explain > 50%).
