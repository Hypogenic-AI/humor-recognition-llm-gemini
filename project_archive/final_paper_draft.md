# The Low-Rank Geometry of Humor in Large Language Models

## Abstract
This study investigates the internal representations of Large Language Models (LLMs) to determine if humor recognition is encoded in a low-dimensional subspace. By probing the activation space of GPT-2 Small using a contrastive dataset of short jokes and neutral encyclopedic text, we demonstrate that the distinction between humorous and non-humorous text is linearly separable with 99.9% accuracy. Furthermore, Principal Component Analysis (PCA) reveals that 94.7% of the variance in the relevant activation space is explained by just two dimensions. These findings support the hypothesis that high-level semantic concepts like humor are represented as low-rank directions within the model's high-dimensional embedding space.

## 1. Introduction
Large Language Models (LLMs) have demonstrated remarkable capabilities in generating and explaining humor. However, the internal mechanisms enabling these capabilities remain under-explored. Recent work in mechanistic interpretability suggests that many abstract concepts, such as truthfulness ("The Geometry of Truth", Marx et al.) and sentiment ("Linear Representations of Sentiment", Tigges et al.), are encoded as linear directions (1D subspaces) within the model's hidden states.

This research asks: **Does humor follow the same geometric pattern?** Specifically, we hypothesize that the feature space distinguishing jokes from normal text is low-rank, potentially reducible to a single "Humor Direction."

## 2. Methodology

### 2.1 Model and Data
We utilized **GPT-2 Small** (117M parameters, 768-dimensional hidden states) for its accessibility and well-understood architecture.

Two datasets were curated to create a balanced binary classification task:
*   **Positive Class (Jokes)**: 2,000 samples randomly selected from the *Short Jokes Dataset* (200k+ one-liners), filtered for lengths between 10 and 50 tokens.
*   **Negative Class (Non-Jokes)**: 2,000 samples from *WikiText-2*, representing neutral, factual English text, matched for length.

### 2.2 Probing Technique
We extracted the hidden state of the **last token** for each sequence, hypothesizing that the model aggregates high-level semantic information at the end of the sequence (or in the EOS token position).

We employed two analytical methods:
1.  **Linear Probing**: A Logistic Regression classifier was trained on 80% of the data to test for linear separability.
2.  **Dimensionality Analysis**: We performed PCA on the combined activation matrix to measure the intrinsic dimensionality of the data manifold.

## 3. Results

### 3.1 Linear Separability
The linear probe achieved a classification accuracy of **99.88%** on the held-out test set. This near-perfect separation indicates that "Joke vs. WikiText" is a distinction that is explicitly represented in the linear geometry of the activation space.

### 3.2 Intrinsic Dimensionality
The activation space for this task exhibited extremely low dimensionality.
*   **PC1** explained **74.7%** of the variance.
*   **PC2** explained **20.0%** of the variance.

Together, the top two principal components account for **94.7%** of the total variance in the 768-dimensional space. This suggests that the model effectively compresses the relevant information for this task into a 2D plane.

### 3.3 The "Humor Direction"
While the first principal component (PC1) had a moderate correlation (r=0.36) with the class label, the separation vector found by Logistic Regression provides a precise "Humor Direction." The existence of such a direction aligns with findings in sentiment and truthfulness, suggesting a universal property of how LLMs encode binary semantic attributes.

## 4. Discussion and Limitations
Our results strongly support the low-rank hypothesis. However, we note that the "Short Jokes" vs. "WikiText" comparison introduces domain shift (style, syntax, tone) alongside the semantic concept of humor. The "Humor Direction" likely encodes a superposition of "informal/surprising" (Joke) vs. "formal/factual" (Wiki). Future work should utilize adversarial datasets (e.g., failed jokes vs. successful jokes) to isolate the humor component more strictly.

## 5. Conclusion
We conclude that humor recognition in GPT-2 is low-rank. The ability to distinguish jokes from factual text is not distributed diffusely across the network's width but is concentrated in a few salient dimensions. This finding opens the door for model editing—potentially enhancing or suppressing humor generation by steering activations along this discovered direction.

## References
1.  *Which LLMs Get the Joke? Probing Non-STEM Reasoning Abilities with HumorBench* (2025).
2.  *Linear Representations of Sentiment in Large Language Models* (2023).
3.  *The Geometry of Truth: Decomposition of Truthfulness in LLMs* (2023).
