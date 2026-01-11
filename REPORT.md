# Research Report: Low Rank Hypothesis for Humor Recognition in LLMs

## 1. Executive Summary
This study investigated the internal representation of humor in the GPT-2 Large language model. We hypothesized that humor recognition is a "low-rank" task, meaning it can be represented by a low-dimensional subspace (or even a single direction) within the model's activation space. Our experiments comparing 2,000 short jokes against 2,000 neutral WikiText samples confirm this hypothesis with striking strength: **humor (specifically, joke vs. non-joke text) is linearly separable with 100% accuracy even in the first layer**, and a single principal component (Rank 1) is sufficient to capture this distinction. This suggests that "Joke Mode" vs "Factual Mode" is a fundamental, highly distinct feature in the model's latent geometry.

## 2. Goal
The primary research question was: **How low rank is humor recognition in LLMs?**
Understanding the geometry of concepts like humor allows for:
-   Better steerability (e.g., adding a "humor vector" to generation).
-   Insight into how LLMs organize semantic information.
-   Verification of the "Linear Representation Hypothesis" for subjective concepts.

## 3. Data Construction

### Datasets
-   **Positive Class (Jokes)**: 2,000 samples from the *Short Jokes Dataset* (Kaggle).
-   **Negative Class (Non-Jokes)**: 2,000 samples from *WikiText-2* (HuggingFace).

### Preprocessing
-   **Filtering**: Both datasets were filtered to include only texts between 15 and 60 tokens. This controls for length as a confounding variable.
-   **Balancing**: The final dataset was perfectly balanced (50/50 split).

### Example Samples
**Joke**:
> "I can't believe I got fired from the calendar factory. All I did was take a day off."

**Non-Joke (WikiText)**:
> "The game was released in Japan on December 12 , 1999 , and in North America on February 29 , 2000 ."

## 4. Methodology

### Model
-   **Architecture**: `gpt2-large` (774M parameters).
-   **Access**: We extracted residual stream activations from the **last token** of the input sequence.

### Experimental Protocol
1.  **Activation Extraction**: We ran the 4,000 samples through the model and saved activations for all 36 layers.
2.  **Linear Probing**: We trained a Logistic Regression classifier on each layer's activations (5-fold CV) to measure linear separability.
3.  **Dimensionality Analysis**:
    -   **1D Difference Mean**: We computed the vector $\vec{d} = \vec{\mu}_{joke} - \vec{\mu}_{non}$, projected data onto it, and measured classification accuracy.
    -   **PCA Rank Analysis**: We performed PCA on the training data and measured probing accuracy using only the top $k$ components.

## 5. Result Analysis

### Key Findings

1.  **Perfect Linear Separability**:
    -   The linear probe achieved **100.0% accuracy** across ALL layers (0 to 36).
    -   This indicates that the distinction between "Joke" and "WikiText" is fully encoded even in the initial embeddings or first attention block.

2.  **It is Rank 1**:
    -   The "Difference-in-Means" classifier (which projects data onto a single line) also achieved **~100% accuracy**.
    -   PCA analysis showed that using just the **Top 1 Principal Component** yields >99% accuracy.

### Visualizations

#### Accuracy vs Layer
*(See `results/plots/accuracy_vs_layer.png`)*
The curve is flat at 1.0. This is unusual for complex semantic tasks (which usually improve in later layers) and suggests that the model distinguishes these domains immediately, likely based on lexical or stylistic cues (e.g., "I", "my", question marks vs. dates, formal nouns).

#### Accuracy vs Rank
*(See `results/plots/accuracy_vs_rank.png`)*
-   Rank 1: 99.1% (Layer 0) -> 100% (Layer 6+)
-   Rank 2: 100%
The subspace for this task is effectively 1-dimensional.

### Interpretation & Limitations
The hypothesis "Humor is low rank" is supported, but with a caveat:
-   **Style vs. Concept**: The task likely measured "Conversational/Joke Style" vs "Encyclopedic Style". The model didn't necessarily "get" the joke, but it instantly recognized the *format* of a joke.
-   **Easy Negatives**: WikiText is a very easy negative class. A more rigorous test would be "Jokes vs Serious conversational text" or "Jokes vs Failed Jokes".

## 6. Conclusion
Humor recognition (defined as distinguishing jokes from encyclopedic text) is an **extremely low-rank (Rank 1)** phenomenon in GPT-2 Large. The representation is robust, global, and present from the very first layer. This implies that "genre" or "mode" is one of the most dominant features in the model's activation space, far overpowering subtler semantic nuances.

## 7. Next Steps
1.  **Harder Negatives**: Repeat the experiment using "Common Crawl" or "Reddit" comments as the negative class to control for conversational style.
2.  **Steering**: Attempt to add the computed "Humor Vector" to the activations during generation to see if it makes the model output funnier or just more informal.
3.  **Fine-grained Analysis**: can we distinguish *types* of humor (Puns vs Sarcasm)?
