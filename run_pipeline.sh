#!/bin/bash
set -e

echo "==========================================="
echo "   Humor Recognition Experiment Pipeline   "
echo "==========================================="

# 1. Check for Dependencies
echo "[1/4] Checking Python environment..."
if ! python3 -c "import torch" &> /dev/null; then
    echo "Error: torch not found. Please run: pip install -r requirements.txt"
    exit 1
fi

# 2. Data Verification
echo "[2/4] Verifying Datasets..."
if [ ! -f "datasets/short_jokes/shortjokes.csv" ]; then
    echo "Warning: Short Jokes dataset not found at datasets/short_jokes/shortjokes.csv"
    echo "Please download it or check the README."
    # Optional: Automate download here if URL is stable
fi

if [ ! -d "datasets/wikitext_hf" ]; then
    echo "Downloading WikiText..."
    python3 download_wikitext.py
fi

# 3. Extraction
echo "[3/4] Extracting Activations (GPT-2 Small)..."
python3 src/extract_activations.py

# 4. Analysis
echo "[4/4] Running Analysis & Generating Plots..."
python3 src/analysis.py

echo "==========================================="
echo "             Pipeline Complete             "
echo "==========================================="
echo "Results available in results/plots/"
echo "Metrics available in results/metrics.json"
