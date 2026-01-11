import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

def run_analysis(activations_path, output_dir):
    print(f"Loading activations from {activations_path}...")
    data = np.load(activations_path)
    labels = data['labels']
    
    # Identify layers
    layers = [k for k in data.keys() if k.startswith('layer_')]
    # Sort layers by index
    layers.sort(key=lambda x: int(x.split('_')[1]))
    
    results = {
        'layer_indices': [],
        'full_probe_acc': [],
        'diff_mean_acc': [],
        'pca_rank_analysis': {} # layer_idx -> {n_components: acc}
    }
    
    print("Running analysis per layer...")
    
    for layer in tqdm(layers):
        layer_idx = int(layer.split('_')[1])
        X = data[layer]
        y = labels
        
        # Split train/test for rigorous evaluation
        # We'll use 5-fold CV for the main metrics
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 1. Full Linear Probe Accuracy
        # Simple Logistic Regression
        lr = LogisticRegression(max_iter=1000, solver='liblinear') # Liblinear is good for high dim
        scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
        mean_acc = scores.mean()
        
        # 2. Difference-in-Means (1D Subspace) Accuracy
        # To do this correctly with CV:
        # Fit mean_diff on train, test on test.
        diff_scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Compute means
            mu_pos = X_train[y_train==1].mean(axis=0)
            mu_neg = X_train[y_train==0].mean(axis=0)
            diff_vec = mu_pos - mu_neg
            
            # Project
            proj_train = X_train @ diff_vec
            proj_test = X_test @ diff_vec
            
            # Find optimal threshold on train (1D classifier)
            # Simple approach: fit logistic regression on 1D projection
            lr_1d = LogisticRegression()
            lr_1d.fit(proj_train.reshape(-1, 1), y_train)
            score = lr_1d.score(proj_test.reshape(-1, 1), y_test)
            diff_scores.append(score)
            
        mean_diff_acc = np.mean(diff_scores)
        
        results['layer_indices'].append(layer_idx)
        results['full_probe_acc'].append(mean_acc)
        results['diff_mean_acc'].append(mean_diff_acc)
        
        # 3. Rank Analysis (Only for middle and last layer to save time, or stride)
        # Doing it for all layers might be slow. Let's do stride of 6 (0, 6, 12, ...)
        if layer_idx % 6 == 0 or layer_idx == len(layers)-1:
            print(f"  Analyzing rank for layer {layer_idx}...")
            # We want to see how accuracy scales with PCA components
            rank_accs = {}
            
            # Use one split for this detailed analysis to save time
            train_idx, test_idx = next(cv.split(X, y))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit PCA on X_train (unsupervised, or just on jokes? Standard is on whole dataset)
            # We use PCA to find principal directions of VARIANCE.
            # Then we use those components for LR.
            pca = PCA(n_components=50) # Top 50 components should be enough
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            components_to_test = [1, 2, 3, 4, 5, 10, 20, 50]
            for n in components_to_test:
                lr_pca = LogisticRegression(solver='liblinear')
                lr_pca.fit(X_train_pca[:, :n], y_train)
                acc = lr_pca.score(X_test_pca[:, :n], y_test)
                rank_accs[n] = acc
                
            results['pca_rank_analysis'][layer_idx] = rank_accs

    # Save results
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    # Plotting
    plot_results(results, output_dir)
    print("Analysis complete.")

def plot_results(results, output_dir):
    layers = results['layer_indices']
    
    # Plot 1: Accuracy vs Layer
    plt.figure(figsize=(10, 6))
    plt.plot(layers, results['full_probe_acc'], label='Full Rank Probe (LogisticReg)', marker='o')
    plt.plot(layers, results['diff_mean_acc'], label='1D Mean Difference', marker='x')
    plt.xlabel('Layer Index')
    plt.ylabel('Classification Accuracy')
    plt.title('Humor Recognition Accuracy across GPT2-Large Layers')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_layer.png'))
    plt.close()
    
    # Plot 2: Rank Analysis
    plt.figure(figsize=(10, 6))
    for layer_idx, rank_data in results['pca_rank_analysis'].items():
        # rank_data keys are strings in json, but integers here
        comps = sorted([int(k) for k in rank_data.keys()])
        accs = [rank_data[k] for k in comps]
        plt.plot(comps, accs, label=f'Layer {layer_idx}', marker='.')
        
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Subspace Rank')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_rank.png'))
    plt.close()

if __name__ == "__main__":
    run_analysis(
        'results/activations/activations.npz',
        'results/plots'
    )