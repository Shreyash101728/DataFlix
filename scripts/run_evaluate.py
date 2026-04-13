"""
DataFlix — Evaluation & Visualization Script
Evaluate all saved models, generate Table 1, and produce plots.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    RESULTS_DIR, PROCESSED_DIR, DEVICE, TRAIN_CSV, TEST_CSV
)


def plot_training_curves():
    """Plot training loss and validation RMSE curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Path A
    path_a_hist = RESULTS_DIR / "dataflix_path_a_history.json"
    if path_a_hist.exists():
        with open(path_a_hist) as f:
            hist = json.load(f)
        
        epochs = range(1, len(hist["train_loss"]) + 1)
        
        axes[0].plot(epochs, hist["train_loss"], "b-", label="Train Loss")
        axes[0].plot(epochs, hist["val_loss"], "r--", label="Val Loss")
        axes[0].set_title("Path A (MSE) — Loss Curves", fontsize=14)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        ax2 = axes[0].twinx()
        ax2.plot(epochs, hist["val_rmse"], "g-.", label="Val RMSE", alpha=0.7)
        ax2.set_ylabel("RMSE", color="green")
        ax2.legend(loc="center right")
    
    # Path B
    path_b_hist = RESULTS_DIR / "dataflix_path_b_history.json"
    if path_b_hist.exists():
        with open(path_b_hist) as f:
            hist = json.load(f)
        
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[1].plot(epochs, hist["train_loss"], "b-", label="BPR Loss")
        axes[1].set_title("Path B (BPR) — Loss Curve", fontsize=14)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("BPR Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {RESULTS_DIR / 'training_curves.png'}")


def plot_umap_embeddings():
    """Generate UMAP visualization of learned embeddings."""
    try:
        import umap
    except ImportError:
        print("umap-learn not installed. Skipping UMAP plot.")
        return
    
    model_path = RESULTS_DIR / "dataflix_path_a.pt"
    if not model_path.exists():
        print("No saved model found. Skipping UMAP plot.")
        return
    
    # Load model and extract embeddings
    stats = json.load(open(PROCESSED_DIR / "stats.json"))
    
    from src.models.hybrid import DataFlixModel
    model = DataFlixModel(stats["n_users"], stats["n_movies"], path="A")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    
    # Get item embeddings
    item_emb = model.item_embedding.weight.detach().numpy()
    
    # Sample if too many
    n_sample = min(5000, len(item_emb))
    indices = np.random.choice(len(item_emb), n_sample, replace=False)
    sample_emb = item_emb[indices]
    
    # UMAP
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(sample_emb)
    
    # Load genre info for coloring
    genre_table = torch.load(PROCESSED_DIR / "genre_table.pt", weights_only=False)
    movie_genres = genre_table.get("movie_genre_ids", {})
    
    # Get primary genre for each sampled movie
    colors = []
    for idx in indices:
        gids = movie_genres.get(idx, [0])
        colors.append(gids[0])
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="tab20",
                         s=5, alpha=0.6)
    ax.set_title("UMAP of Item Embeddings (colored by primary genre)", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.colorbar(scatter, ax=ax, label="Genre ID")
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "umap_embeddings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved UMAP plot to {RESULTS_DIR / 'umap_embeddings.png'}")


def generate_table_1():
    """Generate Table 1 from ablation results."""
    ablation_path = RESULTS_DIR / "ablation_results.csv"
    if not ablation_path.exists():
        print("No ablation results found. Run training first.")
        return
    
    df = pd.read_csv(ablation_path, index_col=0)
    
    # Select key columns for Table 1
    key_cols = ["RMSE", "MAE"]
    ranking_cols = [c for c in df.columns if any(
        m in c for m in ["NDCG", "MRR", "ILD", "Coverage", "Precision", "Recall"]
    )]
    
    display_cols = [c for c in key_cols + ranking_cols if c in df.columns]
    
    if display_cols:
        table = df[display_cols]
        
        print("\n" + "=" * 80)
        print("TABLE 1: Evaluation Results")
        print("=" * 80)
        print(table.to_string(float_format="%.4f"))
        
        # Save as LaTeX
        latex = table.to_latex(float_format="%.4f", bold_rows=True)
        with open(RESULTS_DIR / "table_1.tex", "w") as f:
            f.write(latex)
        print(f"\nLaTeX table saved to {RESULTS_DIR / 'table_1.tex'}")
    
    return df


def plot_ablation_comparison():
    """Bar chart comparing model variants."""
    ablation_path = RESULTS_DIR / "ablation_results.json"
    if not ablation_path.exists():
        return
    
    with open(ablation_path) as f:
        results = json.load(f)
    
    # RMSE comparison
    models = []
    rmses = []
    for name, metrics in results.items():
        if "RMSE" in metrics:
            models.append(name)
            rmses.append(metrics["RMSE"])
    
    if not models:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(models))
    bars = ax.barh(models, rmses, color=colors)
    
    # Highlight best
    best_idx = np.argmin(rmses)
    bars[best_idx].set_color("gold")
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)
    
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Model Comparison — RMSE", fontsize=14)
    ax.invert_yaxis()
    
    for i, v in enumerate(rmses):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison to {RESULTS_DIR / 'model_comparison.png'}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DataFlix — Evaluation & Visualization")
    print("=" * 60)
    
    plot_training_curves()
    plot_umap_embeddings()
    table = generate_table_1()
    plot_ablation_comparison()
    
    print("\n" + "=" * 60)
    print("ALL EVALUATION COMPLETE")
    print("=" * 60)
    print(f"All outputs in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
