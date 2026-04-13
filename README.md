# DataFlix: Hybrid Recommendation System for Netflix Ratings

## Overview
DataFlix is a hybrid recommendation system that combines Matrix Factorization, content-based embeddings, and ranking-aware learning to predict user preferences for movies.

Built on the Netflix Movie Rating Dataset, the system addresses key challenges in recommender systems such as data sparsity, cold-start problems, and ranking optimization.

---

## Features

- Matrix Factorization (MF) using ALS and SGD  
- Hybrid model combining collaborative and content-based filtering  
- Semantic movie embeddings using SBERT  
- User behavior modeling from interaction history  
- Ranking optimization with Bayesian Personalized Ranking (BPR)  
- Cold-start handling via metadata and text features  
- Evaluation on multiple metrics (RMSE, NDCG, Precision@K)

---

## Methodology

### Matrix Factorization
We approximate the user-item rating matrix:

M ≈ P Q^T

- P: User latent vectors  
- Q: Movie latent vectors  

Prediction:

r̂_ui = μ + b_u + b_i + p_u^T q_i

---

### Hybrid Embeddings

#### Movie Representation
- Latent MF vector  
- Genre embeddings  
- SBERT plot embeddings  
- Popularity score  

#### User Representation
- Latent MF vector  
- Aggregated embeddings of liked movies  
- Behavioral features (mean rating, variance, activity)

---

### Ranking Optimization

We incorporate Bayesian Personalized Ranking (BPR):

L = -log σ(r̂_ui - r̂_uj)

This improves top-K recommendation quality.

## Zero-Shot Cross-Domain Setup

Based on recent updates, this project now implements an advanced **Zero-Shot Cross-Domain Recommendation** pipeline:
1. **Training Data**: The model is trained exclusively on **MovieLens 25M** users and interactions.
2. **Evaluation Data**: The model is evaluated on the **Netflix Kaggle Dataset**. 
3. **Alignment**: A TF-IDF textual similarity fuzzy-matcher (`src/data/alignment.py`) aligns Netflix movie IDs to MovieLens movie IDs, mapping around ~55% of the Netflix catalog. Unmapped movies are dropped from the evaluation set.
4. **Zero-Shot Cold-Start**: Because the user bases of MovieLens and Netflix are 100% disjoint, every Netflix user during validation is treated as a "cold-start" user. The model must rely completely on SBERT plot embeddings, genres, and metadata to generate recommendations, without access to learned user latent factors.

To run this pipeline:
```bash
python src/data/alignment.py
python scripts/run_preprocess.py
python scripts/run_train.py
python scripts/run_evaluate.py
```
---

## Dataset

- Netflix Movie Rating Dataset  
  - ~17M ratings  
  - ~480K users  
  - ~17K movies  

### Augmentation Sources
- IMDb / TMDb metadata  
- Movie plot summaries  
- MovieLens dataset (for cold-start support)

---

## Tech Stack

- Language: Python  
- Frameworks: PyTorch, NumPy, SciPy  
- NLP: Sentence-BERT  
- Optimization: Optuna  
- Data Handling: Sparse matrices  

---

## Evaluation Metrics

- RMSE, MAE — rating prediction accuracy  
- Precision@K, Recall@K — top-K recommendation quality  
- NDCG@K — ranking quality  
- Coverage — diversity of recommendations  

---

## Experiments

We compare:
- MF only  
- MF + content features  
- MF + BPR  
- Full hybrid model  

---

## Cold-Start Strategy

Handled using:
- Movie metadata embeddings  
- Text-based similarity  
- Cross-dataset enrichment  

---

## Project Structure

root/
│── data/
│── src/
│ ├── models/
│ ├── training/
│ ├── evaluation/
│── notebooks/
│── results/
│── README.md

---

## Authors

- Aryan Kumar  
- Sura Sravan Kumar  
- Shreyash Pandit  

Indian Institute of Technology Gandhinagar

---

## License

This project is for academic and research purposes.
