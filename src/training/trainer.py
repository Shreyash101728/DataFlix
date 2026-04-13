"""
DataFlix — Training Module
Path A (MSE loss) and Path B (BPR loss) trainers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import json

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE, LR_PATH_A, LR_PATH_B, WEIGHT_DECAY, COSINE_T_MAX,
    EARLY_STOP_PATIENCE, MAX_EPOCHS, BATCH_SIZE, BPR_REG,
    BPR_EPOCHS, BPR_BATCH_SIZE, RESULTS_DIR, SEED
)


# ──────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────

class RatingDataset(Dataset):
    """Dataset for (user, item, rating) triplets."""
    
    def __init__(self, df: pd.DataFrame, sbert_embeddings: torch.Tensor,
                 popularity: torch.Tensor, user_features: torch.Tensor,
                 history_embeddings: torch.Tensor, genre_table: dict):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)
        
        self.sbert = sbert_embeddings
        self.popularity = popularity
        self.user_features = user_features
        self.history = history_embeddings
        self.genre_ids = genre_table.get("movie_genre_ids", {})
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        uid = self.users[idx]
        iid = self.items[idx]
        rating = self.ratings[idx]
        
        sbert_emb = self.sbert[iid] if iid < len(self.sbert) else torch.zeros(self.sbert.shape[1])
        pop = self.popularity[iid].unsqueeze(0) if iid < len(self.popularity) else torch.zeros(1)
        user_feat = self.user_features[uid] if uid < len(self.user_features) else torch.zeros(4)
        hist_emb = self.history[uid] if uid < len(self.history) else torch.zeros(self.sbert.shape[1])
        genre_ids = self.genre_ids.get(int(iid), [0])
        
        return uid, iid, rating, sbert_emb, pop, user_feat, hist_emb, genre_ids


class BPRDataset(Dataset):
    """Dataset for BPR triplets: (user, pos_item, neg_item)."""
    
    def __init__(self, user_positives: dict, all_items: np.ndarray,
                 item_pop: np.ndarray, sbert_embeddings: torch.Tensor,
                 popularity: torch.Tensor, user_features: torch.Tensor,
                 history_embeddings: torch.Tensor, genre_table: dict,
                 n_samples_per_epoch: int = 500000):
        self.user_positives = user_positives
        self.users = list(user_positives.keys())
        self.all_items = all_items
        self.item_pop = item_pop / item_pop.sum()  # Normalise
        self.n_samples = n_samples_per_epoch
        
        self.sbert = sbert_embeddings
        self.popularity = popularity
        self.user_features = user_features
        self.history = history_embeddings
        self.genre_ids = genre_table.get("movie_genre_ids", {})
        
        self._generate_samples()
    
    def _generate_samples(self):
        """Generate BPR training triplets with popularity-proportional negatives."""
        self.samples = []
        rng = np.random.RandomState(SEED)
        
        for _ in range(self.n_samples):
            # Sample a random user with positives
            u = rng.choice(self.users)
            pos_items = list(self.user_positives[u])
            pos = rng.choice(pos_items)
            
            # Popularity-proportional negative sampling (harder negatives)
            neg = rng.choice(self.all_items, p=self.item_pop)
            while neg in self.user_positives[u]:
                neg = rng.choice(self.all_items, p=self.item_pop)
            
            self.samples.append((u, pos, neg))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        u, pos, neg = self.samples[idx]
        
        sbert_pos = self.sbert[pos] if pos < len(self.sbert) else torch.zeros(self.sbert.shape[1])
        sbert_neg = self.sbert[neg] if neg < len(self.sbert) else torch.zeros(self.sbert.shape[1])
        pop_pos = self.popularity[pos].unsqueeze(0) if pos < len(self.popularity) else torch.zeros(1)
        pop_neg = self.popularity[neg].unsqueeze(0) if neg < len(self.popularity) else torch.zeros(1)
        user_feat = self.user_features[u] if u < len(self.user_features) else torch.zeros(4)
        hist_emb = self.history[u] if u < len(self.history) else torch.zeros(self.sbert.shape[1])
        genre_pos = self.genre_ids.get(int(pos), [0])
        genre_neg = self.genre_ids.get(int(neg), [0])
        
        return (torch.tensor(u, dtype=torch.long),
                torch.tensor(pos, dtype=torch.long),
                torch.tensor(neg, dtype=torch.long),
                sbert_pos, sbert_neg, pop_pos, pop_neg,
                user_feat, hist_emb, genre_pos, genre_neg)


def collate_rating(batch):
    """Custom collate for RatingDataset (handles variable-length genre IDs)."""
    uids, iids, ratings, sberts, pops, ufeats, hists, genres = zip(*batch)
    return (
        torch.stack(uids), torch.stack(iids), torch.stack(ratings),
        torch.stack(sberts), torch.stack(pops), torch.stack(ufeats),
        torch.stack(hists), list(genres)
    )


def collate_bpr(batch):
    """Custom collate for BPRDataset."""
    (uids, pos_ids, neg_ids, sbert_pos, sbert_neg,
     pop_pos, pop_neg, ufeats, hists, genre_pos, genre_neg) = zip(*batch)
    return (
        torch.stack(uids), torch.stack(pos_ids), torch.stack(neg_ids),
        torch.stack(sbert_pos), torch.stack(sbert_neg),
        torch.stack(pop_pos), torch.stack(pop_neg),
        torch.stack(ufeats), torch.stack(hists),
        list(genre_pos), list(genre_neg)
    )


# ──────────────────────────────────────────────
# Path A Trainer (MSE)
# ──────────────────────────────────────────────

class PathATrainer:
    """
    Train DataFlix model with MSE loss for rating prediction.
    Uses Adam with cosine annealing and early stopping.
    """
    
    def __init__(self, model, lr=LR_PATH_A, weight_decay=WEIGHT_DECAY,
                 t_max=COSINE_T_MAX, patience=EARLY_STOP_PATIENCE,
                 max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE):
        self.model = model.to(DEVICE)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)
        self.patience = patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        
        self.history = {"train_loss": [], "val_loss": [], "val_rmse": [], "lr": []}
    
    def train(self, train_dataset, val_dataset):
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_rating, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size * 2, shuffle=False,
            collate_fn=collate_rating, num_workers=0
        )
        
        best_val_rmse = float("inf")
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.max_epochs):
            t0 = time.time()
            
            # Train
            self.model.train()
            total_loss = 0
            n_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                uids, iids, ratings, sberts, pops, ufeats, hists, genres = batch
                uids = uids.to(DEVICE)
                iids = iids.to(DEVICE)
                ratings = ratings.to(DEVICE)
                sberts = sberts.to(DEVICE)
                pops = pops.to(DEVICE)
                ufeats = ufeats.to(DEVICE)
                hists = hists.to(DEVICE)
                
                preds = self.model(uids, iids, sberts, pops, genres, hists, ufeats)
                loss = self.criterion(preds, ratings)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            self.scheduler.step()
            avg_train_loss = total_loss / max(n_batches, 1)
            
            # Validate
            val_rmse, val_loss = self._evaluate(val_loader)
            
            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_rmse"].append(val_rmse)
            self.history["lr"].append(lr)
            
            print(f"Epoch {epoch+1}/{self.max_epochs}: "
                  f"train_loss={avg_train_loss:.4f}, val_rmse={val_rmse:.4f}, "
                  f"lr={lr:.6f}, time={elapsed:.1f}s")
            
            # Early stopping
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1} (best val RMSE: {best_val_rmse:.4f})")
                    break
        
        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        return self.history
    
    def _evaluate(self, loader):
        self.model.eval()
        total_se = 0
        total_loss = 0
        n = 0
        
        with torch.no_grad():
            for batch in loader:
                uids, iids, ratings, sberts, pops, ufeats, hists, genres = batch
                uids = uids.to(DEVICE)
                iids = iids.to(DEVICE)
                ratings = ratings.to(DEVICE)
                sberts = sberts.to(DEVICE)
                pops = pops.to(DEVICE)
                ufeats = ufeats.to(DEVICE)
                hists = hists.to(DEVICE)
                
                preds = self.model(uids, iids, sberts, pops, genres, hists, ufeats)
                loss = self.criterion(preds, ratings)
                
                total_se += ((preds - ratings) ** 2).sum().item()
                total_loss += loss.item() * len(ratings)
                n += len(ratings)
        
        rmse = np.sqrt(total_se / max(n, 1))
        avg_loss = total_loss / max(n, 1)
        return rmse, avg_loss


# ──────────────────────────────────────────────
# Path B Trainer (BPR)
# ──────────────────────────────────────────────

class PathBTrainer:
    """
    Train DataFlix model with BPR loss for ranking optimisation.
    L_BPR = -Σ ln σ(r̂_ui - r̂_uj) + λ(||p_u||² + ||q_i||² + ||q_j||²)
    """
    
    def __init__(self, model, lr=LR_PATH_B, reg=BPR_REG,
                 max_epochs=BPR_EPOCHS, batch_size=BPR_BATCH_SIZE):
        self.model = model.to(DEVICE)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        
        self.history = {"train_loss": []}
    
    def bpr_loss(self, score_pos, score_neg):
        """BPR loss: -ln σ(score_pos - score_neg)"""
        return -torch.log(torch.sigmoid(score_pos - score_neg) + 1e-8).mean()
    
    def train(self, bpr_dataset):
        loader = DataLoader(
            bpr_dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_bpr, num_workers=0, pin_memory=True
        )
        
        for epoch in range(self.max_epochs):
            t0 = time.time()
            self.model.train()
            total_loss = 0
            n_batches = 0
            
            for batch in tqdm(loader, desc=f"BPR Epoch {epoch+1}", leave=False):
                (uids, pos_ids, neg_ids, sbert_pos, sbert_neg,
                 pop_pos, pop_neg, ufeats, hists, genre_pos, genre_neg) = batch
                
                uids = uids.to(DEVICE)
                pos_ids = pos_ids.to(DEVICE)
                neg_ids = neg_ids.to(DEVICE)
                sbert_pos = sbert_pos.to(DEVICE)
                sbert_neg = sbert_neg.to(DEVICE)
                pop_pos = pop_pos.to(DEVICE)
                pop_neg = pop_neg.to(DEVICE)
                ufeats = ufeats.to(DEVICE)
                hists = hists.to(DEVICE)
                
                score_pos, score_neg = self.model.predict_pair_scores(
                    uids, pos_ids, neg_ids,
                    sbert_pos, sbert_neg, pop_pos, pop_neg,
                    genre_pos, genre_neg, hists, ufeats
                )
                
                loss = self.bpr_loss(score_pos, score_neg)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            self.history["train_loss"].append(avg_loss)
            elapsed = time.time() - t0
            
            print(f"BPR Epoch {epoch+1}/{self.max_epochs}: "
                  f"loss={avg_loss:.4f}, time={elapsed:.1f}s")
        
        return self.history


# ──────────────────────────────────────────────
# Simple MF Trainer (for baselines)
# ──────────────────────────────────────────────

class SimpleMFTrainer:
    """Trainer for simple PyTorch MF models (VanillaMF, NeuMF)."""
    
    def __init__(self, model, lr=5e-3, weight_decay=1e-4,
                 max_epochs=50, batch_size=4096, patience=5):
        self.model = model.to(DEVICE)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
    
    def train(self, train_df, val_df):
        train_users = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
        train_items = torch.tensor(train_df["movie_idx"].values, dtype=torch.long)
        train_ratings = torch.tensor(train_df["rating"].values, dtype=torch.float32)
        
        val_users = torch.tensor(val_df["user_idx"].values, dtype=torch.long)
        val_items = torch.tensor(val_df["movie_idx"].values, dtype=torch.long)
        val_ratings = torch.tensor(val_df["rating"].values, dtype=torch.float32)
        
        n = len(train_users)
        best_rmse = float("inf")
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.model.train()
            
            # Shuffle
            perm = torch.randperm(n)
            total_loss = 0
            n_batches = 0
            
            for i in range(0, n, self.batch_size):
                idx = perm[i:i+self.batch_size]
                u = train_users[idx].to(DEVICE)
                it = train_items[idx].to(DEVICE)
                r = train_ratings[idx].to(DEVICE)
                
                pred = self.model(u, it)
                loss = self.criterion(pred, r)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            self.scheduler.step()
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(val_users.to(DEVICE), val_items.to(DEVICE))
                val_rmse = torch.sqrt(((val_pred - val_ratings.to(DEVICE)) ** 2).mean()).item()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: train_loss={total_loss/n_batches:.4f}, "
                      f"val_rmse={val_rmse:.4f}")
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        print(f"  Best val RMSE: {best_rmse:.4f}")
        return best_rmse
