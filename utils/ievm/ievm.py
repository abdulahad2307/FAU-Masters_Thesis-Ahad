import torch
import numpy as np
from scipy.stats import weibull_min
from sklearn.metrics import pairwise_distances

class IncrementalEVM:
    """
    Incremental Extreme Value Machine (iEVM)
    PyTorch version based on the TensorFlow-iEVM.
    """

    def __init__(self, tailsize=0.3, ev_budget=10, cover_threshold=0.7, distance_metric="euclidean"):
        self.tailsize = tailsize
        self.ev_budget = ev_budget
        self.cover_threshold = cover_threshold
        self.distance_metric = distance_metric
        self.class_evs = {}    
        self.class_features = {}
        self.initialized = False

    ###################### UTILITY FUNCTIONS ######################

    @staticmethod
    def np_pairwise_distances(A, B):
        return pairwise_distances(A, B)

    @staticmethod
    def torch_pairwise_distances(A, B):
        if not torch.is_tensor(A):
            A = torch.tensor(A, dtype=torch.float32)
        if not torch.is_tensor(B):
            B = torch.tensor(B, dtype=torch.float32)
        return torch.cdist(A, B)

    
    ################### GREEDY K-COVER ###################

    def greedy_k_set_cover(self, evs, k):
        """Greedily select k representative extremes using coverage scores."""
        if len(evs) <= k:
            return evs
        mat = np.array([ev[0] for ev in evs])
        incl_probs = np.zeros((len(evs), len(evs)))
        for i, ev in enumerate(evs):
            dists = np.linalg.norm(mat - ev[0], axis=1)
            incl_probs[i] = 1 - np.exp(-((dists / (ev[1] + 1e-8)) ** (ev[2] + 1e-8)))  # Avoid division by zero
        cover_scores = incl_probs.sum(axis=1)
        idxs = np.argsort(-cover_scores)[:k]
        return [evs[i] for i in idxs]

    
    #################### WEIBULL FITTING ####################

    def fit_weibull(self, point, negatives):
        dists = pairwise_distances([point], negatives, metric=self.distance_metric).flatten()
        t = max(int(len(dists) * self.tailsize), 1) if self.tailsize < 1.0 else min(int(self.tailsize), len(dists))
        tails = np.sort(dists)[:t]
        try:
            shape, loc, scale = weibull_min.fit(tails, floc=0)
        except Exception:
            shape, scale = 1., np.mean(tails)
        max_tail = np.max(tails)
        return scale, shape, max_tail


    def fit(self, features_dict):
        """Fit all classes from scratch."""
        self.class_features = {}
        for label, feats in features_dict.items():
            feats = np.array(feats).astype(np.float32)
            self.class_features[label] = feats
        for label in self.class_features:
            feats = self.class_features[label]
            negatives = np.vstack([self.class_features[lab] for lab in self.class_features if lab != label]) if len(self.class_features) > 1 else feats
            evs = []
            for point in feats:
                scale, shape, max_tail = self.fit_weibull(point, negatives)
                evs.append((point, scale, shape, max_tail))
            self.class_evs[label] = self.greedy_k_set_cover(evs, self.ev_budget)
        self.initialized = True
        return self

    def incremental_update(self, new_features_dict):
        """Incrementally update (add new classes or add samples to existing) and refit efficiently."""
        for label, feats in new_features_dict.items():
            feats = np.array(feats).astype(np.float32)
            # Updating local database
            if label in self.class_features:
                self.class_features[label] = np.vstack((self.class_features[label], feats))
            else:
                self.class_features[label] = feats
            negatives = np.vstack([
                self.class_features[lab] for lab in self.class_features if lab != label
            ]) if len(self.class_features) > 1 else self.class_features[label]
            # Re-fitting only EVs within max_tail, plus any new samples
            new_evs = []
            existing_evs = self.class_evs.get(label, [])
            for ev in existing_evs:
                d_to_new = pairwise_distances([ev[0]], feats, metric=self.distance_metric).flatten()
                if np.any(d_to_new < ev[3]):
                    scale, shape, max_tail = self.fit_weibull(ev[0], negatives)
                    new_evs.append((ev[0], scale, shape, max_tail))
                else:
                    new_evs.append(ev)
            # Addding new EVs for new points:
            for point in feats:
                scale, shape, max_tail = self.fit_weibull(point, negatives)
                new_evs.append((point, scale, shape, max_tail))
            self.class_evs[label] = self.greedy_k_set_cover(new_evs, self.ev_budget)
        self.initialized = True
        return self

    
    ################### PREDICTION API ###################

    def predict_proba(self, features):
        features = np.array(features, dtype=np.float32)
        results = {}
        for label, evs in self.class_evs.items():
            scores = np.zeros(features.shape[0], dtype=np.float32)
            for ev in evs:
                d = pairwise_distances(features, ev[0][None, :], metric=self.distance_metric).flatten()
                prob = 1 - np.exp(-((d / (ev[1] + 1e-8)) ** (ev[2] + 1e-8)))
                scores = np.maximum(scores, prob)
            results[label] = 1 - scores  #higher value for closer samples
        return results

    def predict_proba_tensor(self, features):
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)
        n_samples = features.size(0)
        class_names = list(self.class_evs.keys())
        n_classes = len(class_names)
        prob_mat = torch.zeros((n_samples, n_classes), dtype=torch.float32)
        for cidx, cname in enumerate(class_names):
            evs = self.class_evs[cname]
            scores = torch.zeros(n_samples, dtype=torch.float32)
            for ev in evs:
                point = torch.tensor(ev[0], dtype=torch.float32, device=features.device)
                d = torch.cdist(features, point.unsqueeze(0)).squeeze(1)
                prob = 1 - torch.exp(-((d / (ev[1] + 1e-8)) ** (ev[2] + 1e-8)))
                scores = torch.max(scores, prob)
            prob_mat[:, cidx] = 1 - scores
        return prob_mat.cpu()

    def predict(self, features, threshold=None):
        if threshold is None:
            threshold = self.cover_threshold
        probs = self.predict_proba(features)
        class_names = list(probs.keys())
        prob_matrix = np.column_stack([probs[c] for c in class_names])
        max_probs = np.max(prob_matrix, axis=1)
        max_indices = np.argmax(prob_matrix, axis=1)
        labels = [class_names[idx] if p >= threshold else "unknown"
                  for p, idx in zip(max_probs, max_indices)]
        return labels, max_probs

    def state_dict(self):
        return {
            "tailsize": self.tailsize,
            "ev_budget": self.ev_budget,
            "cover_threshold": self.cover_threshold,
            "distance_metric": self.distance_metric,
            "class_evs": self.class_evs,
            "class_features": self.class_features,
        }

    def load_state_dict(self, state):
        self.tailsize = state["tailsize"]
        self.ev_budget = state["ev_budget"]
        self.cover_threshold = state["cover_threshold"]
        self.distance_metric = state["distance_metric"]
        self.class_evs = state["class_evs"]
        self.class_features = state["class_features"]
        self.initialized = True
