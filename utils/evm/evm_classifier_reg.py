import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from scipy.stats import weibull_min
import gc
from utils.evm.evm_classifier import EVMClassifier

class RegularizedEVMClassifier(EVMClassifier):
    def __init__(self, lambda_reg=0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_reg = lambda_reg  # Regularization strength

    def fit(self, features: dict):
        classes = list(features.keys())
        for class_name, class_features in features.items():
            arr = np.asarray(class_features, dtype=np.float32)
            self.class_features[class_name] = arr
            self.class_means[class_name] = np.mean(arr, axis=0)

        for target_class in classes:
            target_features = self._prune_features(self.class_features[target_class])
            if len(classes) > 1:
                negatives = [self._prune_features(self.class_features[c]) for c in classes if c != target_class]
                negative_features = np.vstack(negatives)
            else:
                negative_features = target_features

            negative_features = negative_features.astype(np.float32, copy=False)
            weibull_models = []

            for start in range(0, target_features.shape[0], self.batch_size):
                end = min(start + self.batch_size, target_features.shape[0])
                batch_feats = target_features[start:end]

                dists = pairwise_distances(batch_feats, negative_features, metric=self.distance_metric)

                for i in range(dists.shape[0]):
                    point_distances = np.sort(dists[i])
                    t = (max(int(len(point_distances) * self.tailsize), 1)
                         if isinstance(self.tailsize, float) and self.tailsize < 1.0
                         else min(int(self.tailsize), len(point_distances)))
                    tailsize_distances = point_distances[:t]
                    try:
                        # Fit Weibull without location parameter for stability
                        shape, loc, scale = weibull_min.fit(tailsize_distances, floc=0)

                        # Regularize scale parameter (L2 penalty)
                        scale_reg = scale + self.lambda_reg * scale

                        weibull_models.append((batch_feats[i], scale_reg, shape))
                    except Exception:
                        weibull_models.append((batch_feats[i],
                                               np.mean(tailsize_distances),
                                               1.0))
                del dists
                gc.collect()

            self.weibull_models[target_class] = weibull_models

        self.initialized = True
        gc.collect()
        return self
