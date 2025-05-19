import numpy as np
from scipy.stats import weibull_min
from sklearn.metrics import pairwise_distances

class EVMDomainClassifier:
    """Extreme Value Machine for domain novelty detection"""
    def __init__(self, tailsize=0.5, threshold=0.7):
        self.tailsize = tailsize
        self.threshold = threshold
        self.weibull_models = {}  # domain -> list of (mean, scale, shape)
        self.domain_features = {}  # domain -> list of feature vectors
        self.initialized = False
        
    def fit(self, features):
        """Fit EVM to feature vectors for each domain"""
        domains = list(features.keys())
        
        # Store domain features
        for domain, domain_features in features.items():
            self.domain_features[domain] = domain_features
        
        # Fit Weibull models for each domain
        for target_domain in domains:
            target_features = features[target_domain]
            
            # Combine features from all other domains as negative examples
            negative_features = []
            for d in domains:
                if d != target_domain:
                    negative_features.append(features[d])
                    
            if negative_features:
                negative_features = np.vstack(negative_features)
                
                # Compute distances between target domain and negative examples
                distances = pairwise_distances(
                    target_features, 
                    negative_features, 
                    metric='euclidean'
                )
                
                # Fit Weibull distribution for each point in target domain
                weibull_models = []
                for i, point_distances in enumerate(distances):
                    # Sort distances in ascending order
                    point_distances = np.sort(point_distances)
                    
                    # Determine tailsize
                    if isinstance(self.tailsize, float) and self.tailsize < 1.0:
                        t = max(int(len(point_distances) * self.tailsize), 1)
                    else:
                        t = min(int(self.tailsize), len(point_distances))
                    
                    # Use only the smallest t distances for fitting
                    tailsize_distances = point_distances[:t]
                    
                    # Fit Weibull distribution
                    try:
                        shape, loc, scale = weibull_min.fit(tailsize_distances, floc=0)
                        weibull_models.append((target_features[i], scale, shape))
                    except:
                        # If fitting fails, use default parameters
                        weibull_models.append((target_features[i], np.mean(tailsize_distances), 1.0))
                
                self.weibull_models[target_domain] = weibull_models
            else:
                # If no negative examples, use default models
                self.weibull_models[target_domain] = [(f, 1.0, 1.0) for f in target_features[:5]]
                
        self.initialized = True
        return self
    
    def predict_proba(self, features):
        """Predict domain probabilities for input features"""
        if not self.initialized:
            raise ValueError("EVM model not initialized. Call fit() first.")
        
        # Initialize probabilities
        probabilities = {}
        
        # Compute probabilities for each domain
        for domain, weibull_models in self.weibull_models.items():
            domain_probs = np.zeros(len(features))
            
            for point, scale, shape in weibull_models:
                # Compute distances to the current point
                distances = pairwise_distances(
                    features, 
                    point.reshape(1, -1), 
                    metric='euclidean'
                ).flatten()
                
                # Compute probabilities using the Weibull CDF
                point_probs = 1 - np.exp(-((distances / scale) ** shape))
                
                # Update domain probabilities (take maximum probability)
                domain_probs = np.maximum(domain_probs, point_probs)
            
            probabilities[domain] = 1 - domain_probs
        
        return probabilities
    
    def predict(self, features, return_probas=False):
        """Predict domain labels for input features"""
        probabilities = self.predict_proba(features)
        
        # Get domain with highest probability for each sample
        domain_names = list(probabilities.keys())
        prob_matrix = np.column_stack([probabilities[d] for d in domain_names])
        
        # Get max probability and corresponding domain
        max_probs = np.max(prob_matrix, axis=1)
        max_indices = np.argmax(prob_matrix, axis=1)
        
        # Assign labels based on threshold
        labels = []
        for i, (prob, idx) in enumerate(zip(max_probs, max_indices)):
            if prob >= self.threshold:
                labels.append(domain_names[idx])
            else:
                labels.append("unknown")
        
        if return_probas:
            return labels, prob_matrix
        return labels
    
    def incremental_update(self, new_features):
        """Update the model with new domains"""
        # Merge new features with existing ones
        all_features = self.domain_features.copy()
        for domain, features in new_features.items():
            if domain in all_features:
                all_features[domain] = np.vstack([all_features[domain], features])
            else:
                all_features[domain] = features
        
        # Refit the model with all features
        model = self.fit(all_features)
        return model
