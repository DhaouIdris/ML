"""
active_learning.py: Stratégies de sélection d'échantillons (Query Strategies).
"""
import numpy as np
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

class ActiveLearningAgent:
    def __init__(self, X_pool, y_pool, seed=42):
        """
        Agent gérant le cycle d'apprentissage actif.
        
        Args:
            X_pool (np.array): L'ensemble complet des features disponibles.
            y_pool (np.array): Les labels correspondants (L'Oracle).
            seed (int): Graine pour le Random Forest.
        """
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.n_samples = X_pool.shape
        
        # Gestion des indices
        self.labeled_indices = set()
        self.unlabeled_indices = set(range(self.n_samples))
        
        # Modèle : Random Forest
        # n_estimators=100 fournit une bonne granularité pour l'estimation de probabilité
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=seed, 
            n_jobs=-1, # Parallélisation CPU
            class_weight='balanced' # Crucial pour MVTec (très déséquilibré)
        )

    def initial_seed(self, n_samples=10):
        """Sélection aléatoire initiale pour démarrer la boucle (Cold Start)."""
        selected = np.random.choice(list(self.unlabeled_indices), n_samples, replace=False)
        self.update(selected)

    def update(self, new_indices):
        """Déplace les indices du pool non étiqueté vers le pool étiqueté."""
        for idx in new_indices:
            self.unlabeled_indices.remove(idx)
            self.labeled_indices.add(idx)
        
        # Ré-entraînement immédiat du modèle
        self._train()

    def _train(self):
        """Entraîne le modèle sur les données actuellement étiquetées."""
        idxs = list(self.labeled_indices)
        X_train = self.X_pool[idxs]
        y_train = self.y_pool[idxs]
        self.model.fit(X_train, y_train)

    def evaluate(self):
        """Évalue le modèle sur le reste du pool non étiqueté (Simulation Validation)."""
        # Dans un vrai scénario, on aurait un set de validation séparé.
        # Ici, on utilise les données non vues pour estimer la performance de généralisation.
        if not self.unlabeled_indices:
            return 1.0
            
        idxs = list(self.unlabeled_indices)
        X_val = self.X_pool[idxs]
        y_val = self.y_pool[idxs]
        
        preds = self.model.predict(X_val)
        return accuracy_score(y_val, preds)

    def query(self, n_instances=5, strategy='uncertainty'):
        """Sélectionne les n_instances les plus informatives selon la stratégie."""
        unlabeled_list = list(self.unlabeled_indices)
        X_unlabeled = self.X_pool[unlabeled_list]
        
        if strategy == 'random':
            selected_indices = np.random.choice(unlabeled_list, n_instances, replace=False)
            return selected_indices
            
        elif strategy == 'uncertainty':
            # Uncertainty Sampling (Max Entropy)
            probs = self.model.predict_proba(X_unlabeled)
            # Calcul de l'entropie sur l'axe des classes (axis=1)
            # entropy() de scipy calcule -sum(p * log(p))
            entropies = entropy(probs.T)
            
            # On veut les indices avec l'entropie la plus ÉLEVÉE
            # argsort trie par ordre croissant, donc on prend les derniers [-n:]
            top_n_local_indices = np.argsort(entropies)[-n_instances:]
            return [unlabeled_list[i] for i in top_n_local_indices]
            
        elif strategy == 'bald':
            scores = self._calculate_bald_scores(X_unlabeled)
            top_n_local_indices = np.argsort(scores)[-n_instances:]
            return [unlabeled_list[i] for i in top_n_local_indices]
        
        else:
            raise ValueError(f"Stratégie inconnue : {strategy}")

    def _calculate_bald_scores(self, X):
        """
        Implémentation de BALD pour Random Forest.
        BALD = H(Mean(P)) - Mean(H(P))
        """
        # Collecter les probabilités de tous les arbres
        # tree_probs shape : [n_estimators, n_samples, n_classes]
        tree_probs = []
        for tree in self.model.estimators_:
            # Précaution : gestion des arbres qui n'ont vu qu'une seule classe
            p = tree.predict_proba(X)
            
            # Si l'arbre n'a vu que la classe 0, il retourne shape (N, 1)
            # Il faut le transformer en (N, 2) [1.0, 0.0] ou [0.0, 1.0]
            if p.shape < 2:
                # Identification de la classe connue par l'arbre
                # tree.classes_ contient les labels connus
                full_p = np.zeros((X.shape, 2))
                class_idx = int(tree.classes_) # 0 ou 1
                full_p[:, class_idx] = 1.0
                tree_probs.append(full_p)
            else:
                tree_probs.append(p)
                
        tree_probs = np.array(tree_probs)
        
        # 1. Calcul de la prédiction moyenne (Consensus)
        avg_probs = np.mean(tree_probs, axis=0)
        
        # 2. Entropie du consensus (Incertitude Totale)
        entropy_mean = entropy(avg_probs.T)
        
        # 3. Moyenne des entropies individuelles (Incertitude Aléatorique)
        entropies_individual = entropy(tree_probs, axis=2) # [n_trees, n_samples]
        mean_entropy = np.mean(entropies_individual, axis=0)
        
        # 4. Score BALD (Incertitude Épistémique - Information Mutuelle)
        bald_scores = entropy_mean - mean_entropy
        
        return bald_scores