"""
utils.py: Utilitaires pour la configuration système et la visualisation.
"""
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def set_seed(seed=42):
    """
    Fixe la graine aléatoire pour assurer la reproductibilité totale des expériences.
    Impacte Python, Numpy et PyTorch (CPU & GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Pour une déterministé absolue (peut ralentir l'exécution)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed fixée à : {seed}")

def get_device():
    """Détecte et retourne le périphérique de calcul optimal."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Utilisation du device : {device}")
    return device

def visualize_features(features, labels, title="Distribution des Features (t-SNE)"):
    """
    Réduit la dimensionnalité des embeddings ViT (768-d) vers 2D via t-SNE pour visualisation.
    Args:
        features (np.array): Matrice [N, 768] des features extraites.
        labels (np.array): Vecteur [N] des labels (0=Good, 1=Defect).
    """
    print("[INFO] Calcul du t-SNE en cours...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 8))
    # Utilisation de Seaborn pour une esthétique professionnelle
    sns.scatterplot(
        x=reduced[:, 0], 
        y=reduced[:, 1], 
        hue=labels, 
        palette={0: 'green', 1: 'red'},
        style=labels,
        alpha=0.7,
        s=60
    )
    plt.title(title, fontsize=15)
    plt.xlabel("Dimension t-SNE 1")
    plt.ylabel("Dimension t-SNE 2")
    plt.legend(title='Classe', labels= ['Good', 'Defect'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()