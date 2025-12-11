"""
optim.py: Moteur d'extraction de features basé sur Vision Transformer.
"""
import torch
import numpy as np
from transformers import ViTModel
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, model_name='google/vit-base-patch16-224', device='cuda'):
        """
        Initialise le modèle ViT en mode évaluation.
        """
        self.device = device
        print(f" Chargement du modèle ViT : {model_name}")
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval() # Désactive le mode entraînement (Dropout, BatchNorm, etc.)
        
        # Gel des poids (Freezing) pour garantir qu'ils ne changent pas
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_features(self, dataloader):
        """
        Passe tout le dataset à travers le ViT et retourne les embeddings.
        
        Args:
            dataloader (DataLoader): Le chargeur de données PyTorch.
            
        Returns:
            X (np.array): Features [N_samples, 768]
            y (np.array): Labels [N_samples]
        """
        features_list = []
        labels_list = []
        
        print(f" Extraction des features sur {len(dataloader.dataset)} images...")
        
        # Utilisation de torch.no_grad() pour réduire l'empreinte mémoire
        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader, desc="Extraction"):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Récupération du 'pooler_output' correspondant au token
                # Ce vecteur (768 dimensions pour ViT-Base) résume l'image entière
                batch_features = outputs.pooler_output
                
                # Transfert CPU immédiat pour libérer le GPU
                features_list.append(batch_features.cpu().numpy())
                labels_list.append(labels.numpy())
                
        # Concaténation finale
        X = np.concatenate(features_list, axis=0)
        y = np.concatenate(labels_list, axis=0)
        
        print(f"[INFO] Extraction terminée. Dimensions : {X.shape}")
        return X, y