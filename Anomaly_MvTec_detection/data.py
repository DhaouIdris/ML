"""
data.py: Gestion du téléchargement et du chargement des données MVTec AD.
"""
import os
import glob
import zipfile
from PIL import Image
from torch.utils.data import Dataset #, DataLoader

def download_mvtec_kaggle(destination_folder='mvtec_data'):
    """
    Télécharge le dataset MVTec AD depuis Kaggle via API.
    Nécessite le fichier 'kaggle.json' à la racine ou dans ~/.kaggle.
    """
    # Configuration des credentials Kaggle
    if not os.path.exists('/root/.kaggle'):
        os.makedirs('/root/.kaggle')
        # On suppose que l'utilisateur a uploadé kaggle.json dans Colab
        if os.path.exists('kaggle.json'):
            os.system('cp kaggle.json /root/.kaggle/')
            os.system('chmod 600 /root/.kaggle/kaggle.json')
        else:
            print(" Fichier kaggle.json introuvable. Assurez-vous de l'avoir uploadé.")
            return None

    if not os.path.exists(destination_folder):
        print(f"[INFO] Téléchargement de MVTec AD dans {destination_folder}...")
        # Commande API Kaggle pour télécharger le dataset compressé
        # Le dataset ipythonx/mvtec-ad est un miroir fiable
        exit_code = os.system('kaggle datasets download -d ipythonx/mvtec-ad')
        
        if exit_code!= 0:
            raise RuntimeError("Échec du téléchargement Kaggle.")

        print("[INFO] Décompression...")
        with zipfile.ZipFile('mvtec-ad.zip', 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        
        # Nettoyage
        os.remove('mvtec-ad.zip')
    else:
        print("[INFO] Données déjà présentes.")
        
    return destination_folder

class MVTecLightDataset(Dataset):
    def __init__(self, root_dir, category='bottle', split='pool', transform=None):
        """
        Dataset personnalisé pour charger une classe spécifique de MVTec.
        
        Args:
            root_dir (str): Chemin racine des données extraites.
            category (str): Classe d'objet (ex: 'bottle', 'wood', 'transistor').
            split (str): 'train' (bonnes images seulement) ou 'pool' (test set mixte).
            transform (callable): ViTImageProcessor de Hugging Face.
        """
        self.root_dir = root_dir
        self.category = category
        self.transform = transform
        self.image_paths = []
        self.labels = [] # 0: Normal, 1: Anomalie

        # Construction des chemins
        # MVTec structure: <root>/<category>/<split>/<defect_type>/<image.png>
        base_path = os.path.join(root_dir, category)
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Catégorie {category} introuvable dans {root_dir}")

        if split == 'train':
            # Chargement des données d'entraînement nominales (Good only)
            # Utile si on veut faire du 'Warm Start' avec des exemples sains
            good_path = os.path.join(base_path, 'train', 'good')
            self._load_folder(good_path, label=0)
            
        elif split == 'pool':
            # Chargement du set de test complet pour simuler le pool non étiqueté
            # Ce set contient des images 'good' et divers types de défauts
            test_path = os.path.join(base_path, 'test')
            
            # Itération sur tous les sous-dossiers (good, broken, contamination, etc.)
            if os.path.exists(test_path):
                for subfolder in os.listdir(test_path):
                    full_sub_path = os.path.join(test_path, subfolder)
                    if os.path.isdir(full_sub_path):
                        # Si le dossier est 'good', label 0, sinon 1
                        lbl = 0 if subfolder == 'good' else 1
                        self._load_folder(full_sub_path, label=lbl)
            else:
                raise FileNotFoundError(f"Dossier test introuvable pour {category}")

        print(f" Chargé {len(self.image_paths)} images pour la catégorie '{category}' (Split: {split})")

    def _load_folder(self, folder_path, label):
        # Support des extensions courantes dans MVTec (généralement.png)
        patterns = ['*.png', '*.jpg', '*.jpeg']
        for pat in patterns:
            files = glob.glob(os.path.join(folder_path, pat))
            self.image_paths.extend(files)
            self.labels.extend([label] * len(files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Chargement PIL en RGB (MVTec est parfois en Niveaux de gris, conversion nécessaire pour ViT)
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            # Le processeur HF retourne un dictionnaire {'pixel_values': tensor}
            # return_tensors="pt" donne un batch de taille 1 
            # On utilise squeeze(0) pour obtenir 
            encoded = self.transform(images=image, return_tensors="pt")
            image = encoded['pixel_values'].squeeze(0)

        # On retourne aussi l'index pour le suivi dans l'Active Learning
        return image, label, idx