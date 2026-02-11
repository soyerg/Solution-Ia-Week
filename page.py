# Transforms pour l'entraînement (avec augmentation)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Transforms pour test/inférence (PAS d'augmentation)
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print(" Transforms définis")


class FeatureExtractor(nn.Module):
    """
    Extrait les features d'une image en utilisant un CNN pré-entraîné.
    """

    def __init__(self, model_name='resnet50'):
        super().__init__()

        if model_name == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            # Retirer la dernière couche FC
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.output_dim = 2048

        elif model_name == 'vgg16':
            base_model = models.vgg16(pretrained=True)
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            # Utiliser seulement les features, pas le classifier
            self.output_dim = 512 * 7 * 7  # 25088

        elif model_name == 'densenet121':
            base_model = models.densenet121(pretrained=True)
            self.features = base_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.output_dim = 1024

        elif model_name == 'mobilenet_v2':
            base_model = models.mobilenet_v2(pretrained=True)
            self.features = base_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.output_dim = 1280

        # Geler les poids (pas de fine-tuning)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        if hasattr(self, 'avgpool'):
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x
import time
from datetime import timedelta

def extract_all_features(data_dir, extractor, transform):
    """
    Extrait les features de toutes les images d'un répertoire.

    Returns:
        features: np.array de shape (n_samples, feature_dim)
        labels: np.array de shape (n_samples,)
        paths: liste des chemins d'images
    """
    features_list = []
    labels_list = []
    paths_list = []

    extractor.eval()

    # Timer global
    start_time = time.time()
    total_images = 0

    # Parcourir chaque classe
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = Path(data_dir) / class_name
        class_start = time.time()
        class_count = 0

        print(f"\n Traitement de la classe: {class_name}")

        # Parcourir chaque image dans le répertoire de la classe
        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            try:
                # 1. Charger l'image avec PIL
                img = Image.open(img_path).convert('RGB')

                # 2. Appliquer transform
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)

                # 3. Passer dans extractor
                with torch.no_grad():
                    features = extractor(img_tensor).cpu().numpy()

                # 4. Ajouter aux listes
                features_list.append(features.squeeze())
                labels_list.append(class_idx)
                paths_list.append(str(img_path))

                class_count += 1
                total_images += 1

                # Afficher progression toutes les 100 images
                if total_images % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = total_images / elapsed
                    print(f"  {total_images} images traitées ({speed:.1f} img/s)")

            except Exception as e:
                print(f"   Erreur avec l'image {img_path}: {str(e)}")
                continue

        # Stats par classe
        class_elapsed = time.time() - class_start
        print(f"  {class_name}: {class_count} images en {class_elapsed:.2f}s")

    # Stats finales
    total_time = time.time() - start_time
    avg_time = total_time / total_images if total_images > 0 else 0

    print(f"\n{'='*60}")
    print(f"EXTRACTION TERMINÉE")
    print(f"{'='*60}")
    print(f"  Temps total: {timedelta(seconds=int(total_time))}")
    print(f"Images traitées: {total_images}")
    print(f"Vitesse moyenne: {total_images/total_time:.2f} img/s")
    print(f"Temps moyen/image: {avg_time*1000:.2f} ms")
    print(f"{'='*60}\n")

    return np.array(features_list), np.array(labels_list), paths_list


# Création de l'extracteur
print("Initialisation de l'extracteur...")
extractor_start = time.time()

extractor = FeatureExtractor(model_name='resnet50')
extractor = extractor.to(DEVICE)
extractor.eval()

extractor_time = time.time() - extractor_start
print(f"Extracteur créé en {extractor_time:.2f}s\n")

# Extraction des features
print("="*60)
print("EXTRACTION DES FEATURES - TRAIN SET")
print("="*60)

X_train, y_train, paths_train = extract_all_features(TRAIN_DIR, extractor, test_transform)

print(f"Features train: {X_train.shape}")
print(f"Labels train: {y_train.shape}")
print(f"Nombre de chemins: {len(paths_train)}")


X_test, y_test, paths_test = extract_all_features(TEST_DIR, extractor, test_transform)

# Normalisation des features (IMPORTANT!)
scaler = StandardScaler()

# : Normaliser vos features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Utiliser le même scaler!


# TODO: Entraîner un SVM
def train_svm(X_train, y_train):
    """
    Entraîne un SVM avec un kernel RBF.

    Paramètres :
    - C : pénalisation des erreurs (grand C = moins de régularisation)
    - kernel : type de noyau (RBF est le plus courant)
    - gamma : influence d'un point d'entraînement (auto ou scale recommandé)
    """

    # Création du modèle SVM
    model = SVC(
        C=1.0,              # Paramètre de régularisation
        kernel="rbf",       # Kernel gaussien
        gamma="scale",      # Valeur recommandée par sklearn
        random_state=42
    )

    # Entraînement du modèle
    model.fit(X_train, y_train)

    return model

# TODO: Entraîner un Random Forest
def train_random_forest(X_train, y_train):
    """
    Entraîne un modèle Random Forest.

    Paramètres :
    - n_estimators : nombre d'arbres
    - max_depth : profondeur maximale des arbres
    """

    # Création du modèle Random Forest
    model = RandomForestClassifier(
        n_estimators=100,   # Nombre d'arbres
        max_depth=None,     # Arbres non limités en profondeur
        random_state=42,
        n_jobs=-1           # Utilise tous les cœurs CPU
    )

    # Entraînement du modèle
    model.fit(X_train, y_train)

    return model

# Entraîner le SVM
svm_model = train_svm(X_train_scaled, y_train)

# Entraîner le Random Forest
forest_model = train_random_forest(X_train, y_train)

from joblib import dump

# Sauvegarder les modèles
dump(svm_model, 'svm_model.joblib')
dump(forest_model, 'random_forest_model.joblib')

print("Modèles sauvegardés avec succès !")