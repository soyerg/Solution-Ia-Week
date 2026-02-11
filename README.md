# 🏭 Casting Quality Control — IA Week (Groupe 6)

> Système de contrôle qualité par vision artificielle pour pièces de fonderie.  
> Classifie automatiquement les pièces en **Conforme ✅** ou **Défectueuse ❌** grâce à un pipeline **ResNet50 + SVM**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-latest-ee4c2c)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

---

## 📋 Table des matières

- [Présentation](#-présentation)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Prérequis](#-prérequis)
- [Installation & Lancement](#-installation--lancement)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [API Endpoints](#-api-endpoints)
- [Pipeline ML](#-pipeline-ml)
- [Configuration](#-configuration)
- [GPU (optionnel)](#-gpu-optionnel)
- [Dépannage](#-dépannage)
- [Équipe](#-équipe)

---

## 🎯 Présentation

Ce projet a été développé dans le cadre de l'**IA Week**. Il s'agit d'une application web qui simule une **chaîne de production industrielle** avec un convoyeur animé. Les images de pièces de fonderie sont analysées par un modèle d'intelligence artificielle qui détecte automatiquement les défauts de fabrication.

**Le pipeline ML combine :**
1. **ResNet50** (réseau de neurones profond pré-entraîné) pour l'extraction de caractéristiques visuelles
2. **SVM** (Support Vector Machine) pour la classification binaire OK/Défaut

---

## ✨ Fonctionnalités

- 🔍 **Classification automatique** — Détection de défauts sur pièces de fonderie
- 🏭 **Convoyeur animé** — Interface industrielle avec animation GSAP du tri des pièces
- � **Recherche de similarité** — Trouve les 10 images les plus proches dans le dataset pour chaque pièce analysée
- 📊 **Statistiques en temps réel** — Taux de conformité, compteurs, historique
- 🖱️ **Drag & Drop** — Glissez-déposez vos images pour les analyser
- 📋 **File d'attente** — Traitement séquentiel avec suivi visuel
- 🎠 **Carousel interactif** — Navigation horizontale dans les résultats de similarité avec zoom au clic
- 🔒 **Authentification** — Page de connexion sécurisée
- 🐳 **Dockerisé** — Déploiement en un seul commande
- ♻️ **Live Reload** — Modification du code sans rebuild en développement
- 📓 **Notebook d'entraînement** — Encodage du dataset et benchmark des métriques de distance

---

## 🏗️ Architecture

L'application est composée de **2 services Docker** communiquant via un réseau interne :

```
  🌐 Navigateur (port 80)
         │
         ▼
  ┌──────────────────┐          ┌──────────────────────┐
  │     FRONTEND     │  proxy   │       BACKEND        │
  │  FastAPI         │─────────▶│  FastAPI              │
  │  HTML/CSS/JS     │  /api/*  │  PyTorch + SVM       │
  │  (port 3000)     │          │  (port 8000)         │
  │                  │          │                      │
  │  • Convoyeur     │          │  /api/classify       │
  │  • Similarité    │          │  /api/similar        │
  │                  │          │  /api/images/*       │
  └──────────────────┘          └──────────────────────┘
                                    │           │
                             ┌──────┴──┐ ┌──────┴───────┐
                             │ /models │ │/casting_data │
                             │svm_model│ │ (images du   │
                             │ scaler  │ │  dataset)    │
                             │features │ └──────────────┘
                             │ _dataset│
                             │  .npz   │
                             └─────────┘
```

> 📖 Pour plus de détails, voir [architecture.md](architecture.md)

---

## 📦 Prérequis

- [Docker](https://docs.docker.com/get-docker/) (≥ 20.10)
- [Docker Compose](https://docs.docker.com/compose/install/) (≥ 2.0)
- **~3 Go d'espace disque** (images Docker + modèles PyTorch)

> **Note :** Aucune installation Python locale n'est nécessaire, tout tourne dans Docker.

---

## 🚀 Installation & Lancement

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd "Solution Ia Week"
```

### 2. Vérifier les modèles

Assurez-vous que les fichiers de modèles sont présents dans le dossier `models/` :

```
models/
├── resnet50_extractor.pth    # Poids ResNet50 (extraction de features)
├── svm_model.joblib          # Modèle SVM entraîné
├── scaler.joblib             # StandardScaler (normalisation des features)
├── features_dataset.npz      # Vecteurs de features du dataset (généré par ia_training)
└── similarity_config.json    # Configuration de la métrique de distance (généré par ia_training)
```

> ⚠️ Les 3 premiers fichiers sont générés lors de l'entraînement initial.
> Les 2 derniers sont générés par le notebook `ia_training.ipynb` (voir section suivante).

### 3. Générer le dataset de similarité (optionnel mais requis pour `/similarity.html`)

```bash
# Créer un environnement Python 3.12 et exécuter le notebook
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install torch==2.9.0+cpu torchvision==0.24.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn==1.6.1 scipy joblib numpy Pillow matplotlib
jupyter notebook ia_training.ipynb
```

Le notebook va :
1. Encoder toutes les images de `casting_data/` en vecteurs 2048-dim
2. Benchmarker plusieurs métriques de distance (cosinus, euclidienne, Manhattan, etc.)
3. Sauvegarder `models/features_dataset.npz` et `models/similarity_config.json`

### 4. Lancer l'application

```bash
docker compose up --build
```

Au premier lancement, Docker va :
1. Télécharger les images Python (3.12 pour le backend, 3.11 pour le frontend)
2. Installer les dépendances (PyTorch, scikit-learn, etc.)
3. Charger les modèles ML
4. Démarrer les deux serveurs

### 4. Accéder à l'application

Ouvrez votre navigateur sur : **[http://localhost](http://localhost)**

### 5. Arrêter l'application

```bash
docker compose down
```

---

## 🖥️ Utilisation

### Connexion

| Champ         | Valeur           |
|---------------|------------------|
| **Identifiant** | `demo_client`  |
| **Mot de passe** | `iaweekgroup6` |

### Analyser des pièces

1. **Connectez-vous** avec les identifiants ci-dessus
2. **Glissez-déposez** des images de pièces dans la zone de chargement (panneau gauche)
3. **Observez** le convoyeur animé traiter chaque pièce :
   - La pièce arrive sur le tapis roulant
   - La caméra IA scanne la pièce
   - Le résultat s'affiche (Conforme ✅ ou Défectueuse ❌)
   - Le bras de tri envoie la pièce dans le bon bac
4. **Consultez** les statistiques et l'historique en temps réel

### Recherche de similarité

1. **Cliquez** sur **🔍 Similarité** dans le header (ou allez sur `/similarity.html`)
2. **Glissez-déposez** une image dans la zone d'upload
3. **Visualisez** le résultat de classification (OK/DEF + confiance)
4. **Parcourez** le carousel des 10 images les plus similaires du dataset
5. **Cliquez** sur une image du carousel pour l'agrandir

### Formats d'images supportés

- JPEG / JPG
- PNG

---

## 📂 Structure du projet

```
Solution Ia Week/
│
├── docker-compose.yml          # Orchestration des 2 services
├── architecture.md             # Documentation de l'architecture
├── README.md                   # Ce fichier
│
├── backend/                    # Service d'inférence ML
│   ├── Dockerfile              # Image Docker du backend
│   ├── main.py                 # API FastAPI (endpoints /api/*)
│   ├── feature_extractor.py    # Classe ResNet50 feature extractor
│   └── requirements.txt        # Dépendances Python backend
│
├── frontend/                   # Service web (UI + proxy)
│   ├── Dockerfile              # Image Docker du frontend
│   ├── main.py                 # Serveur FastAPI (proxy + static)
│   ├── requirements.txt        # Dépendances Python frontend
│   └── static/                 # Fichiers servis au navigateur
│       ├── index.html          # Page principale (convoyeur)
│       ├── similarity.html     # Page de recherche de similarité
│       ├── login.html          # Page de connexion
│       ├── style.css           # Styles (thème industriel sombre)
│       ├── conveyor.js         # Logique JS convoyeur + animations GSAP
│       └── similarity.js       # Logique JS recherche de similarité
│
├── models/                     # Modèles ML sérialisés
│   ├── resnet50_extractor.pth  # Poids ResNet50 (extraction features)
│   ├── svm_model.joblib        # SVM entraîné (classification)
│   ├── scaler.joblib           # StandardScaler (normalisation)
│   ├── features_dataset.npz   # Vecteurs de features du dataset
│   └── similarity_config.json  # Config métrique de distance
│
├── casting_data/               # Dataset d'images de pièces de fonderie
│   ├── train/
│   │   ├── def_front/          # Images défectueuses (entraînement)
│   │   └── ok_front/           # Images conformes (entraînement)
│   └── test/
│       ├── def_front/          # Images défectueuses (test)
│       └── ok_front/           # Images conformes (test)
│
├── ia_training.ipynb           # Notebook : encodage dataset + benchmark distances
│
└── exemple_dimage/             # Images d'exemple pour tester
```

---

## 🔌 API Endpoints

### `GET /api/health` — État du serveur

**Réponse :**
```json
{
  "status": "ok",
  "device": "cpu",
  "cuda_available": false,
  "svm_loaded": true,
  "scaler_loaded": true
}
```

### `POST /api/classify` — Classifier une image

**Requête :** `multipart/form-data` avec un champ `file` (image)

```bash
curl -X POST http://localhost/api/classify \
  -F "file=@mon_image.jpg"
```

**Réponse :**
```json
{
  "label": "ok",
  "label_fr": "Pièce Conforme ✅",
  "color": "#22c55e",
  "confidence": 0.932,
  "inference_time_ms": 145.3,
  "filename": "mon_image.jpg"
}
```

### `POST /api/similar` — Recherche de similarité

**Requête :** `multipart/form-data` avec un champ `file` (image)

```bash
curl -X POST http://localhost/api/similar \
  -F "file=@mon_image.jpg"
```

**Réponse :**
```json
{
  "label": "def",
  "label_fr": "Pièce Défectueuse ❌",
  "color": "#ef4444",
  "confidence": 0.87,
  "inference_time_ms": 152.4,
  "filename": "mon_image.jpg",
  "metric": "cosine",
  "similar": [
    {
      "rank": 1,
      "path": "test/def_front/cast_def_0_100.jpeg",
      "label": "def",
      "distance": 0.0523,
      "image_url": "/api/images/test/def_front/cast_def_0_100.jpeg"
    }
  ]
}
```

### `GET /api/images/{path}` — Servir une image du dataset

```bash
curl http://localhost/api/images/test/ok_front/cast_ok_0_100.jpeg --output image.jpeg
```

Retourne l'image depuis le dossier `casting_data/`. Protégé contre le path traversal.

| Champ               | Description                                    |
|---------------------|------------------------------------------------|
| `label`             | `"ok"` ou `"def"`                              |
| `label_fr`          | Label en français avec emoji                   |
| `color`             | Code couleur (vert = OK, rouge = défaut)       |
| `confidence`        | Score de confiance entre 0.5 et 1.0            |
| `inference_time_ms` | Temps de traitement en millisecondes            |
| `filename`          | Nom du fichier envoyé                          |
| `metric`            | Métrique de distance utilisée (sur `/api/similar`) |
| `similar`           | Top 10 images les plus proches (sur `/api/similar`) |

---

## 🧠 Pipeline ML

Le pipeline d'inférence suit 5 étapes :

```
Image → Preprocessing → ResNet50 (features 2048-dim) → Scaler → SVM → OK/DEF
```

1. **Preprocessing** — L'image est redimensionnée à 224×224 pixels puis normalisée avec les moyennes/écarts-types d'ImageNet
2. **Feature Extraction** — ResNet50 pré-entraîné (sans la dernière couche) extrait un vecteur de 2048 caractéristiques
3. **Scaling** — Le `StandardScaler` normalise les features (même transformation que l'entraînement)
4. **Classification** — Le SVM prédit la classe : `0` = conforme (ok), `1` = défaut (def)
5. **Confiance** — Calculée via la sigmoïde de la `decision_function` du SVM

### Pourquoi ResNet50 + SVM ?

- **ResNet50** est excellent pour extraire des features visuelles de haut niveau (textures, formes, motifs)
- **SVM** est efficace pour la classification binaire sur des features de haute dimension
- Cette approche est plus **légère à entraîner** qu'un fine-tuning complet du réseau

---

## ⚙️ Configuration

### Variables principales (`backend/main.py`)

| Variable        | Valeur par défaut                | Description                      |
|-----------------|----------------------------------|----------------------------------|
| `MODEL_DIR`     | `/models`                        | Dossier des modèles dans Docker  |
| `IMG_SIZE`      | `224`                            | Taille des images en entrée      |
| `DEVICE`        | Auto (`cuda` si dispo, sinon `cpu`) | Device PyTorch                |
| `CLASSES`       | `["ok", "def"]`                  | Labels (index 0 = ok, 1 = défaut) |

### Proxy frontend (`frontend/main.py`)

| Variable        | Valeur                    | Description                     |
|-----------------|---------------------------|---------------------------------|
| `BACKEND_URL`   | `http://backend:8000`     | URL interne du backend Docker   |

---

## 🖥️ GPU (optionnel)

Pour utiliser un **GPU NVIDIA**, décommentez le bloc dans `docker-compose.yml` :

```yaml
backend:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**Prérequis :**
- Drivers NVIDIA installés sur la machine hôte
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Sans GPU, l'application fonctionne normalement sur **CPU** (inférence un peu plus lente).

---

## 🔧 Dépannage

### Le backend ne démarre pas

```bash
# Vérifier les logs
docker compose logs backend
```

**Causes fréquentes :**
- Fichiers `models/svm_model.joblib` ou `models/scaler.joblib` manquants
- Pas assez de mémoire RAM (PyTorch + ResNet50 nécessitent ~1-2 Go)

### "Backend hors ligne" dans l'interface

- Le backend met quelques secondes à démarrer (chargement de ResNet50)
- Vérifier que les deux conteneurs tournent : `docker compose ps`
- Consulter les logs : `docker compose logs -f`

### Erreur de classification

- Vérifier que l'image est bien au format JPEG ou PNG
- L'image doit représenter une pièce de fonderie (le modèle est spécialisé)

### Rebuild complet

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

---

## 👥 Équipe

**IA Week — Groupe 6**

---

## 📄 Licence

Projet académique — IA Week.
