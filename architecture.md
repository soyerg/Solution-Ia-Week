# 🏗️ Architecture du Projet — Casting Quality Control

## Vue d'ensemble

Le projet est une application web conteneurisée (Docker) à **deux services** qui classifie des images de pièces de fonderie en **conforme (OK)** ou **défectueuse (DEF)** grâce à un pipeline de Machine Learning combinant un réseau de neurones profond (ResNet50) et un SVM. L'application permet également de **rechercher les 5 images les plus similaires** dans le dataset pour chaque pièce analysée.

L'interface est une **Single Page Application (SPA)** avec deux vues (Convoyeur et Similarité) partageant un **historique commun** des images analysées. Un **notebook Jupyter** (`ia_training.ipynb`) est fourni pour encoder le dataset et benchmarker les métriques de distance.

---

## Schéma d'architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Docker Network (casting-net)                 │
│                                                                     │
│  ┌──────────────────────────┐     ┌──────────────────────────────┐  │
│  │       FRONTEND           │     │          BACKEND             │  │
│  │   (casting-frontend)     │     │     (casting-backend)        │  │
│  │                          │     │                              │  │
│  │  FastAPI (port 3000)     │────▶│  FastAPI (port 8000)         │  │
│  │                          │     │                              │  │
│  │  ┌────────────────────┐  │     │  ┌────────────────────────┐  │  │
│  │  │  Fichiers statiques│  │     │  │  ResNet50 (PyTorch)    │  │  │
│  │  │  HTML / CSS / JS   │  │     │  │  Feature Extraction    │  │  │
│  │  └────────────────────┘  │     │  └──────────┬─────────────┘  │  │
│  │                          │     │             │                │  │
│  │  ┌────────────────────┐  │     │  ┌──────────▼─────────────┐  │  │
│  │  │  Proxy API (httpx) │──│─────│─▶│  StandardScaler        │  │  │
│  │  │  /api/* → backend  │  │     │  │  (scaler.joblib)       │  │  │
│  │  └────────────────────┘  │     │  └──────────┬─────────────┘  │  │
│  │                          │     │             │                │  │
│  └──────────────────────────┘     │  ┌──────────▼─────────────┐  │  │
│                                   │  │  SVM Classifier        │  │  │
│                                   │  │  (svm_model.joblib)    │  │  │
│                                   │  └────────────────────────┘  │  │
│                                   └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
         │
         │ Port 80 (hôte) → 3000 (conteneur)
         ▼
    🌐 Navigateur Web
```

---

## Services

### 1. Frontend (`casting-frontend`)

| Propriété       | Valeur                                     |
|-----------------|--------------------------------------------|
| **Technologie** | Python 3.11 + FastAPI + Uvicorn            |
| **Port**        | `3000` (mappé sur `80` côté hôte)          |
| **Rôle**        | Servir l'interface web + proxy API         |
| **Image Docker**| `python:3.11-slim`                         |

**Responsabilités :**
- Servir les fichiers statiques (`index.html`, `style.css`, `conveyor.js`, `similarity.js`, `history.js`, `nav.js`, `login.html`)
- L'interface est une **SPA** : `index.html` contient les deux vues (Convoyeur + Similarité) avec basculement JavaScript sans rechargement
- `similarity.html` redirige simplement vers `index.html` (rétrocompatibilité)
- Proxifier les appels `/api/*` vers le backend (via `httpx`) : `/api/classify`, `/api/similar`, `/api/images/*`, `/api/health`
- Gérer le routage : les routes `/api/*` sont déclarées avant le montage statique pour avoir la priorité

**Dépendances Python :**
- `fastapi` — framework web asynchrone
- `uvicorn[standard]` — serveur ASGI
- `httpx` — client HTTP asynchrone pour le proxy
- `python-multipart` — gestion des uploads de fichiers

---

### 2. Backend (`casting-backend`)

| Propriété       | Valeur                                     |
|-----------------|--------------------------------------------|
| **Technologie** | Python 3.12 + FastAPI + PyTorch + scikit-learn |
| **Port**        | `8000` (interne uniquement, non exposé)    |
| **Rôle**        | Serveur d'inférence Machine Learning       |
| **Image Docker**| `python:3.12-slim`                         |

**Responsabilités :**
- Charger les modèles au démarrage (ResNet50, SVM, Scaler, dataset de features)
- Extraire les features des images avec ResNet50
- Classifier les images avec le SVM
- Retourner le résultat (label, confiance, temps d'inférence)
- Rechercher les 5 images les plus similaires dans le dataset encodé
- Servir les images du dossier `casting_data/` via un endpoint dédié

**Dépendances Python :**
- `torch` + `torchvision` — réseau de neurones (ResNet50)
- `scikit-learn` — modèle SVM + StandardScaler
- `scipy` — calcul des distances (similarité)
- `joblib` — chargement des modèles sérialisés
- `numpy` — calculs numériques
- `Pillow` — manipulation d'images
- `fastapi` + `uvicorn[standard]` — serveur web
- `python-multipart` — gestion des uploads

---

## Réseau Docker

```yaml
networks:
  casting-net:
    driver: bridge
```

- Le réseau `casting-net` est un bridge Docker interne
- Le **frontend** peut atteindre le backend via `http://backend:8000` (résolution DNS Docker)
- Le **backend n'a aucun port exposé** vers l'hôte → isolation de sécurité
- Seul le frontend est accessible de l'extérieur sur le **port 80**

---

## Pipeline ML — Flux de classification

```
  Image (JPEG/PNG)
       │
       ▼
  ┌─────────────────────────┐
  │ 1. Preprocessing        │
  │    Resize → 224×224     │
  │    ToTensor             │
  │    Normalize (ImageNet) │
  │    mean=[.485,.456,.406]│
  │    std=[.229,.224,.225] │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 2. Feature Extraction   │
  │    ResNet50 (gelé)      │
  │    Sans couche FC       │
  │    → Vecteur 2048-dim   │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 3. Scaling              │
  │    StandardScaler       │
  │    (scaler.joblib)      │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 4. Classification       │
  │    SVM (svm_model.joblib)│
  │    Prédiction: 0=ok     │
  │                1=def    │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 5. Confiance            │
  │    sigmoid(|decision|)  │
  │    → score [0.5, 1.0]   │
  └───────────┬─────────────┘
              │
              ▼
      Réponse JSON
      {label, label_fr,
       confidence,
       inference_time_ms}
```

---

## Feature Extractor — Modèles supportés

La classe `FeatureExtractor` (`backend/feature_extractor.py`) supporte 4 backbones :

| Modèle        | Dimension de sortie | Poids                              |
|---------------|--------------------:|------------------------------------|
| **ResNet50** ✅| 2048               | Sauvegardés (`resnet50_extractor.pth`) |

> ✅ = Modèle utilisé en production. Les poids sont chargés depuis `resnet50_extractor.pth` (sauvegardés après entraînement). Tous les poids sont **gelés** (pas de fine-tuning).

---

## Endpoints API

### Backend (port 8000, interne)

| Méthode | Route              | Description                                    |
|---------|--------------------|------------------------------------------------|
| `GET`   | `/api/health`      | État du serveur (device, modèles)              |
| `POST`  | `/api/classify`    | Classifier une image (multipart/form)          |
| `POST`  | `/api/similar`     | Classification + top 5 images similaires       |
| `GET`   | `/api/images/{p}`  | Servir une image du dataset `casting_data/`    |

### Frontend (port 3000 → 80)

| Méthode | Route              | Description                                    |
|---------|--------------------|------------------------------------------------|
| `GET`   | `/api/health`      | Proxy → backend `/api/health`                  |
| `POST`  | `/api/classify`    | Proxy → backend `/api/classify`                |
| `POST`  | `/api/similar`     | Proxy → backend `/api/similar`                 |
| `GET`   | `/api/images/{p}`  | Proxy → backend `/api/images/{p}`              |
| `GET`   | `/*`               | Fichiers statiques (HTML, CSS, JS)             |

---

## Volumes Docker

```yaml
volumes:
  - ./frontend:/app       # Code source frontend (live reload)
  - ./backend:/app        # Code source backend (live reload)
  - ./models:/models      # Modèles ML (resnet50_extractor.pth, svm_model.joblib, scaler.joblib, features_dataset.npz)
  - ./casting_data:/casting_data  # Dataset d'images (servi via /api/images)
```

- Les volumes permettent le **rechargement automatique** du code en développement (`--reload`)
- Les modèles sont montés séparément pour pouvoir être mis à jour sans rebuild
- Le dataset `casting_data/` est monté en lecture dans le backend pour servir les images et calculer les similarités

---

## Support GPU (optionnel)

Le bloc GPU est commenté dans `docker-compose.yml`. Pour l'activer :

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

**Prérequis :** Drivers NVIDIA + NVIDIA Container Toolkit installés sur l'hôte.

Sans GPU, PyTorch utilise automatiquement le **CPU** (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

---

## Sécurité

| Aspect                    | Implémentation                                           |
|---------------------------|----------------------------------------------------------|
| **Isolation backend**     | Pas de port exposé, accessible uniquement via le réseau Docker |
| **Authentification**      | Côté client uniquement (`sessionStorage`), adaptée pour démo |
| **Validation des fichiers** | Vérification du `content_type` (doit commencer par `image/`) |
| **Timeout proxy**         | 60s pour `/api/classify`, 120s pour `/api/similar`, 30s pour images, 10s pour `/api/health` |
| **Gestion d'erreurs**     | Codes HTTP appropriés (400, 500, 502, 503)               |
| **Path traversal**        | Protection sur `/api/images/*` (vérification du chemin résolu) |

---

## Flux de données complet

```
Navigateur                    Frontend (3000)              Backend (8000)
    │                              │                            │
    │  1. Login (client-side)      │                            │
    │──────────────────────────▶   │                            │
    │                              │                            │
    │  2. Upload image(s)          │                            │
    │──────────────────────────▶   │                            │
    │                              │  3. POST /api/classify     │
    │                              │───────────────────────────▶│
    │                              │                            │
    │                              │     4. ResNet50 → SVM      │
    │                              │                            │
    │                              │  5. JSON {label, conf}     │
    │                              │◀───────────────────────────│
    │  6. Animation convoyeur      │                            │
    │◀─────────────────────────    │                            │
    │  7. Tri dans bac OK/DEF      │                            │
    │                              │                            │
    ─── Flux de Recherche de Similarité (SPA) ───

    │  8. Clic historique →         │                            │
    │     vue Similarité (SPA)      │                            │
    │  9. Bouton "Rechercher"       │                            │
    │────────────────────────────▶  │                            │
    │                              │  10. POST /api/similar     │
    │                              │───────────────────────────▶│
    │                              │     11. ResNet50 → SVM     │
    │                              │     + Similarity search    │
    │                              │     (distance metric on    │
    │                              │      features_dataset.npz) │
    │                              │                            │
    │                              │  12. JSON {label, similar} │
    │                              │◀───────────────────────────│
    │  13. Carousel top 5          │                            │
    │◀─────────────────────────    │                            │
    │  14. GET /api/images/* (×5)  │                            │
    │                        ──────│───────────────────────────▶│
    │                              │                            │```
