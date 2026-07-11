# 🧪 MLflow + Docker - Tutoriel avec Bonnes Pratiques

> Un environnement MLflow prêt à l'emploi avec Docker pour le tracking d'expériences ML.

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.19.0-orange)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.11-green?logo=python)](https://python.org/)

---

## 🎯 Objectif

Ce projet fournit :
- ✅ Un **serveur MLflow** containerisé prêt pour la production
- ✅ Un **tutoriel complet** en français pour apprendre MLflow
- ✅ Des **exemples de code** avec les bonnes pratiques

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   MLflow    │────▶│   MinIO     │     │ PostgreSQL  │
│   :5000     │     │   :9000     │     │   :5432     │
│  (Tracking) │     │  (S3 Store) │     │  (Backend)  │
└─────────────┘     └─────────────┘     └─────────────┘
      ▲                   │                   │
      │                   └───────────────────┘
      │                         Docker Compose
      │
┌─────────────┐
│   Runner    │  ← Environnement Python isolé
│  (examples) │    pour exécuter les scripts
└─────────────┘
```

Le `docker compose up` orchestre **5 services** (un conteneur `minio-setup`
éphémère crée le bucket S3 au démarrage puis s'arrête) :

| Service | Image / Build | Port interne | Port hôte | Rôle |
|---------|---------------|--------------|-----------|------|
| `mlflow` | build `./mlflow` (Python 3.11 + MLflow 2.19.0) | 5000 | 5000 | Serveur de tracking + UI |
| `postgres` | `postgres:15-alpine` | 5432 | *(non exposé)* | Backend store (métadonnées) |
| `minio` | `minio/minio:latest` | 9000 / 9001 | 9000 / 9001 | Artifact store S3 + console |
| `minio-setup` | `minio/mc:latest` | — | — | Crée le bucket `mlflow-artifacts` puis s'arrête |
| `runner` | build `./examples` (Python 3.11) | — | *(non exposé)* | Environnement d'exécution des exemples |

> Détails pédagogiques (rôle de chaque brique, diagramme Mermaid, choix du Dockerfile
> custom) : [**TUTORIAL.md**](TUTORIAL.md).

## 🚀 Démarrage rapide

### 1. Prérequis

- **WSL 2** (recommandé) avec une distribution Linux (ex: Ubuntu)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé avec l'intégration WSL 2 activée
- Python 3.9+ (dans votre distribution WSL)

### 2. Lancer les services

```bash
# Démarrer tous les services
docker compose up -d

# Vérifier le statut
docker compose ps
```

### 3. Accéder à MLflow

Ouvrir **http://localhost:5000** dans votre navigateur.

### 4. Exécuter un exemple

```bash
# Exécuter l'exemple scikit-learn dans le conteneur Docker
docker compose exec runner python sklearn_example.py
```

> **💡 Note** : Aucune installation Python locale n'est requise !
> Le service `runner` contient un environnement Python isolé avec toutes les dépendances.
> Il exécute d'ailleurs `sklearn_example.py` **automatiquement au démarrage** (voir la
> directive `command` dans `docker-compose.yml`), puis reste actif pour un usage interactif.

## 📁 Structure du projet

```text
mlflow_tuto/
├── docker-compose.yml       # Orchestration des 5 services
├── .env.example             # Modèle de configuration (à copier en .env)
├── mlflow/
│   ├── Dockerfile           # Image serveur MLflow (multi-stage, non-root)
│   └── entrypoint.sh        # Lancement de `mlflow server`
├── examples/
│   ├── Dockerfile           # Image du runner Python
│   ├── requirements.txt     # Dépendances des exemples
│   └── sklearn_example.py   # Exemple complet : comparaison + Model Registry
├── documentation/
│   └── SECURITY.md          # Posture de sécurité (secrets, réseau, conteneurs)
├── TUTORIAL.md              # Tutoriel pédagogique complet
├── LICENSE                  # MIT
└── README.md
```

## 🌐 Ports exposés

| Service | Port | URL |
|---------|------|-----|
| MLflow UI | 5000 | http://localhost:5000 |
| MinIO API | 9000 | http://localhost:9000 |
| MinIO Console | 9001 | http://localhost:9001 |

## 📚 Documentation

| Document | Contenu |
|----------|---------|
| [**TUTORIAL.md**](TUTORIAL.md) | Tutoriel pédagogique complet : concepts, guide pratique, Model Registry, troubleshooting |
| [documentation/SECURITY.md](documentation/SECURITY.md) | Posture de sécurité : secrets, isolation réseau, conteneurs, risques connus |

➡️ Pour apprendre à utiliser MLflow, commencez par le [**Tutoriel complet**](TUTORIAL.md).

## 🛠️ Commandes utiles

```bash
# Démarrer les services
docker compose up -d

# Voir les logs
docker compose logs -f mlflow

# Arrêter les services
docker compose down

# Réinitialiser (supprime les données)
docker compose down -v
docker compose up -d --build
```

## ⚙️ Configuration

Copiez `.env.example` en `.env`, puis renseignez les valeurs. Variables réelles
(source : [`.env.example`](.env.example)) :

| Variable | Défaut (`.env.example`) | Effet |
|----------|-------------------------|-------|
| `POSTGRES_USER` | `mlflow` | Utilisateur du backend store PostgreSQL |
| `POSTGRES_PASSWORD` | `CHANGEZ_MOI_EN_PRODUCTION` | Mot de passe PostgreSQL (**à changer**) |
| `POSTGRES_DB` | `mlflow` | Nom de la base de métadonnées |
| `MINIO_ROOT_USER` | `minio_admin` | Identifiant admin MinIO = `AWS_ACCESS_KEY_ID` |
| `MINIO_ROOT_PASSWORD` | `CHANGEZ_MOI_EN_PRODUCTION` | Secret MinIO = `AWS_SECRET_ACCESS_KEY` (**à changer**) |
| `MINIO_BUCKET` | `mlflow-artifacts` | Bucket S3 des artefacts (créé par `minio-setup`) |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | URL du serveur MLflow (usage depuis l'hôte) |

> ⚠️ `.env` contient les secrets et n'est **pas** versionné (voir `.gitignore`).
> Ne jamais committer ce fichier. Détails : [documentation/SECURITY.md](documentation/SECURITY.md).

## 📖 Exemples inclus

| Fichier | Description |
|---------|-------------|
| `examples/sklearn_example.py` | Exemple complet de bout en bout : chargement du dataset **Wine**, comparaison de 4 modèles scikit-learn (Random Forest, Logistic Regression, SVM RBF, Random Forest Large), logging des paramètres/métriques/artefacts (matrices de confusion via `log_figure`), puis enregistrement du meilleur modèle dans le **Model Registry** et mise en Production |

> Le tutoriel évoque d'autres concepts (auto-logging, logging de dataset) sous forme
> de snippets ; le seul script exécutable fourni est `sklearn_example.py`.

## 🔧 Troubleshooting

| Problème | Solution |
|----------|----------|
| "Connection refused" | `docker compose up -d` puis attendre 30 secondes |
| L'UI est vide | Exécutez un exemple, puis rafraîchissez |
| Erreur S3 | Vérifiez que les variables d'environnement sont bien configurées |
| **Erreur Buildx (WSL)** | `fork/exec ... docker-buildx: no such file or directory`. Voir Troubleshooting. |

---

## 📦 Licences & composants

Briques réellement utilisées (source : `docker-compose.yml`, `mlflow/Dockerfile`,
`examples/requirements.txt`) et leur licence. Valeurs à vérifier avant tout usage
en production.

| Composant | Rôle | Licence |
|-----------|------|---------|
| MLflow 2.19.0 | Serveur de tracking + Model Registry | Apache-2.0 |
| PostgreSQL 15 | Backend store (métadonnées) | PostgreSQL License (open-source, proche BSD/MIT) |
| MinIO | Artifact store S3 | AGPL-3.0 *(à confirmer selon la version `latest`)* |
| MinIO Client (`mc`) | Création du bucket au démarrage | AGPL-3.0 *(à confirmer)* |
| Python 3.11 | Runtime des images | PSF License |
| scikit-learn, pandas, numpy | ML & données (exemples) | BSD-3-Clause |
| matplotlib | Graphiques (matrices de confusion) | Matplotlib License (type BSD) |
| boto3 | Client S3 pour MinIO | Apache-2.0 |
| psycopg2-binary | Driver PostgreSQL du serveur | LGPL-3.0 *(à confirmer)* |
| python-dotenv | Chargement du `.env` | BSD-3-Clause |
| **Ce projet** | Code & documentation | MIT — Copyright (c) 2026 Florian H. |

> Le tag `minio/minio:latest` n'épingle pas de version : la licence exacte dépend de
> l'image tirée au moment du build. À figer et vérifier pour un usage sérieux.

## 📄 Licence

Ce projet est sous licence **MIT** - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

*Créé avec ❤️ pour apprendre MLflow avec Docker*
