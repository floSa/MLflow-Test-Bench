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
# Exécuter le quickstart dans le conteneur Docker
docker compose exec runner python 01_quickstart.py

# Ou l'exemple scikit-learn
docker compose exec runner python 02_sklearn_example.py
```

> **💡 Note** : Aucune installation Python locale n'est requise !
> Le service `runner` contient un environnement Python isolé avec toutes les dépendances.

## 📁 Structure du projet

```
mlflow/
├── docker-compose.yml      # Orchestration des 3 services
├── .env                    # Variables d'environnement (secrets)
├── mlflow/
│   ├── Dockerfile          # Image MLflow personnalisée
│   └── entrypoint.sh       # Script de démarrage
├── examples/
│   ├── requirements.txt    # Dépendances Python
│   ├── 01_quickstart.py    # Exemple de base
│   ├── 02_sklearn_example.py  # Comparaison de modèles
│   └── 03_model_registry.py   # Gestion du Model Registry
├── docs/
│   └── TUTORIAL.md         # Tutoriel complet
└── README.md
```

## 🌐 Ports exposés

| Service | Port | URL |
|---------|------|-----|
| MLflow UI | 5000 | http://localhost:5000 |
| MinIO API | 9000 | http://localhost:9000 |
| MinIO Console | 9001 | http://localhost:9001 |

## 📚 Documentation

➡️ Consultez le [**Tutoriel complet**](docs/TUTORIAL.md) pour apprendre à utiliser MLflow.

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

Modifiez le fichier `.env` pour personnaliser les credentials :

```env
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=votre_mot_de_passe
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=votre_mot_de_passe
```

## 📖 Exemples inclus

| Fichier | Description |
|---------|-------------|
| `01_quickstart.py` | Introduction aux concepts de base (params, metrics, tags) |
| `02_sklearn_example.py` | Comparaison de modèles avec scikit-learn + Model Registry |
| `03_model_registry.py` | Gestion du cycle de vie des modèles (versioning, production) |

## 🔧 Troubleshooting

| Problème | Solution |
|----------|----------|
| "Connection refused" | `docker compose up -d` puis attendre 30 secondes |
| L'UI est vide | Exécutez un exemple, puis rafraîchissez |
| Erreur S3 | Vérifiez que les variables d'environnement sont bien configurées |
| **Erreur Buildx (WSL)** | `fork/exec ... docker-buildx: no such file or directory`. Voir Troubleshooting. |

---

## 📄 Licence

Ce projet est sous licence **MIT** - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

*Créé avec ❤️ pour apprendre MLflow avec Docker*
