# Tutoriel MLflow avec Docker

> **Guide complet pour maîtriser le tracking d'expériences ML avec MLflow**

---

## Table des matières

8. [Introduction](#-introduction)
9. [Architecture détaillée](#-architecture-détaillée)
10. [Prérequis](#-prérequis)
11. [Concepts clés](#-concepts-clés-de-mlflow)
12. [Guide pratique](#-guide-pratique)
13. [Bonnes pratiques](#-bonnes-pratiques)
14. [Troubleshooting](#-troubleshooting)

---

## Introduction

### Qu'est-ce que MLflow ?

**MLflow** est une plateforme open-source pour gérer le cycle de vie complet du Machine Learning. Elle répond aux défis courants des data scientists :

| Problème | Solution MLflow |
|----------|-----------------|
| "Quels hyperparamètres ai-je utilisé ?" | **Tracking** - Enregistrement automatique |
| "Quel modèle était le meilleur ?" | **Projects** - Comparaison des runs |
| "Comment déployer mon modèle ?" | **Models** - Format standardisé |
| "Quelle version est en production ?" | **Registry** - Gestion des versions |

### Architecture détaillée

Ce projet utilise une architecture conteneurisée moderne pour garantir la reproductibilité et l'isolation. Voici le rôle de chaque composant :

```mermaid
graph TD
    User[Utilisateur] -->|Code| Runner[Runner Container]
    User -->|Browser| MLflowUI[MLflow UI :5000]
    
    subgraph "Infrastructure Docker"
        Runner -->|Log Metrics| Server[MLflow Server]
        Runner -->|Upload Artefacts| MinIO[MinIO S3 :9000]
        
        Server -->|Read/Write Metadata| DB[(PostgreSQL)]
        Server -->|List Artefacts| MinIO
        
        MinIO_Setup[MinIO Setup] -.->|Init Bucket| MinIO
    end
```

#### 1. PostgreSQL (Backend Store)
Base de données relationnelle qui stocke toutes les **métadonnées légères** :
- La liste des expériences et des runs
- Les paramètres, métriques et tags
- L'historique des versions du Model Registry
- *Pourquoi ?* Offre performance et intégrité des données par rapport à un stockage fichier.

#### 2. MinIO (Artifact Store)
Serveur de stockage d'objets compatible Amazon S3. Il stocke les **fichiers lourds** (Artefacts) :
- Modèles sérialisés (`.pkl`, `.h5`)
- Graphiques et images
- Datasets
- *Pourquoi ?* Simule un environnement cloud (AWS S3) en local. MLflow ne stocke pas les fichiers dans la base de données.

#### 3. MinIO Setup
Conteneur éphémère qui ne tourne que quelques secondes au démarrage.
- *Rôle* : Créer automatiquement le bucket `mlflow-artifacts` dans MinIO.
- *Pourquoi ?* MinIO démarre vide par défaut. Ce service nous évite de devoir configurer manuellement le bucket à chaque lancement.

#### 4. Runner
Environnement d'exécution isolé contenant Python et les librairies nécessaires.
- *Rôle* : Exécuter vos scripts d'entraînement.
- *Pourquoi ?* Garantit que le code tourne exactement de la même manière pour tout le monde, indépendamment de votre machine hôte (plus de "ça marche sur ma machine").

#### Pourquoi un Dockerfile custom au lieu de `bitnami/mlflow` ?

On aurait pu utiliser une image prête à l'emploi comme `bitnami/mlflow` ou `ghcr.io/mlflow/mlflow`. Voici pourquoi on a choisi de construire notre propre image :

| Critère | Image tierce (Bitnami) | Dockerfile custom |
|---------|------------------------|-------------------|
| **Contrôle** | Versions fixées par le mainteneur | Vous choisissez exactement chaque lib |
| **Taille** | ~500 MB+ | ~300 MB (multi-stage) |
| **Sécurité** | Dépendances inconnues | Audit complet du contenu |
| **Apprentissage** | Boîte noire | Vous comprenez chaque étape |
| **Personnalisation** | Limitée | Totale (entrypoint, user, healthcheck) |

> [!TIP]
> **En production**, vous pouvez très bien utiliser `bitnami/mlflow` pour démarrer rapidement. Le Dockerfile custom de ce tutoriel est avant tout **pédagogique**.

---

## Prérequis

### Logiciels requis

| Logiciel | Usage | Vérification |
|----------|-------|--------------|
| WSL 2 (Ubuntu) | Environnement Linux sous Windows | `wsl --version` |
| Docker | Moteur de conteneurisation | `docker --version` |
| Docker Compose | Orchestration des services | `docker compose version` |

> [!TIP]
> **Pourquoi WSL ?** MLflow et de nombreux outils MLOps sont conçus pour Linux. WSL vous offre un environnement natif performant sans les problèmes de compatibilité de Windows.

### Démarrage du projet

1. **Lancer les services** :
   ```bash
   docker compose up -d
   ```

2. **Vérifier l'état** :
   ```bash
   docker compose ps
   ```
   Tous les services (`mlflow-server`, `mlflow-minio`, `mlflow-postgres`) doivent être `healthy`.

3. **Accéder à l'interface** :
   Ouvrez [http://localhost:5000](http://localhost:5000)

4. **Lancer un entraînement** :
   ```bash
   # Exécute le script dans le conteneur 'runner'
   docker compose exec runner python 02_sklearn_example.py
   ```

---

## Concepts clés de MLflow

### 1. Expérience (Experiment)

Une **expérience** est un conteneur pour organiser vos runs. Par exemple :
- `detection-pneumonie-v1`
- `classification-images-resnet`

```python
import mlflow

# Créer ou sélectionner une expérience
mlflow.set_experiment("mon-experience")
```

### 2. Run

Un **run** est une exécution unique d'un entraînement. Chaque run a :
- Un **ID unique**
- Des **paramètres** (hyperparamètres)
- Des **métriques** (résultats)
- Des **artefacts** (fichiers)
- Des **tags** (métadonnées)

```python
with mlflow.start_run(run_name="test-lr-0.01"):
    # Votre code d'entraînement
    pass
```

### 3. Paramètres vs Métriques

| Paramètres | Métriques |
|------------|-----------|
| Valeurs **fixes** en entrée | Valeurs **calculées** en sortie |
| Ex: learning_rate, batch_size | Ex: accuracy, loss |
| Log une seule fois | Peuvent être loggées plusieurs fois (par époque) |

```python
# Paramètres
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("epochs", 100)

# Métriques
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metric("loss", 0.1, step=10)  # Avec numéro d'étape
```

### 4. Artefacts

Les **artefacts** sont des fichiers associés au run :
- Modèles entraînés
- Graphiques (courbes de loss, matrices de confusion)
- Fichiers de configuration
- Datasets

```python
# Logger un fichier générique (texte, json, etc)
mlflow.log_artifact("config.yaml")

# Logger une figure Matplotlib (recommandé pour les images)
# Permet la visualisation native dans l'onglet "Images"
mlflow.log_figure(fig, "confusion_matrix.png")

# Logger un modèle scikit-learn
mlflow.sklearn.log_model(model, "model")
```

---

## Guide pratique

### Exemple 1 : Logging basique

```python
import mlflow

# Configuration
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tutoriel-basique")

with mlflow.start_run():
    # 1. Paramètres
    mlflow.log_param("algorithm", "random_forest")
    mlflow.log_param("n_estimators", 100)
    
    # 2. Entraînement (votre code)
    accuracy = 0.92
    
    # 3. Métriques
    mlflow.log_metric("accuracy", accuracy)
    
    # 4. Tags
    mlflow.set_tag("auteur", "Florian")
```

### Exemple 2 : Auto-logging avec scikit-learn

MLflow peut automatiquement logger les paramètres et métriques pour certains frameworks :

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier

# Activer l'auto-logging
mlflow.sklearn.autolog()

mlflow.set_experiment("autolog-demo")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    # MLflow log automatiquement les paramètres et métriques !
```

### Exemple 3 : Comparaison de modèles

Voir le fichier `examples/sklearn_example.py` pour un exemple complet de comparaison de plusieurs algorithmes.

### Exemple 4 : Logging du Dataset

Pour tracer quelles données ont été utilisées pour l'entraînement :

```python
import mlflow
import mlflow.data
import pandas as pd

# Créer un DataFrame avec vos données
df_train = pd.DataFrame(X_train, columns=feature_names)
df_train['target'] = y_train

# Créer et loguer le dataset MLflow
# source : origine des données (URL, fichier, API...)
# name : nom descriptif du dataset
dataset = mlflow.data.from_pandas(
    df_train,
    source="sklearn.datasets.load_wine",
    name="wine_training_data"
)

# Loguer le dataset comme input du run
# context : contexte d'utilisation ("training", "validation", "test")
mlflow.log_input(dataset, context="training")
```

> **Résultat** : Le dataset apparaîtra dans l'onglet "Overview" du run dans l'interface MLflow.

---

## Model Registry et Mise en Production

### Qu'est-ce que le Model Registry ?

Le **Model Registry** est un composant central de MLflow pour gérer le cycle de vie des modèles en production. Il permet de centraliser et de valider les modèles.

#### Concepts Avancés

1.  **Stages de déploiement**
    Un modèle peut évoluer à travers des étapes prédéfinies :
    *   **None** : Nouvellement enregistré.
    *   **Staging** : Prêt pour les tests (QA, intégration).
    *   **Production** : Validé et déployé pour les utilisateurs finaux.
    *   **Archived** : Ancienne version retirée.

2.  **Aliases (Nouveauté MLflow 2.x)**
    Au lieu d'utiliser des numéros de version ou des stages, vous pouvez utiliser des alias flexibles :
    *   `@champion` : Le modèle de référence actuel.
    *   `@challenger` : Un candidat potentiel pour remplacer le champion.
    
    ```python
    # Assignation d'un alias
    client.set_registered_model_alias("wine-classifier", "champion", 1)
    
    # Chargement via alias
    model = mlflow.sklearn.load_model("models:/wine-classifier@champion")
    ```

3.  **Politique de Versioning**
    Chaque nouvel enregistrement incrémente automatiquement la version (`v1`, `v2`, `v3`). MLflow garde l'historique complet, permettant un rollback instantané en cas de problème.

### Stages de déploiement

```
┌──────────┐     ┌──────────┐     ┌────────────┐     ┌──────────┐
│   None   │ ──► │ Staging  │ ──► │ Production │ ──► │ Archived │
│  (créé)  │     │  (test)  │     │ (déployé)  │     │ (ancien) │
└──────────┘     └──────────┘     └────────────┘     └──────────┘
```

| Stage | Description |
|-------|-------------|
| **None** | Modèle enregistré, non validé |
| **Staging** | En test/validation |
| **Production** | Déployé et actif |
| **Archived** | Ancienne version archivée |

### Enregistrer un modèle

```python
import mlflow
from mlflow.tracking import MlflowClient

# Après l'entraînement, enregistrer le modèle
model_uri = f"runs:/{run_id}/model"
model_version = mlflow.register_model(
    model_uri=model_uri,
    name="mon-modele-classifier"
)

print(f"Version créée : {model_version.version}")
```

### Promouvoir en Production

```python
client = MlflowClient()

# Transition vers Production
# archive_existing_versions=True : archive les anciennes versions en prod
client.transition_model_version_stage(
    name="mon-modele-classifier",
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True
)
```

### Charger un modèle en Production

```python
# Chargement par stage (recommandé pour les applications)
model = mlflow.sklearn.load_model("models:/mon-modele-classifier/Production")

# Chargement par version spécifique
model = mlflow.sklearn.load_model("models:/mon-modele-classifier/1")

# Utilisation
predictions = model.predict(X_new)
```



---

## Comparaison des Runs dans MLflow

### Utilisation de l'onglet "Compare"

1. Sélectionnez plusieurs runs (checkbox)
2. Cliquez sur "Compare"
3. Explorez les visualisations natives (Scatter Plot, Contour Plot, Parallel Coordinates)
4. **Onglet "Images"** : Grâce à `mlflow.log_figure()`, vous pouvez comparer visuellement les matrices de confusion côte à côte ou en superposition.

---



## Troubleshooting

### Problème : "Connection refused" sur port 5000

**Solution :**
```bash
# Vérifier que les services sont lancés
docker compose ps

# Si non, les démarrer
docker compose up -d

# Attendre que MLflow soit healthy (30 secondes)
docker compose logs -f mlflow
```

### Problème : "Permission denied" sur scripts shell

**Cause :** Les scripts `.sh` peuvent perdre leurs permissions d'exécution lors du passage Windows -> WSL.

**Solution :**
```bash
chmod +x mlflow/entrypoint.sh
```

### Problème : "Cannot set a deleted experiment"

**Cause :** Vous avez supprimé l'expérience dans l'interface MLflow. MLflow ne supprime pas définitivement l'expérience (soft delete), mais le nom reste réservé.

**Solution :**
1.  **Réinitialiser la base de données** (le plus simple) :
    ```bash
    docker compose down -v
    docker compose up -d
    ```
2.  Ou changer le nom de l'expérience dans le script Python (`experiment_name = "..."`).

### Problème : "Error connecting to S3"

**Cause :** Les credentials MinIO ne sont pas configurés.

**Solution :**
```python
import os

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio_admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio_secret_2024"
```

### Problème : L'interface MLflow est vide

**Solution :**
1. Vérifiez que vous avez exécuté un script d'exemple
2. Rafraîchissez la page
3. Sélectionnez la bonne expérience dans le menu déroulant



### Réinitialiser complètement

```bash
# Arrêter et supprimer tout
docker compose down -v

# Relancer proprement
docker compose up -d --build
```

---

## Ressources supplémentaires

- [Documentation officielle MLflow](https://mlflow.org/docs/latest/index.html)
- [MLflow GitHub](https://github.com/mlflow/mlflow)
- [Tutoriels vidéo MLflow](https://www.youtube.com/results?search_query=mlflow+tutorial)

---
