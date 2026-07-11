# Sécurité — MLflow + Docker (tutoriel)

> Ce projet est un **environnement de tutoriel local**. La posture ci-dessous
> décrit ce qui est réellement en place dans le code, et signale ⚠️ les points à
> durcir avant tout usage exposé ou en production.

---

## 1. Secrets & configuration

Les secrets vivent dans un fichier `.env` **non versionné** (première ligne de
[`.gitignore`](../.gitignore)). Le dépôt ne fournit qu'un modèle
[`.env.example`](../.env.example) avec des valeurs marquées `CHANGEZ_MOI_EN_PRODUCTION`.
Aucun secret n'est écrit en dur dans `docker-compose.yml` : tout passe par des
variables d'environnement (`${POSTGRES_PASSWORD}`, `${MINIO_ROOT_PASSWORD}`…).

| Secret | Où | Rotation |
|--------|----|----------|
| `POSTGRES_PASSWORD` | `.env` (non versionné) | manuelle |
| `MINIO_ROOT_PASSWORD` | `.env` (non versionné) | manuelle |

Point d'attention : les identifiants MinIO servent aussi de credentials S3. Les
services `mlflow` et `runner` réutilisent `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`
comme `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. Un même secret couvre donc
l'admin MinIO **et** l'accès aux artefacts.

> ⚠️ `.env` ne doit **jamais** être committé. Il est déjà couvert par `.gitignore`,
> mais vérifiez-le après un clone ou un copier-coller Windows → WSL.

---

## 2. Isolation réseau

Tous les services sont sur un réseau Docker dédié `mlflow-network` (driver `bridge`).
Seuls les ports strictement nécessaires sont publiés sur l'hôte.

| Service | Exposé à l'hôte ? | Port(s) | Justification |
|---------|-------------------|---------|---------------|
| `mlflow` | oui | 5000 | UI + API de tracking |
| `minio` | oui | 9000 (API S3), 9001 (console) | Accès aux artefacts et console web |
| `postgres` | **non** | 5432 (interne) | Accessible uniquement par `mlflow` via le réseau Docker |
| `runner` | **non** | — | Exécution de scripts, aucun service à exposer |
| `minio-setup` | **non** | — | Conteneur éphémère (init du bucket) |

Le backend PostgreSQL n'est donc **pas** joignable depuis la machine hôte, ce qui
réduit la surface d'attaque.

> ⚠️ Ni l'UI MLflow (`:5000`) ni la console MinIO (`:9001`) n'ont d'authentification
> applicative ou de TLS dans cette configuration. Acceptable en local ; à placer
> derrière un reverse-proxy authentifié + HTTPS si exposé au réseau.

---

## 3. Conteneurs

Les deux images construites appliquent des bonnes pratiques :

| Durcissement | Détail | Source |
|--------------|--------|--------|
| Utilisateur non-root | `mlflow` (uid 1000) / `runner` (uid 1000) | `mlflow/Dockerfile`, `examples/Dockerfile` |
| Build multi-stage | Étape `builder` séparée de l'image finale (surface réduite) | `mlflow/Dockerfile` |
| Runtime minimal | `libpq5` + `curl` seulement dans l'image finale | `mlflow/Dockerfile` |
| Healthchecks | `pg_isready`, `mc ready`, `curl /health` | `docker-compose.yml` |
| `restart: unless-stopped` | Résilience des services longs | `docker-compose.yml` |

---

## 4. Dépendances

Le serveur MLflow épingle sa version (`mlflow==2.19.0`, `mlflow/Dockerfile`). Les
dépendances des exemples ([`examples/requirements.txt`](../examples/requirements.txt))
utilisent des bornes minimales (`>=`) et ne sont **pas** verrouillées par un lockfile.

Les images d'infrastructure utilisent des tags mouvants :
`minio/minio:latest` et `minio/mc:latest` (non reproductibles) ;
`postgres:15-alpine` est figé sur une majeure mais suit les correctifs.

Audit possible des dépendances Python :

```bash
docker compose exec runner pip install pip-audit && \
docker compose exec runner pip-audit
```

---

## 5. Données & accès

Les données manipulées par l'exemple sont **publiques et non sensibles** (dataset
`sklearn.datasets.load_wine`). Aucune donnée personnelle n'est stockée.

La persistance repose sur deux volumes Docker nommés :

| Volume | Monté par | Contenu |
|--------|-----------|---------|
| `mlflow-postgres-data` | `postgres` | Métadonnées (runs, params, métriques, Model Registry) |
| `mlflow-minio-data` | `minio` | Artefacts (modèles sérialisés, figures) |

---

## 6. Risques connus (non traités)

Honnêteté : ce projet est un tutoriel local, plusieurs points ne sont pas durcis.

- ⚠️ **Bucket d'artefacts en lecture anonyme** — le conteneur `minio-setup` exécute
  `mc anonymous set download myminio/${MINIO_BUCKET}`, ce qui rend le bucket
  `mlflow-artifacts` **lisible sans authentification** (source :
  `docker-compose.yml`). Pratique pour explorer les artefacts en local ; à retirer
  pour tout déploiement, car cela expose modèles et figures.
- ⚠️ **Pas d'authentification sur l'UI MLflow** — quiconque atteint `:5000` peut
  lire/modifier les expériences et le registry. Prévoir un proxy authentifié en prod.
- ⚠️ **Secrets partagés** — les credentials MinIO servent à la fois d'admin objet et
  de clés S3 applicatives. Séparer les rôles (clé applicative dédiée) en prod.
- ⚠️ **Tags d'images `latest`** — `minio/minio:latest` et `minio/mc:latest` ne sont
  pas reproductibles ; figer des versions précises pour un usage sérieux.
- ⚠️ **Pas de TLS** — tout le trafic (UI, S3, Postgres) est en clair sur le réseau
  Docker et l'hôte.
