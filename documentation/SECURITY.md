# Sécurité — MLflow + Docker (tutoriel)

> Ce projet est un **environnement de tutoriel local**. La posture ci-dessous
> décrit ce qui est réellement en place dans le code, et signale les points à
> durcir avant tout usage exposé ou en production.

---

## 1. Secrets & configuration

Les secrets vivent dans un fichier `.env` **non versionné** (première ligne de
[`.gitignore`](../.gitignore)). Le dépôt ne fournit qu'un modèle
[`.env.example`](../.env.example) avec des valeurs marquées `CHANGEZ_MOI_EN_PRODUCTION`.
Aucun secret n'est écrit en dur dans `docker-compose.yml` : tout passe par des
variables d'environnement (`${POSTGRES_PASSWORD}`, `${S3_SECRET_KEY}`…).

| Secret | Où | Rotation |
|--------|----|----------|
| `POSTGRES_PASSWORD` | `.env` (non versionné) | manuelle |
| `S3_SECRET_KEY` | `.env` (non versionné) | manuelle |

Point d'attention : SeaweedFS lit ses identités dans un fichier. Le conteneur l'écrit au
démarrage dans un tmpfs, à partir de `S3_ACCESS_KEY` / `S3_SECRET_KEY` : il n'est ni
versionné ni monté depuis le dépôt. Les services `mlflow` et `runner` réutilisent ces
valeurs comme `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. Une seule identité, avec les
droits `Admin`, couvre donc toute la gestion du stockage **et** l'accès aux artefacts.

> **Attention** : `.env` ne doit **jamais** être committé. Il est déjà couvert par
> `.gitignore`, mais vérifiez-le après un clone ou un copier-coller Windows → WSL.

---

## 2. Isolation réseau

Tous les services sont sur un réseau Docker dédié `mlflow-network` (driver `bridge`).
Seuls les ports strictement nécessaires sont publiés sur l'hôte.

| Service | Exposé à l'hôte ? | Port(s) | Justification |
|---------|-------------------|---------|---------------|
| `mlflow` | oui, sur `127.0.0.1` | 5000 | UI + API de tracking |
| `seaweedfs` | oui, sur `127.0.0.1` | 8333 (API S3), 8888 (filer), 9333 (cluster) | Accès aux artefacts et interfaces web |
| `postgres` | **non** | 5432 (interne) | Accessible uniquement par `mlflow` via le réseau Docker |
| `runner` | **non** | — | Exécution de scripts, aucun service à exposer |
| `seaweedfs-setup` | **non** | — | Conteneur éphémère (init du bucket) |

Le backend PostgreSQL n'est donc **pas** joignable depuis la machine hôte, ce qui
réduit la surface d'attaque.

> **Attention** : l'UI MLflow (`:5000`) et les interfaces SeaweedFS (`:8888`, `:9333`)
> n'ont aucune authentification, et rien n'est chiffré. L'API S3 (`:8333`) refuse les requêtes anonymes
> (403). Tous ces ports sont publiés seulement sur `127.0.0.1`. Acceptable en local ; à
> placer derrière un reverse-proxy authentifié + HTTPS si exposé au réseau.

Depuis MLflow 3.5, le serveur refuse aussi les en-têtes `Host` inconnus, pour se protéger
du DNS rebinding : une requête avec `Host: evil.example:5000` reçoit un 403.
`MLFLOW_SERVER_ALLOWED_HOSTS` autorise `mlflow:5000`, l'adresse du serveur vue du `runner`,
ainsi que `localhost` et `127.0.0.1`.

**Télémétrie** : MLflow 3 envoie par défaut des statistiques d'usage à ses serveurs.
`MLFLOW_DISABLE_TELEMETRY` la désactive pour le serveur et le `runner`.

---

## 3. Conteneurs

Les deux images construites appliquent des bonnes pratiques :

| Durcissement | Détail | Source |
|--------------|--------|--------|
| Utilisateur non-root | `mlflow` (uid 1000) / `runner` (uid 1000) | `mlflow/Dockerfile`, `examples/Dockerfile` |
| Build multi-stage | Étape `builder` séparée de l'image finale (surface réduite) | `mlflow/Dockerfile` |
| Runtime minimal | `libpq5` + `curl` seulement dans l'image finale | `mlflow/Dockerfile` |
| Healthchecks | `pg_isready`, `wget /healthz`, `curl /health` | `docker-compose.yml` |
| `restart: unless-stopped` | Résilience des services longs | `docker-compose.yml` |

---

## 4. Dépendances

Le serveur et les exemples installent les versions exactes de leur `requirements.lock`,
généré par `uv pip compile` depuis `requirements.txt`. Un `pip-audit` de ces deux fichiers
ne remonte aucune vulnérabilité connue au 30 octobre 2026. Pour y arriver, `filelock` est
imposé en version >= 4.2.1, ce qui redescend LiteLLM à 1.103.10.

`chrislusf/seaweedfs:4.51` est épinglée sur une version précise ;
`postgres:15-alpine` est figé sur une majeure mais suit les correctifs.

Audit des dépendances, depuis la racine du dépôt :

```bash
uvx pip-audit --no-deps --disable-pip -r examples/requirements.lock
```

---

## 5. Données & accès

Les données manipulées par l'exemple sont **publiques et non sensibles** (dataset
`sklearn.datasets.load_wine`). Aucune donnée personnelle n'est stockée.

La persistance repose sur deux volumes Docker nommés :

| Volume | Monté par | Contenu |
|--------|-----------|---------|
| `mlflow-postgres-data` | `postgres` | Métadonnées (runs, params, métriques, Model Registry) |
| `mlflow-seaweedfs-data` | `seaweedfs` | Artefacts (modèles sérialisés, figures) |

---

## 6. Charger un modèle

Charger un modèle peut exécuter du code contenu dans ses fichiers, quel que soit le format.
`skops`, le format par défaut, bloque les types inconnus, mais sa liste de types de confiance
est lue dans le modèle lui-même. L'autolog utilise toujours cloudpickle. Ne chargez que des
modèles dont vous connaissez l'origine.

---

## 7. Risques connus (non traités)

Honnêteté : ce projet est un tutoriel local, plusieurs points ne sont pas durcis.

- **Filer sans authentification** — toute personne ayant accès à `127.0.0.1:8888` peut
  parcourir et modifier les artefacts sans passer par l'API S3. Pratique pour explorer
  en local ; à ne pas publier pour tout déploiement, car cela expose modèles et figures.
- **Pas d'authentification sur l'UI MLflow** — quiconque atteint `:5000` peut
  lire/modifier les expériences et le registry. Prévoir un proxy authentifié en prod.
- **Identité unique** — MLflow et le `runner` partagent une même clé S3 avec les droits
  `Admin`. Déclarer des identités distinctes dans le fichier d'identités en prod.
- **Pas de TLS** — tout le trafic (UI, S3, Postgres) est en clair sur le réseau
  Docker et l'hôte.
