# =========================================
# Dockerfile MLflow - Bonnes Pratiques
# =========================================

# Étape 1: Image de base légère
FROM python:3.11-slim AS base

# Métadonnées de l'image
LABEL maintainer="Florian"
LABEL description="Serveur MLflow avec support PostgreSQL et S3"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# =========================================
# Étape 2: Installation des dépendances
# =========================================
FROM base AS builder

# Installation des dépendances système pour la compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Création d'un environnement virtuel
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Installation des packages Python
# sqlalchemy<2.1 : MLflow 2.19.0 plante à l'import avec SQLAlchemy 2.1
RUN pip install --upgrade pip && \
    pip install \
    mlflow==2.19.0 \
    "sqlalchemy<2.1" \
    psycopg2-binary \
    boto3

# =========================================
# Étape 3: Image finale légère
# =========================================
FROM base AS final

# Installation des dépendances runtime minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copie de l'environnement virtuel depuis le builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Création d'un utilisateur non-root (bonne pratique sécurité)
RUN groupadd --gid 1000 mlflow && \
    useradd --uid 1000 --gid mlflow --shell /bin/bash --create-home mlflow

# Création des répertoires de travail
RUN mkdir -p /mlflow/artifacts && \
    chown -R mlflow:mlflow /mlflow

# Copie du script d'entrée
COPY --chown=mlflow:mlflow entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Changement vers l'utilisateur non-root
USER mlflow
WORKDIR /mlflow

# Exposition du port MLflow
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Point d'entrée
ENTRYPOINT ["/entrypoint.sh"]
