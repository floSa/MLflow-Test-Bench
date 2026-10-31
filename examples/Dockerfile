# =========================================
# Dockerfile - Runner pour les exemples MLflow
# Environnement Python isolé avec venv
# =========================================

FROM python:3.12-slim AS base

# Métadonnées de l'image
LABEL maintainer="Florian"
LABEL description="Environnement pour exécuter les exemples MLflow"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv

# Création de l'environnement virtuel
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Répertoire de travail
WORKDIR /app

# Copie et installation des dépendances
COPY requirements.lock .
RUN pip install --upgrade pip && \
    pip install -r requirements.lock

# Copie des scripts d'exemples
COPY . .

# Utilisateur non-root (bonne pratique sécurité)
RUN groupadd --gid 1000 runner && \
    useradd --uid 1000 --gid runner --shell /bin/bash --create-home runner && \
    chown -R runner:runner /app /opt/venv

USER runner

# Commande par défaut - garde le conteneur actif
CMD ["tail", "-f", "/dev/null"]

# Image pour le deep learning : le runner, plus PyTorch en version CPU
FROM base AS torch
COPY requirements-torch.lock .
RUN pip install -r requirements-torch.lock

# Image par défaut, en dernier : le runner seul
FROM base AS runner
