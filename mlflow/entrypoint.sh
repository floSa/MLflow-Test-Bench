#!/bin/bash
# =========================================
# Script d'entrée MLflow
# =========================================

set -e

echo "========================================="
echo "   🚀 Démarrage du serveur MLflow"
echo "========================================="

# Affichage de la configuration (sans les secrets)
echo "📦 Backend Store: PostgreSQL"
echo "📁 Artifact Store: S3 (SeaweedFS)"
echo "🌐 Port: 5000"
echo ""

# Démarrage du serveur MLflow
# --no-serve-artifacts : les clients écrivent directement dans le stockage S3
# avec leurs propres identifiants, le serveur ne relaie pas les artefacts
exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
    --no-serve-artifacts \
    --workers 2
