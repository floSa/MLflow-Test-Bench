"""
promotion_example.py - Promotion contrôlée d'un modèle
======================================================

L'entraînement ne met rien en production : il propose un @challenger. Ce script
est le seul à déplacer @champion, et seulement si le challenger passe le contrôle :
- mlflow.models.evaluate() : évalue @challenger et @champion sur le même jeu de test
- mlflow.validate_evaluation_results() : applique les seuils
- tags de la version : la décision et le run qui la justifie restent traçables
- @precedent : l'ancien champion reste désigné, pour revenir en arrière

Prérequis:
---------
docker compose exec runner python sklearn_example.py

Exécution:
---------
docker compose exec runner python promotion_example.py

Pour voir un refus, ce paramètre enregistre d'abord un modèle volontairement
mauvais : @champion ne bouge pas.
docker compose exec runner python promotion_example.py --modele-degrade

Interface MLflow:
----------------
Menu "Model registry" sur http://localhost:5000, modèle "wine-classifier"
"""

import sys

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from sklearn_example import TYPES_SKOPS_DE_CONFIANCE, configurer_mlflow, creer_pipeline

MODELE = "wine-classifier"

# Le même découpage que sklearn_example.py : ce jeu de test n'a pas servi à choisir
df = load_wine(as_frame=True).frame
_, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])


def evaluer(alias, version):
    """Évalue une version du modèle sur le jeu de test, dans un run qui garde la preuve."""
    with mlflow.start_run(run_name=f"{alias}-v{version}") as run:
        resultat = mlflow.models.evaluate(
            f"models:/{MODELE}/{version}",
            data=df_test,
            targets="target",
            model_type="classifier",
            evaluators="default",
            evaluator_config={"log_explainer": False, "log_model_explainability": False},
        )
    print(f"@{alias} (v{version}) : accuracy {resultat.metrics['accuracy_score']:.4f}")
    return resultat, run.info.run_id


def enregistrer_modele_degrade(client):
    """Propose un modèle qui prédit toujours la même classe, pour montrer un refus."""
    X, y = df.drop(columns="target"), df["target"]
    with mlflow.start_run(run_name="modele-degrade"):
        modele = creer_pipeline(DummyClassifier(strategy="most_frequent")).fit(X, y)
        info = mlflow.sklearn.log_model(
            modele,
            name="model",
            input_example=X.head(5),
            skops_trusted_types=TYPES_SKOPS_DE_CONFIANCE,
        )
    version = mlflow.register_model(info.model_uri, MODELE)
    client.set_registered_model_alias(MODELE, "challenger", version.version)


def promouvoir(client):
    """Déplace @champion sur @challenger si les seuils sont respectés."""
    aliases = client.get_registered_model(MODELE).aliases
    challenger = aliases["challenger"]
    champion = aliases.get("champion")

    print("\n--- Évaluation sur le jeu de test ---")
    resultat, run_id = evaluer("challenger", challenger)
    # Sans champion en place, seul le seuil absolu s'applique
    reference = evaluer("champion", champion)[0] if champion else None

    # Au moins 0,9 dans l'absolu, et pas moins bien que le champion en place
    seuil = MetricThreshold(
        threshold=0.9, min_relative_change=0.0 if champion else None, greater_is_better=True
    )

    print("\n--- Décision ---")
    try:
        mlflow.validate_evaluation_results({"accuracy_score": seuil}, resultat, reference)
    except MlflowException as e:
        decision = "refuse"
        print(f"Refusé : {e.message}")
    else:
        decision = "approuve"
        if champion and champion != challenger:
            client.set_registered_model_alias(MODELE, "precedent", champion)
        client.set_registered_model_alias(MODELE, "champion", challenger)
        print(f"Approuvé : @champion passe sur la version {challenger}")

    # Tags de version : contrairement à ceux d'un Logged Model, l'interface les affiche
    client.set_model_version_tag(MODELE, challenger, "validation", decision)
    client.set_model_version_tag(MODELE, challenger, "validation_run_id", run_id)


def afficher_alias(client):
    aliases = client.get_registered_model(MODELE).aliases
    print(f"\n--- Alias de {MODELE} ---")
    for alias, version in sorted(aliases.items()):
        tags = client.get_model_version(MODELE, version).tags
        print(f"@{alias} : version {version}, validation={tags.get('validation', '-')}")


def main():
    configurer_mlflow()
    mlflow.set_experiment("promotion-modeles")
    client = mlflow.MlflowClient()

    if "--modele-degrade" in sys.argv:
        enregistrer_modele_degrade(client)

    promouvoir(client)
    afficher_alias(client)


if __name__ == "__main__":
    main()
