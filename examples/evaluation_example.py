"""
evaluation_example.py - Logged Models, évaluation et validation par seuils
==========================================================================

Compare un modèle candidat à un modèle de référence avec les fonctions de
MLflow 3, puis n'approuve le candidat que s'il respecte des seuils :
- mlflow.sklearn.log_model() : chaque modèle devient un Logged Model
- mlflow.models.evaluate() : métriques et graphiques calculés par MLflow
- mlflow.validate_evaluation_results() : le contrôle qualité avant promotion
- log_system_metrics=True : CPU et mémoire suivis pendant le run

Exécution:
---------
docker compose exec runner python evaluation_example.py

Interface MLflow:
----------------
Expérience "evaluation-modeles" sur http://localhost:5000, onglet "Models"
"""

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn_example import TYPES_SKOPS_DE_CONFIANCE, configurer_mlflow, creer_pipeline


def evaluer(nom, estimateur, df_train, dataset_test):
    """Entraîne un modèle, le logge comme Logged Model et l'évalue sur le test."""
    # log_system_metrics : un fil de fond relève CPU et mémoire pendant le run
    with mlflow.start_run(run_name=nom, log_system_metrics=True):
        modele = creer_pipeline(estimateur)
        modele.fit(df_train.drop(columns="target"), df_train["target"])

        model_info = mlflow.sklearn.log_model(
            modele,
            name=nom,
            input_example=df_train.drop(columns="target").head(5),
            skops_trusted_types=TYPES_SKOPS_DE_CONFIANCE,
        )

        # Métriques et graphiques (matrice de confusion, ROC...) calculés par
        # MLflow, puis rattachés au run, au Logged Model et au dataset
        resultat = mlflow.models.evaluate(
            model_info.model_uri,
            data=dataset_test,
            model_type="classifier",
            evaluators="default",
            evaluator_config={"log_explainer": False, "log_model_explainability": False},
        )

    print(f"{nom} : accuracy {resultat.metrics['accuracy_score']:.4f}, "
          f"f1 {resultat.metrics['f1_score']:.4f}")
    return model_info, resultat


def valider(seuils, candidat, reference):
    """Renvoie True si le candidat respecte les seuils, en affichant le motif sinon."""
    try:
        mlflow.validate_evaluation_results(seuils, candidat, reference)
    except MlflowException as e:
        print(f"Validation refusée : {e.message}")
        return False
    print("Validation réussie")
    return True


def main():
    configurer_mlflow()
    experience = mlflow.set_experiment("evaluation-modeles")

    # Toutes les secondes au lieu de 10 : ces runs ne durent que quelques secondes
    mlflow.set_system_metrics_sampling_interval(1)
    mlflow.set_system_metrics_samples_before_logging(1)

    df = load_wine(as_frame=True).frame
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
    dataset_test = mlflow.data.from_pandas(
        df_test, source="sklearn.datasets.load_wine", targets="target", name="wine_test"
    )

    print("\n--- Évaluation ---")
    # La référence prédit toujours la classe la plus fréquente : un candidat
    # doit au moins faire mieux que ça
    _, reference = evaluer(
        "reference", DummyClassifier(strategy="most_frequent"), df_train, dataset_test
    )
    info_candidat, candidat = evaluer(
        "candidat", RandomForestClassifier(n_estimators=1000, random_state=42), df_train, dataset_test
    )

    print("\n--- Seuils raisonnables ---")
    # Accuracy d'au moins 0,9 et d'au moins 5 % supérieure à celle de la référence
    approuve = valider(
        {"accuracy_score": MetricThreshold(
            threshold=0.9, min_relative_change=0.05, greater_is_better=True
        )},
        candidat,
        reference,
    )

    print("\n--- Seuil impossible, pour voir un refus ---")
    valider(
        {"accuracy_score": MetricThreshold(threshold=1.01, greater_is_better=True)},
        candidat,
        reference,
    )

    # Le statut suit le Logged Model. L'interface 3.17 n'affiche pas ces tags : il
    # se relit en Python (tags.validation peut aussi servir de filtre de recherche)
    statut = "approuve" if approuve else "refuse"
    mlflow.MlflowClient().set_logged_model_tags(info_candidat.model_id, {"validation": statut})

    print("\n--- Logged Models de l'expérience, par accuracy décroissante ---")
    modeles = mlflow.search_logged_models(
        experiment_ids=[experience.experiment_id],
        filter_string="metrics.accuracy_score > 0",
        order_by=[{"field_name": "metrics.accuracy_score", "ascending": False}],
        output_format="list",
    )
    for modele in modeles:
        accuracy = next(m.value for m in modele.metrics if m.key == "accuracy_score")
        print(f"{modele.name} : {modele.model_id}, accuracy {accuracy:.4f}, "
              f"validation={modele.tags.get('validation', '-')}")


if __name__ == "__main__":
    main()
