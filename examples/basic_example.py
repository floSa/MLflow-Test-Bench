"""
basic_example.py - Logging manuel avec MLflow
=============================================

Le strict minimum : un run avec des paramètres, une métrique suivie
par étape et des tags, sans entraîner de vrai modèle.

Exécution:
---------
docker compose exec runner python basic_example.py

Interface MLflow:
----------------
Expérience "tutoriel-basique" sur http://localhost:5000
"""

import mlflow

from sklearn_example import configurer_mlflow


def main():
    configurer_mlflow()
    mlflow.set_experiment("tutoriel-basique")

    with mlflow.start_run(run_name="premier-run") as run:
        # Paramètres : les valeurs d'entrée, loggées une seule fois
        mlflow.log_param("algorithm", "random_forest")
        mlflow.log_param("n_estimators", 100)

        # Métrique : plusieurs valeurs au fil des étapes (donne une courbe)
        for step, loss in enumerate([0.9, 0.5, 0.3, 0.2, 0.15]):
            mlflow.log_metric("loss", loss, step=step)
        mlflow.log_metric("accuracy", 0.92)

        # Tags : métadonnées libres, utiles pour filtrer
        mlflow.set_tag("auteur", "tutoriel")

    print(f"Run terminé : {run.info.run_id}")


if __name__ == "__main__":
    main()
