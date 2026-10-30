"""
autolog_example.py - Auto-logging avec scikit-learn
===================================================

mlflow.sklearn.autolog() enregistre seul les hyperparamètres, les métriques
d'entraînement et le modèle : aucun appel à log_param ou log_metric.

Exécution:
---------
docker compose exec runner python autolog_example.py

Interface MLflow:
----------------
Expérience "autolog-demo" sur http://localhost:5000
"""

import mlflow
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn_example import configurer_mlflow


def main():
    configurer_mlflow()
    mlflow.set_experiment("autolog-demo")

    # À appeler avant fit() : c'est lui qui déclenche l'enregistrement
    mlflow.sklearn.autolog()

    X, y = load_wine(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run(run_name="random-forest-autolog") as run:
        RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

    # Relecture du run pour voir ce qui a été enregistré sans le demander
    data = mlflow.get_run(run.info.run_id).data
    print(f"Run terminé : {run.info.run_id}")
    print(f"Paramètres enregistrés : {len(data.params)}")
    print(f"Métriques enregistrées : {sorted(data.metrics)}")


if __name__ == "__main__":
    main()
