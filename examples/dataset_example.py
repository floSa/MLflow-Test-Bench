"""
dataset_example.py - Logging d'un dataset
=========================================

mlflow.log_input() relie un run aux données qui l'ont nourri : nom, source,
empreinte (digest) et schéma, sans copier les données elles-mêmes.

Exécution:
---------
docker compose exec runner python dataset_example.py

Interface MLflow:
----------------
Expérience "dataset-demo" sur http://localhost:5000
"""

import mlflow
import mlflow.data
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from sklearn_example import configurer_mlflow


def main():
    configurer_mlflow()
    mlflow.set_experiment("dataset-demo")

    df = load_wine(as_frame=True).frame
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])

    with mlflow.start_run(run_name="wine-train-test") as run:
        # context : rôle du dataset dans ce run ("training", "validation", "test")
        for df_split, context in [(df_train, "training"), (df_test, "test")]:
            dataset = mlflow.data.from_pandas(
                df_split,
                source="sklearn.datasets.load_wine",
                name=f"wine_{context}",
            )
            mlflow.log_input(dataset, context=context)

    # Relecture du run pour vérifier ce qui a été enregistré
    print(f"Run terminé : {run.info.run_id}")
    for entry in mlflow.get_run(run.info.run_id).inputs.dataset_inputs:
        print(f"Dataset {entry.dataset.name} : digest {entry.dataset.digest}, {entry.tags[0].value}")


if __name__ == "__main__":
    main()
