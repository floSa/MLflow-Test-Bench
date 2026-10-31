"""
recherche_example.py - Recherche d'hyperparamètres et requêtes sur les runs
===========================================================================

Une recherche produit des dizaines de runs : les imbriquer garde la table lisible,
et mlflow.search_runs() permet de les interroger en Python au lieu de les parcourir :
- mlflow.sklearn.autolog() : un run parent, et un run enfant par combinaison testée
- mlflow.search_runs() : filtre et tri, avec le résultat dans un DataFrame

Exécution:
---------
docker compose exec runner python recherche_example.py

Interface MLflow:
----------------
Expérience "recherche-demo" sur http://localhost:5000 : le run parent se déplie
"""

import mlflow
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn_example import configurer_mlflow, creer_pipeline


def main():
    configurer_mlflow()
    experience = mlflow.set_experiment("recherche-demo")

    wine = load_wine(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
    )

    # log_models=False : seul le meilleur modèle est enregistré, en skops, plus bas
    mlflow.sklearn.autolog(max_tuning_runs=None, log_models=False, log_datasets=False, silent=True)

    # Le préfixe "modele__" vise cette étape du Pipeline ; le scaler reste ajusté
    # dans chaque fold, donc sans voir les données de validation
    grille = {"modele__C": [0.001, 0.01, 0.1, 1, 10], "modele__solver": ["lbfgs", "saga"]}
    recherche = GridSearchCV(creer_pipeline(LogisticRegression(max_iter=5000, random_state=42)), grille, cv=5)

    with mlflow.start_run(run_name="recherche-logreg") as parent:
        recherche.fit(X_train, y_train)
        mlflow.log_metric("test_accuracy", recherche.score(X_test, y_test))
        info = mlflow.sklearn.log_model(recherche.best_estimator_, name="model", input_example=X_train.head(5))

    mlflow.sklearn.autolog(disable=True)

    print(f"Meilleurs paramètres : {recherche.best_params_}")
    print(f"Modèle : {info.model_uri}")

    print("\n--- Les 3 meilleurs runs enfants, avec search_runs ---")
    enfants = mlflow.search_runs(
        experiment_ids=[experience.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
        order_by=["metrics.mean_test_score DESC"],
    )
    colonnes = ["params.modele__C", "params.modele__solver", "metrics.mean_test_score"]
    print(f"{len(enfants)} combinaisons testées")
    print(enfants[colonnes].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
