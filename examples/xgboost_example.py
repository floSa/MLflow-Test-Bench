"""
xgboost_example.py - Un autre framework, le même parcours
=========================================================

Changer de framework ne change que la ligne qui enregistre le modèle : le registre,
les alias, la promotion et le service restent identiques.
- mlflow.xgboost.autolog() : paramètres et courbes d'apprentissage, sans code
- mlflow.xgboost.log_model() : le modèle au format natif de XGBoost
- mlflow.pyfunc.load_model() : la même interface que pour scikit-learn

Exécution:
---------
docker compose exec runner python xgboost_example.py
docker compose exec runner python promotion_example.py --modele wine-xgboost

Interface MLflow:
----------------
Expérience "xgboost-demo" sur http://localhost:5000, onglet "Model metrics" du run
"""

import mlflow
import sklearn
import xgboost
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn_example import configurer_mlflow


def main():
    configurer_mlflow()
    mlflow.set_experiment("xgboost-demo")

    # Le même découpage que sklearn_example.py, puis une validation pour l'arrêt anticipé
    wine = load_wine(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # log_models=False : le modèle est enregistré plus bas, avec son exemple d'entrée
    mlflow.xgboost.autolog(log_models=False, log_datasets=False)

    with mlflow.start_run(run_name="xgboost") as run:
        modele = XGBClassifier(n_estimators=200, max_depth=3, early_stopping_rounds=10, random_state=42)
        modele.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

        info = mlflow.xgboost.log_model(
            modele,
            name="model",
            input_example=X_train.head(5),
            # Sans cette liste, MLflow ajoute le paquet xgboost, et sa dépendance GPU,
            # à xgboost-cpu : l'image du modèle pèse 800 Mo de plus
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"xgboost-cpu=={xgboost.__version__}",
                f"scikit-learn=={sklearn.__version__}",
            ],
        )

    print(f"Arbres retenus par l'arrêt anticipé : {modele.best_iteration + 1} sur 200")

    client = mlflow.MlflowClient()
    version = mlflow.register_model(info.model_uri, "wine-xgboost")
    client.set_registered_model_alias("wine-xgboost", "challenger", version.version)
    print(f"wine-xgboost v{version.version} désormais @challenger (run {run.info.run_id})")

    # Le code qui consomme le modèle ne sait pas de quel framework il vient
    charge = mlflow.pyfunc.load_model("models:/wine-xgboost@challenger")
    print(f"Prédictions : {charge.predict(X_test.head(5)).tolist()}, réel : {y_test.head(5).tolist()}")


if __name__ == "__main__":
    main()
