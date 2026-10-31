"""
pyfunc_example.py - Un modèle personnalisé
==========================================

Quand un framework n'est pas pris en charge, ou qu'il faut du code autour du modèle
(pré-traitement, règles métier, sortie sur mesure), pyfunc enveloppe n'importe quel
code Python derrière la même interface et la même API :
- python_model="modele_maison.py" : le code est enregistré tel quel
- artifacts : les fichiers dont le code a besoin, ici un classifieur entraîné
- params : un réglage modifiable à chaque requête

Exécution:
---------
docker compose exec runner python pyfunc_example.py

Interface MLflow:
----------------
Expérience "pyfunc-demo" sur http://localhost:5000, onglet "Models"
"""

import tempfile

import mlflow
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn_example import configurer_mlflow, creer_pipeline


def main():
    configurer_mlflow()
    mlflow.set_experiment("pyfunc-demo")

    wine = load_wine(as_frame=True)
    X_train, X_test, y_train, _ = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
    )
    classifieur = creer_pipeline(LogisticRegression(max_iter=1000, random_state=42)).fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as dossier:
        chemin = f"{dossier}/classifieur"
        mlflow.sklearn.save_model(classifieur, chemin)

        with mlflow.start_run(run_name="classifieur-explique"):
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model="modele_maison.py",
                artifacts={"classifieur": chemin},
                # Le dictionnaire déclare le paramètre et sa valeur par défaut dans la signature
                input_example=(X_train.head(3), {"seuil": 0.8}),
            )

    # Directement @champion pour la démonstration : promotion_example.py attend une
    # classe en sortie. Un vrai parcours lui donnerait son propre contrôle
    version = mlflow.register_model(info.model_uri, "wine-classifier-explique")
    mlflow.MlflowClient().set_registered_model_alias("wine-classifier-explique", "champion", version.version)
    print(f"wine-classifier-explique v{version.version} désormais @champion")

    modele = mlflow.pyfunc.load_model("models:/wine-classifier-explique@champion")
    print("\n--- Seuil par défaut (0,8) ---")
    print(modele.predict(X_test.head(4)).to_string())
    print("\n--- Seuil de 0,999, passé à la requête ---")
    print(modele.predict(X_test.head(4), params={"seuil": 0.999}).to_string())


if __name__ == "__main__":
    main()
