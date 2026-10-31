"""
serving_example.py - Interroger le modèle en production
=======================================================

Les deux façons d'utiliser @champion, avec le même résultat :
- en ligne : l'API du service model-server, requête par requête
- par lot : mlflow.pyfunc.load_model(), sur tout un fichier

Le script montre aussi que l'API refuse une requête qui ne respecte pas la
signature du modèle.

Prérequis:
---------
docker compose exec runner python sklearn_example.py
docker compose exec runner python promotion_example.py
docker compose --profile serving up -d --wait model-server

Exécution:
---------
docker compose exec runner python serving_example.py
"""

import os

import mlflow
import requests
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn_example import configurer_mlflow

URL = os.getenv("MODEL_SERVER_URL", "http://model-server:8080")


def predire_en_ligne(X):
    """Envoie un DataFrame à l'API et renvoie ses prédictions."""
    corps = {"dataframe_split": X.to_dict(orient="split", index=False)}
    reponse = requests.post(f"{URL}/invocations", json=corps, timeout=30)
    reponse.raise_for_status()
    return reponse.json()["predictions"]


def main():
    configurer_mlflow()

    wine = load_wine(as_frame=True)
    _, X_test, _, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
    )

    print("\n--- État du service ---")
    # /ping répond 200 quand le modèle est chargé : c'est le healthcheck
    print(f"/ping : {requests.get(f'{URL}/ping', timeout=5).status_code}")
    print(f"/version : MLflow {requests.get(f'{URL}/version', timeout=5).text}")

    print("\n--- Prédiction en ligne ---")
    en_ligne = predire_en_ligne(X_test)
    print(f"5 premières prédictions : {en_ligne[:5]}")
    print(f"5 premières valeurs réelles : {y_test.head(5).tolist()}")

    print("\n--- Requête invalide : une colonne manque ---")
    corps = {"dataframe_split": X_test.drop(columns="alcohol").to_dict(orient="split", index=False)}
    reponse = requests.post(f"{URL}/invocations", json=corps, timeout=30)
    print(f"HTTP {reponse.status_code} : {reponse.json()['message'].split('Error: ')[-1]}")

    print("\n--- Prédiction par lot ---")
    # Même alias, même signature, mais sans serveur : adapté à un traitement planifié
    modele = mlflow.pyfunc.load_model("models:/wine-classifier@champion")
    par_lot = modele.predict(X_test)
    print(f"Accuracy sur {len(X_test)} lignes : {accuracy_score(y_test, par_lot):.4f}")
    print(f"Identique à l'API : {list(par_lot) == en_ligne}")


if __name__ == "__main__":
    main()
