"""
images_example.py - Suivre des images et des tableaux au fil de l'entraînement
==============================================================================

Une métrique résume un entraînement en un nombre ; une image montre où il se trompe :
- mlflow.log_image(key=..., step=...) : une image par étape, avec un curseur dans l'interface
- mlflow.log_figure() : une figure unique, en simple artefact
- mlflow.log_table() : un tableau, consultable dans l'onglet "Artifacts" et relisible en Python

Exécution:
---------
docker compose exec runner python images_example.py

Interface MLflow:
----------------
Expérience "images-demo" sur http://localhost:5000, onglet "Model metrics" du run
"""

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn_example import configurer_mlflow

EPOQUES = 8


def matrice_en_image(y_true, y_pred):
    """Dessine la matrice de confusion et la renvoie en tableau de pixels pour log_image."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=range(10))).plot(ax=ax, colorbar=False)
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return pixels


def main():
    configurer_mlflow()
    mlflow.set_experiment("images-demo")

    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=42, stratify=digits.target
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    modele = SGDClassifier(loss="log_loss", random_state=42)
    # Entraînement volontairement lent, pour que la matrice évolue d'une époque à l'autre
    lots = np.array_split(np.arange(len(X_train)), EPOQUES * 4)

    with mlflow.start_run(run_name="chiffres-sgd") as run:
        for epoque in range(EPOQUES):
            for lot in lots[epoque * 4:(epoque + 1) * 4]:
                modele.partial_fit(X_train[lot], y_train[lot], classes=np.arange(10))
            y_pred = modele.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy, step=epoque)
            # La même clé à chaque étape : l'interface affiche un curseur
            mlflow.log_image(matrice_en_image(y_test, y_pred), key="matrice_confusion", step=epoque)
            print(f"Époque {epoque} : accuracy {accuracy:.4f}")

        # Le détail des erreurs finales, en tableau
        erreurs = pd.DataFrame({"reel": y_test, "predit": y_pred}).query("reel != predit")
        mlflow.log_table(erreurs, artifact_file="erreurs.json")

        # Les images incertaines, en figure unique
        fig, axes = plt.subplots(1, 6, figsize=(9, 2))
        indices = np.argsort(modele.predict_proba(X_test).max(axis=1))[:6]
        for ax, i in zip(axes, indices):
            ax.imshow(X_test[i].reshape(8, 8), cmap="gray_r")
            ax.set_title(f"{y_pred[i]} / {y_test[i]}", fontsize=9)
            ax.axis("off")
        mlflow.log_figure(fig, "chiffres_incertains.png")
        plt.close(fig)

    relu = mlflow.load_table("erreurs.json", run_ids=[run.info.run_id])
    print(f"\nErreurs sur le test : {len(relu)} sur {len(y_test)}")
    print(f"Run : {run.info.run_id}")


if __name__ == "__main__":
    main()
