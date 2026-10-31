"""
pytorch_example.py - Un réseau de neurones, le même parcours
============================================================

Le deep learning demande une image plus lourde, séparée du runner, mais MLflow s'utilise
de la même façon :
- une métrique et une image par époque : la boucle d'entraînement reste la vôtre
- mlflow.pytorch.log_model() : le réseau et sa signature
- mlflow.pyfunc.load_model() : la même interface que scikit-learn et XGBoost

Exécution:
---------
docker compose --profile deep up -d --build runner-torch
docker compose exec runner-torch python pytorch_example.py

Interface MLflow:
----------------
Expérience "pytorch-demo" sur http://localhost:5000, onglet "Model metrics" du run
"""

import mlflow
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn

from images_example import matrice_en_image
from sklearn_example import configurer_mlflow

EPOQUES = 10


class Reseau(nn.Module):
    """Petit réseau convolutif pour des images de chiffres de 8 x 8 pixels."""

    def __init__(self):
        super().__init__()
        self.couches = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 4 * 4, 10),
        )

    def forward(self, x):
        return self.couches(x)


def main():
    configurer_mlflow()
    mlflow.set_experiment("pytorch-demo")
    torch.manual_seed(42)

    digits = load_digits()
    # Pixels entre 0 et 1, au format (lot, canal, hauteur, largeur)
    X = (digits.images / 16).astype(np.float32)[:, None, :, :]
    X_train, X_test, y_train, y_test = train_test_split(
        X, digits.target, test_size=0.25, random_state=42, stratify=digits.target
    )
    X_train_t, y_train_t = torch.from_numpy(X_train), torch.from_numpy(y_train)

    reseau = Reseau()
    optimiseur = torch.optim.Adam(reseau.parameters(), lr=0.001)
    perte = nn.CrossEntropyLoss()
    parametres = {"epoques": EPOQUES, "lr": 0.001, "taille_lot": 64}

    with mlflow.start_run(run_name="reseau-chiffres"):
        mlflow.log_params(parametres)

        for epoque in range(EPOQUES):
            reseau.train()
            for lot in torch.randperm(len(X_train_t)).split(parametres["taille_lot"]):
                optimiseur.zero_grad()
                valeur = perte(reseau(X_train_t[lot]), y_train_t[lot])
                valeur.backward()
                optimiseur.step()

            reseau.eval()
            with torch.no_grad():
                y_pred = reseau(torch.from_numpy(X_test)).argmax(dim=1).numpy()
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metrics({"loss": valeur.item(), "accuracy": accuracy}, step=epoque)
            mlflow.log_image(matrice_en_image(y_test, y_pred), key="matrice_confusion", step=epoque)
            print(f"Époque {epoque} : loss {valeur.item():.3f}, accuracy {accuracy:.4f}")

        info = mlflow.pytorch.log_model(
            reseau,
            name="model",
            input_example=X_train[:2],
            # torch vient de l'index CPU de PyTorch : à déclarer pour réinstaller le modèle ailleurs
            extra_pip_requirements=["--extra-index-url https://download.pytorch.org/whl/cpu"],
        )

    # Directement @champion pour la démonstration : promotion_example.py porte sur les
    # données Wine. Un vrai parcours donnerait à ce modèle son propre contrôle
    version = mlflow.register_model(info.model_uri, "chiffres-pytorch")
    mlflow.MlflowClient().set_registered_model_alias("chiffres-pytorch", "champion", version.version)
    print(f"\nchiffres-pytorch v{version.version} désormais @champion")

    # Le code qui consomme le modèle ne dépend toujours pas du framework
    charge = mlflow.pyfunc.load_model("models:/chiffres-pytorch@champion")
    print(f"Prédictions : {charge.predict(X_test[:5]).argmax(axis=1).tolist()}, réel : {y_test[:5].tolist()}")


if __name__ == "__main__":
    main()
