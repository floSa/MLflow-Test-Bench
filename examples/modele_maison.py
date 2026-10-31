"""
modele_maison.py - Le code d'un modèle personnalisé
===================================================

MLflow enregistre ce fichier tel quel avec le modèle ("models from code") : rien
n'est sérialisé en pickle, et le code se relit comme n'importe quel fichier.
Il est utilisé par pyfunc_example.py.
"""

import mlflow
import pandas as pd

CLASSES = ["class_0", "class_1", "class_2"]


class ClassifieurExplique(mlflow.pyfunc.PythonModel):
    """Enveloppe un classifieur : renvoie le nom de la classe et la confiance."""

    def load_context(self, context):
        # Appelé une fois au chargement : context.artifacts donne les fichiers du modèle
        self.classifieur = mlflow.sklearn.load_model(context.artifacts["classifieur"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        # params : un réglage par requête, sans réentraîner ni réenregistrer le modèle
        seuil = (params or {}).get("seuil", 0.8)
        probabilites = self.classifieur.predict_proba(model_input)
        confiance = probabilites.max(axis=1)
        return pd.DataFrame({
            "classe": [CLASSES[i] for i in probabilites.argmax(axis=1)],
            "confiance": confiance.round(4),
            "a_verifier": confiance < seuil,
        })


mlflow.models.set_model(ClassifieurExplique())
