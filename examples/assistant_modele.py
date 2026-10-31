"""
assistant_modele.py - Le code de l'assistant LLM
================================================

Un petit assistant répond à des questions sur le dataset Wine à partir de fiches.
MLflow enregistre ce fichier tel quel ("models from code") : l'assistant se sert ensuite
comme n'importe quel modèle. Il est utilisé par genai_example.py.

Le LLM est servi par Ollama, en local (aucune clé payante), et le prompt vient du
registre de prompts : le changer ne demande pas de redéployer le code.
"""

import os

import mlflow
from mlflow.entities import SpanType
from openai import OpenAI

MODELE = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
# Les prompts et les modèles partagent les mêmes noms : celui-ci doit rester distinct
PROMPT = "assistant-wine-prompt"

# Le juge de genai_example.py passe par LiteLLM, qui lit OPENAI_API_BASE ; le client
# OpenAI lit OPENAI_BASE_URL. Ollama expose une API compatible et ignore la clé.
URL_OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
os.environ["OPENAI_BASE_URL"] = URL_OLLAMA
os.environ["OPENAI_API_BASE"] = URL_OLLAMA
os.environ["OPENAI_API_KEY"] = "ollama"

FICHES = {
    "classes": "Le dataset Wine contient 3 classes de vins : class_0, class_1 et class_2.",
    "caractéristiques": "Chaque vin est décrit par 13 caractéristiques chimiques, dont l'alcool et le magnésium.",
    "échantillons": "Le dataset Wine contient 178 échantillons.",
}

# Ajoute à la trace un span par appel au LLM : messages, réponse, jetons
mlflow.openai.autolog()


@mlflow.trace(span_type=SpanType.RETRIEVER)
def chercher_fiche(question):
    """Renvoie la fiche dont le titre apparaît dans la question."""
    return next((fiche for titre, fiche in FICHES.items() if titre in question.lower()), "")


@mlflow.trace
def repondre(question):
    """Répond à une question : la fiche d'abord, le LLM ensuite (deux spans de la trace)."""
    fiche = chercher_fiche(question)
    # L'alias désigne la version en production ; MLflow la garde en cache
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT}@production")
    reponse = OpenAI().chat.completions.create(
        model=MODELE,
        # Le plus reproductible possible : deux évaluations restent comparables
        temperature=0,
        messages=[
            {"role": "system", "content": prompt.format(fiche=fiche)},
            {"role": "user", "content": question},
        ],
    )
    return reponse.choices[0].message.content


class Assistant(mlflow.pyfunc.PythonModel):
    """L'interface de service : une liste de questions, une liste de réponses."""

    # Les annotations donnent la signature, sans appeler le LLM à l'enregistrement
    def predict(self, context, model_input: list[str], params=None) -> list[str]:
        return [repondre(question) for question in model_input]


mlflow.models.set_model(Assistant())
