"""
genai_example.py - Tracing et évaluation d'un assistant LLM
===========================================================

Un petit assistant répond à des questions sur le dataset Wine à partir de
fiches, avec un modèle local servi par Ollama (aucune clé payante) :
- mlflow.openai.autolog() et @mlflow.trace : chaque réponse devient une trace
- mlflow.genai.evaluate() : note les réponses sur un jeu de questions
- @scorer : un critère écrit en Python, sans LLM
- Correctness et Guidelines : des critères notés par un LLM, le "juge"

Prérequis:
---------
Un serveur Ollama joignable depuis le conteneur runner, avec le modèle
OLLAMA_MODEL installé (voir la section GenAI du tutoriel).

Exécution:
---------
docker compose exec -e OLLAMA_BASE_URL=http://ollama-central:11434/v1 runner python genai_example.py

Interface MLflow:
----------------
Expérience "genai-assistant" sur http://localhost:5000, onglets "Traces" et "Evaluation runs"
"""

import os

import mlflow
from mlflow.entities import Feedback, SpanType
from mlflow.genai.scorers import Correctness, Guidelines, scorer
from openai import OpenAI

from sklearn_example import configurer_mlflow

MODELE = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# Le juge passe par LiteLLM, qui lit OPENAI_API_BASE ; le client OpenAI lit
# OPENAI_BASE_URL. Ollama expose une API compatible et ignore la clé.
URL_OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
os.environ["OPENAI_BASE_URL"] = URL_OLLAMA
os.environ["OPENAI_API_BASE"] = URL_OLLAMA
os.environ["OPENAI_API_KEY"] = "ollama"
JUGE = f"openai:/{MODELE}"

FICHES = {
    "classes": "Le dataset Wine contient 3 classes de vins : class_0, class_1 et class_2.",
    "caractéristiques": "Chaque vin est décrit par 13 caractéristiques chimiques, dont l'alcool et le magnésium.",
    "échantillons": "Le dataset Wine contient 178 échantillons.",
}

QUESTIONS = [
    {
        "inputs": {"question": "Combien de classes contient le dataset Wine ?"},
        "expectations": {"expected_facts": ["Le dataset contient 3 classes"], "mot_cle": "3"},
    },
    {
        "inputs": {"question": "Combien de caractéristiques décrivent un vin ?"},
        "expectations": {"expected_facts": ["Un vin est décrit par 13 caractéristiques"], "mot_cle": "13"},
    },
    {
        "inputs": {"question": "Combien d'échantillons contient le dataset Wine ?"},
        "expectations": {"expected_facts": ["Le dataset contient 178 échantillons"], "mot_cle": "178"},
    },
]


@mlflow.trace(span_type=SpanType.RETRIEVER)
def chercher_fiche(question):
    """Renvoie la fiche dont le titre apparaît dans la question."""
    return next((fiche for titre, fiche in FICHES.items() if titre in question.lower()), "")


@mlflow.trace
def repondre(question):
    """Répond à une question : la fiche d'abord, le LLM ensuite (deux spans de la trace)."""
    fiche = chercher_fiche(question)
    reponse = OpenAI().chat.completions.create(
        model=MODELE,
        messages=[
            {"role": "system", "content": f"Réponds en français, en une phrase, d'après cette fiche : {fiche}"},
            {"role": "user", "content": question},
        ],
    )
    return reponse.choices[0].message.content


@scorer
def contient_mot_cle(outputs, expectations):
    """Critère sans LLM : la réponse contient-elle le mot-clé attendu ?"""
    return expectations["mot_cle"] in outputs


def main():
    configurer_mlflow()
    mlflow.set_experiment("genai-assistant")

    # Ajoute à la trace un span par appel au LLM : messages, réponse, jetons
    mlflow.openai.autolog()

    print("\n--- Une question, une trace ---")
    question = QUESTIONS[0]["inputs"]["question"]
    print(f"{question}\n-> {repondre(question)}")
    print(f"Trace : {mlflow.get_last_active_trace_id()}")

    print("\n--- Évaluation sur le jeu de questions ---")
    resultat = mlflow.genai.evaluate(
        data=QUESTIONS,
        predict_fn=repondre,
        scorers=[
            contient_mot_cle,
            Correctness(model=JUGE),
            Guidelines(
                name="en_francais",
                guidelines="La réponse doit être rédigée en français.",
                model=JUGE,
            ),
        ],
    )
    print(f"Run d'évaluation : {resultat.run_id}")
    for nom, valeur in sorted(resultat.metrics.items()):
        print(f"{nom} : {valeur:.2f}")
    
    print("\n--- Détail par question ---")
    for trace in mlflow.search_traces(run_id=resultat.run_id, return_type="list"):
        notes = {a.name: a.value for a in trace.info.assessments if isinstance(a, Feedback)}
        print(f"{trace.data.request}\n-> {trace.data.response}\n   {notes}")


if __name__ == "__main__":
    main()
