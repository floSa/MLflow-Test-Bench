"""
genai_example.py - Le parcours complet d'un assistant LLM
=========================================================

Du prompt versionné à l'assistant prêt à servir, avec un modèle local servi par Ollama
(aucune clé payante). Le code de l'assistant est dans assistant_modele.py :
- mlflow.genai.register_prompt() : le prompt est versionné et porte un alias
- mlflow.pyfunc.log_model() : l'assistant devient un Logged Model
- mlflow.openai.autolog() et @mlflow.trace : chaque réponse devient une trace
- mlflow.genai.evaluate() : note les réponses sur un jeu de questions
- @scorer : un critère écrit en Python, sans LLM
- Correctness et Guidelines : des critères notés par un LLM, le "juge"
- @champion : posé seulement si les notes respectent les seuils

Prérequis:
---------
Un serveur Ollama joignable depuis le conteneur runner, avec le modèle
OLLAMA_MODEL installé (voir la section GenAI du tutoriel).

Exécution:
---------
docker compose exec -e OLLAMA_BASE_URL=http://ollama-central:11434/v1 runner python genai_example.py

Interface MLflow:
----------------
Mode "GenAI" sur http://localhost:5000 : menu "Prompts", et dans l'expérience
"genai-assistant" les onglets "Traces" et "Evaluation runs"
"""

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import Correctness, Guidelines, scorer

from assistant_modele import MODELE, PROMPT, repondre
from sklearn_example import configurer_mlflow

JUGE = f"openai:/{MODELE}"

# Une variable s'écrit entre doubles accolades
TEMPLATE = "Réponds en français, en une phrase, d'après cette fiche : {{fiche}}"

# Chaque critère doit atteindre cette note moyenne pour que l'assistant soit promu
SEUILS = {"contient_mot_cle/mean": 1.0, "correctness/mean": 1.0, "en_francais/mean": 1.0}

QUESTIONS = [
    {
        "inputs": {"question": "Combien de classes contient le dataset Wine ?"},
        "expectations": {"expected_facts": ["Le dataset contient 3 classes"], "mots_cles": ["3", "trois"]},
    },
    {
        "inputs": {"question": "Combien de caractéristiques décrivent un vin ?"},
        "expectations": {"expected_facts": ["Un vin est décrit par 13 caractéristiques"], "mots_cles": ["13", "treize"]},
    },
    {
        "inputs": {"question": "Combien d'échantillons contient le dataset Wine ?"},
        "expectations": {"expected_facts": ["Le dataset contient 178 échantillons"], "mots_cles": ["178"]},
    },
]


@scorer
def contient_mot_cle(outputs, expectations):
    """Critère sans LLM : la réponse contient-elle l'un des mots-clés attendus ?"""
    return any(mot in outputs for mot in expectations["mots_cles"])


def versionner_prompt():
    """Crée une version du prompt s'il a changé, et pose l'alias production dessus."""
    actuel = mlflow.genai.load_prompt(f"prompts:/{PROMPT}@latest", allow_missing=True)
    if actuel is None or actuel.template != TEMPLATE:
        actuel = mlflow.genai.register_prompt(PROMPT, TEMPLATE, commit_message="Réponse courte, d'après une fiche")
    mlflow.genai.set_prompt_alias(PROMPT, "production", actuel.version)
    print(f"Prompt {PROMPT} : version {actuel.version}, alias production")
    return actuel


def promouvoir(model_uri, metriques):
    """Enregistre l'assistant, et le désigne @champion s'il respecte les seuils."""
    version = mlflow.register_model(model_uri, "assistant-wine")
    # Un critère absent compte pour 0 : un juge en erreur ne promeut pas l'assistant
    ecarts = [nom for nom, seuil in SEUILS.items() if metriques.get(nom, 0) < seuil]
    if ecarts:
        print(f"Version {version.version} refusée, sous le seuil : {ecarts}")
    else:
        mlflow.MlflowClient().set_registered_model_alias("assistant-wine", "champion", version.version)
        print(f"Version {version.version} désormais @champion")


def main():
    configurer_mlflow()
    mlflow.set_experiment("genai-assistant")

    print("\n--- Prompt ---")
    prompt = versionner_prompt()

    # Le code devient un Logged Model, relié à son prompt. Les traces suivantes s'y rattachent
    info = mlflow.pyfunc.log_model(name="assistant", python_model="assistant_modele.py", prompts=[prompt])
    mlflow.set_active_model(model_id=info.model_id)

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
        model_id=info.model_id,
    )
    print(f"Run d'évaluation : {resultat.run_id}")
    for nom, valeur in sorted(resultat.metrics.items()):
        print(f"{nom} : {valeur:.2f}")

    print("\n--- Détail par question ---")
    for trace in mlflow.search_traces(run_id=resultat.run_id, return_type="list"):
        notes = {a.name: a.value for a in trace.info.assessments if isinstance(a, Feedback)}
        print(f"{trace.data.request}\n-> {trace.data.response}\n   {notes}")

    print("\n--- Promotion ---")
    promouvoir(info.model_uri, resultat.metrics)


if __name__ == "__main__":
    main()
