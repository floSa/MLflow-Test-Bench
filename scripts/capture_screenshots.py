# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["playwright"]
# ///
"""
capture_screenshots.py - Régénère les captures de la documentation
==================================================================

Prérequis:
---------
1. La stack tourne : docker compose up -d
   Pour les captures PyTorch et GenAI, lancer d'abord pytorch_example.py et genai_example.py
2. Chromium est installé : uv run --with playwright playwright install chromium

Exécution:
---------
uv run scripts/capture_screenshots.py

Si l'interface n'est pas sur le port 5000 : MLFLOW_UI=http://localhost:15000 uv run ...

Les images sont écrites dans documentation/images/. Les pastilles numérotées
renvoient aux légendes du tutoriel et du manuel utilisateur.
"""

import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

MLFLOW = os.getenv("MLFLOW_UI", "http://localhost:5000")
FILER = "http://127.0.0.1:8888"
MASTER = "http://127.0.0.1:9333"
IMAGES = Path(__file__).resolve().parent.parent / "documentation" / "images"
EXEMPLES = {
    "comparaison-modeles": "sklearn_example.py",
    "tutoriel-basique": "basic_example.py",
    "autolog-demo": "autolog_example.py",
    "dataset-demo": "dataset_example.py",
    "evaluation-modeles": "evaluation_example.py",
    "promotion-modeles": "promotion_example.py",
    "recherche-demo": "recherche_example.py",
    "images-demo": "images_example.py",
    "pyfunc-demo": "pyfunc_example.py",
    "xgboost-demo": "xgboost_example.py",
}

# Interface en mode "Model training", sans le panneau de l'assistant qui s'ouvre
# à la première visite ; pytorch_example.py a besoin de son image, et
# genai_example.py d'Ollama : ils se lancent à part
PREFERENCES = """() => {
    localStorage.setItem('mlflow.workflowType_v1', '"machine_learning"');
    localStorage.setItem('mlflow.assistant.fab.autoOpenedOnFirstLoad_v1', 'true');
    localStorage.setItem('mlflow.assistant.panelOpen_v1', 'false');
}"""

# Dessine un cadre et une pastille numérotée autour de chaque élément ciblé
ANNOTER = """([x, y, w, h, n]) => {
    const cadre = document.createElement('div');
    cadre.style.cssText = `position:fixed;left:${x - 4}px;top:${y - 4}px;width:${w + 8}px;
        height:${h + 8}px;border:3px solid #d92d20;border-radius:6px;z-index:99998;pointer-events:none`;
    const pastille = document.createElement('div');
    pastille.textContent = n;
    pastille.style.cssText = `position:fixed;left:${x - 16}px;top:${y - 16}px;width:24px;height:24px;
        border-radius:50%;background:#d92d20;color:#fff;font:bold 14px/24px sans-serif;
        text-align:center;z-index:99999;pointer-events:none`;
    document.body.append(cadre, pastille);
}"""


def api(chemin, corps=None):
    donnees = json.dumps(corps).encode() if corps is not None else None
    requete = urllib.request.Request(
        f"{MLFLOW}/api/2.0/mlflow/{chemin}", donnees, {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(requete))


def experiences():
    return {e["name"]: e["experiment_id"] for e in api("experiments/search", {"max_results": 50})["experiments"]}


def runs(experiment_id):
    return api("runs/search", {"experiment_ids": [experiment_id]})["runs"]


def preparer_donnees():
    """Lance les exemples dont l'expérience n'existe pas encore."""
    for nom, script in EXEMPLES.items():
        if nom not in experiences():
            subprocess.run(["docker", "compose", "exec", "-T", "runner", "python", script], check=True)


def capturer(page, url, nom, cibles=(), attente=3500, avant=None):
    page.goto("about:blank")
    page.goto(url)
    page.wait_for_timeout(attente)
    if avant:
        avant(page)
    for numero, cible in enumerate(cibles, start=1):
        locator = page.get_by_text(cible, exact=False).first if isinstance(cible, str) else cible(page)
        boite = locator.bounding_box()
        if boite is None:
            raise SystemExit(f"{nom} : cible introuvable : {cible}")
        page.evaluate(ANNOTER, [boite["x"], boite["y"], boite["width"], boite["height"], numero])
    page.screenshot(path=IMAGES / f"{nom}.png")
    print(f"{nom}.png")


def fermer_promotion(page):
    """Ferme les fenêtres de découverte affichées à la première visite d'une page."""
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)


def deplier_parent(pg):
    """Déplie le run parent pour montrer ses runs enfants."""
    pg.locator("button[aria-label*='xpand'], .ag-row [data-testid*='expand'], .ag-row button").first.click()
    pg.wait_for_timeout(800)


def agrandir_image(pg):
    """Ouvre le graphique d'images en plein écran, à la dernière étape, puis l'image."""
    # L'icône plein écran suit le titre du graphique ; elle n'a pas de nom accessible
    titre = pg.get_by_text("matrice_confusion", exact=True).first.bounding_box()
    pg.mouse.click(titre["x"] + titre["width"] + 16, titre["y"] + 16)
    pg.wait_for_timeout(1500)
    fenetre = pg.get_by_role("dialog")
    # Le curseur ne réagit qu'à la souris : un clic au bout de son rail, avant le champ
    poignee = fenetre.get_by_role("slider").bounding_box()
    champ = fenetre.locator("input").last.bounding_box()
    pg.mouse.click(champ["x"] - 28, poignee["y"] + poignee["height"] / 2)
    pg.wait_for_timeout(2500)
    fenetre.locator("img").first.click()
    pg.wait_for_timeout(2000)


def main():
    preparer_donnees()
    IMAGES.mkdir(parents=True, exist_ok=True)
    ids = experiences()
    comparaison = ids["comparaison-modeles"]
    evaluation = ids["evaluation-modeles"]
    champion = api("registered-models/alias?name=wine-classifier&alias=champion")["model_version"]
    modele_champion = champion["source"].split("/")[-1]
    tous = [r["info"]["run_id"] for r in runs(comparaison)]
    run_basique = runs(ids["tutoriel-basique"])[0]["info"]["run_id"]
    run_autolog = runs(ids["autolog-demo"])[0]["info"]["run_id"]
    run_dataset = runs(ids["dataset-demo"])[0]["info"]["run_id"]
    run_candidat = next(r["info"]["run_id"] for r in runs(evaluation) if r["info"]["run_name"] == "candidat")
    url_run = f"{MLFLOW}/#/experiments/{comparaison}/runs/{champion['run_id']}"
    url_stockage = f"{FILER}/buckets/mlflow-artifacts/{comparaison}"

    with sync_playwright() as p:
        navigateur = p.chromium.launch()
        page = navigateur.new_page(viewport={"width": 1440, "height": 760})
        page.goto(MLFLOW)
        page.evaluate(PREFERENCES)

        capturer(page, f"{MLFLOW}/#/experiments/{comparaison}/runs", "01-table-des-runs", [
            lambda pg: pg.get_by_role("link", name="comparaison-modeles").first,
            lambda pg: pg.get_by_role("link", name="Random Forest Large"),
            lambda pg: pg.get_by_text("wine_training_data").first,
            lambda pg: pg.get_by_role("link", name="wine-classifier").first,
        ])
        
        def selectionner(pg):
            lignes = pg.locator(".ag-row .ag-checkbox-input")
            for i in range(lignes.count()):
                if lignes.nth(i).is_visible():
                    lignes.nth(i).check(force=True)
            pg.wait_for_timeout(500)

        capturer(page, f"{MLFLOW}/#/experiments/{comparaison}/runs", "02-selection-pour-comparer", [
            lambda pg: pg.get_by_role("button", name="Compare"),
        ], avant=selectionner)

        page.set_viewport_size({"width": 1440, "height": 1700})
        capturer(page, url_run, "03-run-overview", [
            lambda pg: pg.get_by_role("heading", name="Datasets"),
            lambda pg: pg.get_by_role("heading", name="Tags"),
            lambda pg: pg.get_by_role("heading", name="Registered models"),
            lambda pg: pg.get_by_role("heading", name="Metrics"),
            lambda pg: pg.get_by_role("heading", name="Parameters"),
            lambda pg: pg.get_by_role("heading", name="Logged models"),
        ])
        capturer(page, f"{MLFLOW}/#/experiments/{ids['autolog-demo']}/runs/{run_autolog}", "04-run-autolog", [
            lambda pg: pg.get_by_role("heading", name="Metrics"),
            lambda pg: pg.get_by_role("heading", name="Parameters"),
        ])
        page.set_viewport_size({"width": 1440, "height": 760})
        capturer(page, f"{MLFLOW}/#/experiments/{ids['dataset-demo']}/runs/{run_dataset}", "05-run-datasets", [
            lambda pg: pg.get_by_role("heading", name="Datasets"),
        ])
        capturer(page, f"{MLFLOW}/#/experiments/{ids['tutoriel-basique']}/runs/{run_basique}/model-metrics",
                 "06-courbe-de-metrique", [lambda pg: pg.get_by_text("loss", exact=True).first])
        capturer(page, f"{url_run}/artifacts/confusion_matrix.png", "07-run-artefacts", [
            lambda pg: pg.get_by_text("confusion_matrix.png").first,
            lambda pg: pg.get_by_text("s3://mlflow-artifacts"),
        ])

        def axes_parlants(pg):
            """Remplace les axes par défaut par le modèle et ses scores."""
            pg.get_by_role("button", name="Clear All").click()
            for selecteur, valeurs in [(0, ["model_type"]), (1, ["cv_mean", "test_accuracy"])]:
                champ = pg.locator(".ant-select-selection-search-input, .du-bois-light-select-selection-search-input").nth(selecteur)
                for valeur in valeurs:
                    champ.click(force=True)
                    champ.fill(valeur)
                    pg.wait_for_timeout(300)
                    champ.press("Enter")
                    pg.wait_for_timeout(500)
                pg.keyboard.press("Escape")
            pg.mouse.click(1000, 650)
            pg.wait_for_timeout(1500)

        ids_json = urllib.parse.quote(json.dumps(tous))
        page.set_viewport_size({"width": 1440, "height": 1000})
        capturer(page, f'{MLFLOW}/#/compare-runs?runs={ids_json}&experiments=["{comparaison}"]', "08-comparaison", [
            lambda pg: pg.get_by_text("Parallel Coordinates Plot"),
        ], attente=5000, avant=axes_parlants)
        page.set_viewport_size({"width": 1440, "height": 760})
        capturer(page, f"{MLFLOW}/#/models", "09-registre-des-modeles", [
            lambda pg: pg.get_by_role("link", name="wine-classifier", exact=True),
            lambda pg: pg.locator("tr, [role=row]").filter(has_text="wine-classifier").first.get_by_text("challenger"),
        ], avant=fermer_promotion)
        capturer(page, f"{MLFLOW}/#/models/wine-classifier/versions/{champion['version']}", "10-version-du-modele", [
            lambda pg: pg.get_by_text("Source Run"),
            lambda pg: pg.get_by_text("Source Model"),
            lambda pg: pg.get_by_text("champion").first,
        ], avant=fermer_promotion)
        capturer(page, f"{url_stockage}/{champion['run_id']}/artifacts/",
                 "11-stockage-seaweedfs", [lambda pg: pg.get_by_role("link", name="confusion_matrix.png").first])
        capturer(page, f"{FILER}/buckets/mlflow-artifacts/", "12-seaweedfs-bucket", [
            lambda pg: pg.get_by_role("link", name="mlflow-artifacts"),
            lambda pg: pg.get_by_role("link", name=comparaison, exact=True).last,
        ], attente=1500)
        capturer(page, f"{url_stockage}/", "13-seaweedfs-runs", [
            lambda pg: pg.get_by_role("link", name="models", exact=True),
            lambda pg: pg.get_by_role("link", name=champion["run_id"]),
        ], attente=1500)
        capturer(page, f"{url_stockage}/models/{modele_champion}/artifacts/", "14-seaweedfs-modele", [
            lambda pg: pg.get_by_role("link", name="MLmodel"),
            lambda pg: pg.get_by_role("link", name="model.skops"),
        ], attente=1500)
        capturer(page, MASTER, "15-seaweedfs-cluster", [
            lambda pg: pg.get_by_text("Cluster status"),
            lambda pg: pg.get_by_role("heading", name="Topology"),
        ], attente=1500)

        page.set_viewport_size({"width": 2100, "height": 500})
        capturer(page, f"{MLFLOW}/#/experiments/{evaluation}/models", "16-logged-models", [
            lambda pg: pg.get_by_role("link", name="candidat").first,
            lambda pg: pg.get_by_text("accuracy_score").first,
        ])
        modele_candidat = page.get_by_role("link", name="candidat").first.get_attribute("href").rstrip("/").split("/")[-1]
        page.set_viewport_size({"width": 1440, "height": 1100})
        capturer(page, f"{MLFLOW}/#/experiments/{evaluation}/models/{modele_candidat}", "17-logged-model", [
            lambda pg: pg.get_by_role("heading", name="Metrics"),
            lambda pg: pg.get_by_text("wine_test").first,
        ])
        page.set_viewport_size({"width": 1440, "height": 760})
        capturer(page, f"{MLFLOW}/#/experiments/{evaluation}/runs/{run_candidat}/artifacts/confusion_matrix.png",
                 "18-graphiques-evaluation", [lambda pg: pg.get_by_text("confusion_matrix.png").first])
        capturer(page, f"{MLFLOW}/#/experiments/{evaluation}/runs/{run_candidat}/system-metrics",
                 "19-metriques-systeme", [lambda pg: pg.get_by_text("cpu_utilization_percentage").first])

        capturer(page, f"{MLFLOW}/#/models/wine-classifier", "24-versions-et-alias", [
            lambda pg: pg.get_by_text("champion").first,
            lambda pg: pg.get_by_text("challenger").first,
        ], avant=fermer_promotion)
        capturer(page, f"{MLFLOW}/#/models/wine-classifier/versions/2", "25-version-refusee", [
            lambda pg: pg.get_by_text("validation", exact=True).first,
            lambda pg: pg.get_by_text("refuse", exact=True).first,
        ], avant=fermer_promotion)
        
        recherche = ids["recherche-demo"]
        capturer(page, f"{MLFLOW}/#/experiments/{recherche}/runs", "26-runs-imbriques", [
            lambda pg: pg.get_by_role("link", name="recherche-logreg").first,
        ], avant=deplier_parent)
        
        images = ids["images-demo"]
        run_images = runs(images)[0]["info"]["run_id"]
        page.set_viewport_size({"width": 1440, "height": 1000})
        capturer(page, f"{MLFLOW}/#/experiments/{images}/runs/{run_images}/model-metrics", "27-images-par-etape",
                 attente=6000, avant=agrandir_image)
        page.set_viewport_size({"width": 1440, "height": 760})
        capturer(page, f"{MLFLOW}/#/experiments/{images}/runs/{run_images}/artifacts/erreurs.json", "28-table-erreurs", [
            lambda pg: pg.get_by_text("erreurs.json").first,
        ])
        run_xgboost = runs(ids["xgboost-demo"])[0]["info"]["run_id"]
        capturer(page, f"{MLFLOW}/#/experiments/{ids['xgboost-demo']}/runs/{run_xgboost}/model-metrics",
                 "29-courbe-xgboost", [lambda pg: pg.get_by_text("validation_0-mlogloss").first], attente=5000)
        if "pytorch-demo" in ids:
            run_torch = runs(ids["pytorch-demo"])[0]["info"]["run_id"]
            page.set_viewport_size({"width": 1440, "height": 1000})
            capturer(page, f"{MLFLOW}/#/experiments/{ids['pytorch-demo']}/runs/{run_torch}/model-metrics",
                     "30-reseau-pytorch", attente=6000, avant=agrandir_image)
            page.set_viewport_size({"width": 1440, "height": 760})
        else:
            print("pytorch-demo absente : capture 30 ignorée (lancer pytorch_example.py)")
        
        if "genai-assistant" in ids:
            genai = ids["genai-assistant"]
            page.goto("about:blank")
            page.goto(MLFLOW)
            page.wait_for_timeout(1500)
            page.get_by_text("GenAI", exact=True).click()

            capturer(page, f"{MLFLOW}/#/prompts/assistant-wine-prompt", "31-registre-de-prompts", [
                lambda pg: pg.get_by_text("production").first,
                lambda pg: pg.get_by_text("{{fiche}}", exact=False).first,
            ])
            
            def fermer_decouverte(pg):
                pg.get_by_role("button", name="Got it").click()
                pg.wait_for_timeout(500)

            capturer(page, f"{MLFLOW}/#/experiments/{genai}/traces", "20-traces", [
                lambda pg: pg.get_by_text("Combien", exact=False).first,
                lambda pg: pg.get_by_text("Output", exact=True).first,
            ], avant=fermer_decouverte)

            def ouvrir_trace(pg):
                pg.get_by_text("Combien", exact=False).first.click()
                pg.wait_for_timeout(2500)

            capturer(page, f"{MLFLOW}/#/experiments/{genai}/traces", "21-detail-trace", [
                lambda pg: pg.get_by_text("chercher_fiche").first,
                lambda pg: pg.get_by_text("Completions").first,
            ], avant=ouvrir_trace)
            capturer(page, f"{MLFLOW}/#/experiments/{genai}/evaluation-runs", "22-evaluation-genai", [
                lambda pg: pg.get_by_text("contient_mot_cle", exact=False).first,
            ])

            def ouvrir_evaluation(pg):
                pg.locator("a", has_text="-").filter(has_not_text="genai").last.click()
                pg.wait_for_timeout(3000)

            page.set_viewport_size({"width": 2300, "height": 620})
            capturer(page, f"{MLFLOW}/#/experiments/{genai}/evaluation-runs", "23-detail-evaluation", [
                lambda pg: pg.get_by_text("Response", exact=True).first,
                lambda pg: pg.get_by_text("correctness").last,
            ], avant=ouvrir_evaluation)
        else:
            print("genai-assistant absente : captures 20 à 23 ignorées (lancer genai_example.py)")
        navigateur.close()


if __name__ == "__main__":
    main()
