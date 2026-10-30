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
2. Chromium est installé : uv run --with playwright playwright install chromium

Exécution:
---------
uv run scripts/capture_screenshots.py

Les images sont écrites dans documentation/images/. Les pastilles numérotées
renvoient aux légendes du tutoriel et du manuel utilisateur.
"""

import json
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

MLFLOW = "http://localhost:5000"
FILER = "http://127.0.0.1:8888"
MASTER = "http://127.0.0.1:9333"
IMAGES = Path(__file__).resolve().parent.parent / "documentation" / "images"
EXEMPLES = {
    "tutoriel-basique": "basic_example.py",
    "autolog-demo": "autolog_example.py",
    "dataset-demo": "dataset_example.py",
}

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
    """Attend le premier entraînement du runner et lance les exemples manquants."""
    for _ in range(60):
        try:
            api("registered-models/alias?name=wine-classifier&alias=champion")
            break
        except OSError:
            time.sleep(5)
    else:
        raise SystemExit("Pas de modèle @champion : la stack tourne-t-elle ?")
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
    """Ferme la fenêtre promotionnelle du registre, affichée à la première visite."""
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)


def main():
    preparer_donnees()
    IMAGES.mkdir(parents=True, exist_ok=True)
    ids = experiences()
    comparaison = ids["comparaison-modeles"]
    champion = api("registered-models/alias?name=wine-classifier&alias=champion")["model_version"]
    tous = [r["info"]["run_id"] for r in runs(comparaison)]
    run_basique = runs(ids["tutoriel-basique"])[0]["info"]["run_id"]
    run_autolog = runs(ids["autolog-demo"])[0]["info"]["run_id"]
    run_dataset = runs(ids["dataset-demo"])[0]["info"]["run_id"]
    url_run = f"{MLFLOW}/#/experiments/{comparaison}/runs/{champion['run_id']}"

    with sync_playwright() as p:
        navigateur = p.chromium.launch()
        page = navigateur.new_page(viewport={"width": 1440, "height": 760})

        capturer(page, f"{MLFLOW}/#/experiments/{comparaison}", "01-table-des-runs", [
            lambda pg: pg.get_by_role("link", name="comparaison-modeles").first,
            lambda pg: pg.get_by_role("link", name="Random Forest Large"),
            lambda pg: pg.get_by_text("wine_training_data").first,
            lambda pg: pg.get_by_role("link", name="wine-classifier"),
        ])

        def selectionner(pg):
            lignes = pg.locator(".ag-row .ag-checkbox-input")
            for i in range(lignes.count()):
                if lignes.nth(i).is_visible():
                    lignes.nth(i).check(force=True)
            pg.wait_for_timeout(500)

        capturer(page, f"{MLFLOW}/#/experiments/{comparaison}", "02-selection-pour-comparer", [
            lambda pg: pg.get_by_role("button", name="Compare"),
        ], avant=selectionner)

        page.set_viewport_size({"width": 1440, "height": 1050})
        capturer(page, url_run, "03-run-overview", [
            lambda pg: pg.get_by_text("Datasets used"),
            lambda pg: pg.get_by_text("Tags", exact=True),
            lambda pg: pg.get_by_text("Logged models"),
            lambda pg: pg.get_by_text("Registered models"),
            lambda pg: pg.get_by_role("heading", name="Parameters"),
            lambda pg: pg.get_by_role("heading", name="Metrics"),
        ])
        capturer(page, f"{MLFLOW}/#/experiments/{ids['autolog-demo']}/runs/{run_autolog}", "04-run-autolog", [
            lambda pg: pg.get_by_role("heading", name="Parameters"),
            lambda pg: pg.get_by_role("heading", name="Metrics"),
        ])
        page.set_viewport_size({"width": 1440, "height": 760})
        capturer(page, f"{MLFLOW}/#/experiments/{ids['dataset-demo']}/runs/{run_dataset}", "05-run-datasets", [
            lambda pg: pg.get_by_text("Datasets used"),
        ])
        capturer(page, f"{MLFLOW}/#/experiments/{ids['tutoriel-basique']}/runs/{run_basique}/model-metrics",
                 "06-courbe-de-metrique", [lambda pg: pg.get_by_text("loss", exact=True).first])
        capturer(page, f"{url_run}/artifacts/confusion_matrix.png", "07-run-artefacts", [
            lambda pg: pg.get_by_text("confusion_matrix.png").first,
            lambda pg: pg.get_by_text("s3://mlflow-artifacts"),
        ])

        ids_json = urllib.parse.quote(json.dumps(tous))
        page.set_viewport_size({"width": 1440, "height": 1000})
        capturer(page, f'{MLFLOW}/#/compare-runs?runs={ids_json}&experiments=["{comparaison}"]', "08-comparaison", [
            lambda pg: pg.get_by_text("Parallel Coordinates Plot"),
        ], attente=5000)
        page.set_viewport_size({"width": 1440, "height": 760})
        capturer(page, f"{MLFLOW}/#/models", "09-registre-des-modeles", [
            lambda pg: pg.get_by_role("link", name="wine-classifier"),
            lambda pg: pg.get_by_text("champion").first,
        ], avant=fermer_promotion)
        capturer(page, f"{MLFLOW}/#/models/wine-classifier/versions/{champion['version']}", "10-version-du-modele", [
            lambda pg: pg.get_by_role("link", name="Random Forest"),
            lambda pg: pg.get_by_text("champion").first,
        ], avant=fermer_promotion)
        capturer(page, f"{FILER}/buckets/mlflow-artifacts/{comparaison}/{champion['run_id']}/artifacts/",
                 "11-stockage-seaweedfs", [lambda pg: pg.get_by_role("link", name="confusion_matrix.png").first])
        capturer(page, f"{FILER}/buckets/mlflow-artifacts/", "12-seaweedfs-bucket", [
            lambda pg: pg.get_by_role("link", name="mlflow-artifacts"),
            lambda pg: pg.get_by_role("link", name=comparaison, exact=True).last,
        ], attente=1500)
        capturer(page, f"{FILER}/buckets/mlflow-artifacts/{comparaison}/", "13-seaweedfs-runs", [
            lambda pg: pg.get_by_role("link", name=champion["run_id"]),
        ], attente=1500)
        capturer(page, f"{FILER}/buckets/mlflow-artifacts/{comparaison}/{champion['run_id']}/artifacts/model/",
                 "14-seaweedfs-modele", [
            lambda pg: pg.get_by_role("link", name="MLmodel"),
            lambda pg: pg.get_by_role("link", name="model.pkl"),
        ], attente=1500)
        capturer(page, MASTER, "15-seaweedfs-cluster", [
            lambda pg: pg.get_by_text("Cluster status"),
            lambda pg: pg.get_by_role("heading", name="Topology"),
        ], attente=1500)
        navigateur.close()


if __name__ == "__main__":
    main()
