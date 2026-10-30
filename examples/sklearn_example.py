"""
sklearn_example.py - MLflow avec Scikit-Learn
=============================================

Ce script démontre l'utilisation complète de MLflow avec scikit-learn :
- Logging du dataset pour la traçabilité
- Comparaison de plusieurs modèles (runs)
- Génération d'artefacts standardisés pour la comparaison
- Enregistrement du meilleur modèle dans le Model Registry
- Attribution de l'alias "champion" au meilleur modèle

Prérequis:
---------
1. Lancer les services Docker : docker compose up -d
2. Attendre que les services soient healthy : docker compose ps

Exécution:
---------
docker compose exec runner python sklearn_example.py

Interface MLflow:
----------------
Après exécution, consultez http://localhost:5000 pour voir :
- Les runs dans l'expérience "comparaison-modeles"
- Les datasets loggés (onglet Overview)
- Les artefacts comparables (matrices de confusion)
- Le modèle enregistré dans le Model Registry
"""

import os
import tempfile
import warnings
import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from dotenv import load_dotenv

# Suppression des warnings pour plus de clarté dans la sortie console
warnings.filterwarnings('ignore')

# Chargement des variables d'environnement depuis le fichier .env
# Cela permet de configurer les credentials sans les hardcoder
load_dotenv()


# ============================================================================
# SECTION 1 : CONFIGURATION DE MLFLOW
# ============================================================================

def configurer_mlflow():
    """
    Configure la connexion au serveur MLflow.
    
    MLflow nécessite deux types de configuration :
    1. Le Tracking URI : où envoyer les métadonnées (paramètres, métriques)
    2. Les credentials S3 : pour stocker les artefacts dans MinIO
    
    Note: Si exécuté dans le conteneur Docker 'runner', les variables 
    d'environnement sont déjà définies dans docker-compose.yml.
    
    Returns:
        None
    """
    # --------------------------------------------------------------------
    # Configuration du Tracking URI
    # --------------------------------------------------------------------
    # Le tracking URI est l'adresse du serveur MLflow qui reçoit les logs
    # Par défaut : http://localhost:5000 (accessible depuis votre machine)
    # Dans Docker : http://mlflow:5000 (nom du service Docker)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # --------------------------------------------------------------------
    # Configuration S3 pour MinIO (stockage des artefacts)
    # --------------------------------------------------------------------
    # MinIO est une alternative open-source à Amazon S3
    # MLflow utilise le protocole S3 pour stocker les fichiers volumineux :
    # - Modèles sérialisés (.pkl, .joblib)
    # - Graphiques (.png, .jpg)
    # - Fichiers de données
    
    # Endpoint S3 : adresse du serveur MinIO
    if "MLFLOW_S3_ENDPOINT_URL" not in os.environ:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    
    # Credentials S3 : identifiants pour accéder à MinIO
    # Ces valeurs doivent correspondre à MINIO_ROOT_USER/PASSWORD dans docker-compose.yml
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ROOT_USER", "minio_admin")
    # Pas de valeur par défaut pour le mot de passe : il vient uniquement du .env
    if "AWS_SECRET_ACCESS_KEY" not in os.environ and os.getenv("MINIO_ROOT_PASSWORD"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["MINIO_ROOT_PASSWORD"]
    
    print("✅ Configuration MLflow terminée")
    print(f"   📡 Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"   🪣 S3 Endpoint: {os.environ.get('MLFLOW_S3_ENDPOINT_URL')}")


# ============================================================================
# SECTION 2 : CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

def charger_donnees():
    """
    Charge et prépare le dataset Wine de scikit-learn.
    
    Le dataset Wine contient 178 échantillons de vins italiens avec :
    - 13 features (caractéristiques chimiques)
    - 3 classes (types de vins)
    
    Étapes de préparation :
    1. Chargement du dataset
    2. Division train/test (80/20) avec stratification
    3. Normalisation des features (StandardScaler)
    
    La normalisation est importante car :
    - Les features ont des échelles différentes (ex: alcool vs magnésium)
    - SVM et Logistic Regression sont sensibles à l'échelle
    - Random Forest est robuste mais bénéficie quand même de la normalisation
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, target_names, df_train)
               df_train est un DataFrame pour le logging MLflow
    """
    print("\n📊 Chargement du dataset Wine...")
    
    # --------------------------------------------------------------------
    # Étape 1 : Chargement du dataset
    # --------------------------------------------------------------------
    # load_wine() retourne un objet Bunch avec :
    # - data : array numpy des features (178, 13)
    # - target : array numpy des labels (178,)
    # - feature_names : noms des 13 features
    # - target_names : noms des 3 classes
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    # --------------------------------------------------------------------
    # Étape 2 : Division train/test
    # --------------------------------------------------------------------
    # test_size=0.2 : 20% pour le test, 80% pour l'entraînement
    # random_state=42 : graine aléatoire pour la reproductibilité
    # stratify=y : maintient la proportion des classes dans train et test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Stratification importante pour les petits datasets
    )
    
    # --------------------------------------------------------------------
    # Étape 3 : Normalisation des features
    # --------------------------------------------------------------------
    # StandardScaler : transforme chaque feature pour avoir moyenne=0 et std=1
    # fit_transform sur train : calcule les paramètres ET transforme
    # transform sur test : utilise les paramètres du train (pas de data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --------------------------------------------------------------------
    # Création d'un DataFrame pour le logging MLflow
    # --------------------------------------------------------------------
    # MLflow peut logger un DataFrame comme dataset d'entrée
    # Cela permet de tracer exactement quelles données ont été utilisées
    df_train = pd.DataFrame(X_train_scaled, columns=feature_names)
    df_train['target'] = y_train
    df_train['target_name'] = [target_names[t] for t in y_train]
    
    # Affichage des informations sur le dataset
    print(f"   📁 Échantillons d'entraînement: {len(X_train_scaled)}")
    print(f"   📁 Échantillons de test: {len(X_test_scaled)}")
    print(f"   📐 Nombre de features: {X_train_scaled.shape[1]}")
    print(f"   🏷️  Classes: {list(target_names)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_names, df_train


# ============================================================================
# SECTION 3 : CRÉATION DES ARTEFACTS (GRAPHIQUES)
# ============================================================================

def creer_graphique_confusion(y_true, y_pred, target_names, nom_modele):
    """
    Crée une matrice de confusion sous forme de figure Matplotlib.
    
    La matrice de confusion montre :
    - Diagonale : prédictions correctes
    - Hors diagonale : erreurs de classification
    
    IMPORTANT : Cette fonction retourne un objet Figure pour être utilisé
    avec mlflow.log_figure(), qui l'enregistre comme artefact du run.
    
    Args:
        y_true: Labels réels (ground truth)
        y_pred: Labels prédits par le modèle
        target_names: Noms des classes pour l'affichage
        nom_modele: Nom du modèle (utilisé dans le titre)
    
    Returns:
        matplotlib.figure.Figure: Figure contenant la matrice de confusion
    """
    # Création de la figure avec une taille adaptée
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calcul de la matrice de confusion
    # confusion_matrix retourne un array 2D : cm[i,j] = nb échantillons de classe i prédits comme j
    cm = confusion_matrix(y_true, y_pred)
    
    # Affichage avec ConfusionMatrixDisplay pour un rendu propre
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    # Personnalisation du graphique
    ax.set_title(f'Matrice de Confusion - {nom_modele}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def creer_rapport_classification(y_true, y_pred, target_names, nom_modele):
    """
    Crée un rapport de classification textuel.
    
    Le rapport inclut pour chaque classe :
    - Precision : parmi les prédictions positives, combien sont correctes
    - Recall : parmi les vrais positifs, combien sont détectés
    - F1-score : moyenne harmonique de precision et recall
    - Support : nombre d'échantillons dans la classe
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        target_names: Noms des classes
        nom_modele: Nom du modèle
    
    Returns:
        str: Chemin du fichier texte créé
    """
    # Génération du rapport avec sklearn
    rapport = classification_report(y_true, y_pred, target_names=target_names)
    
    # Création du fichier texte
    chemin = os.path.join(tempfile.gettempdir(), "classification_report.txt")
    
    with open(chemin, 'w', encoding='utf-8') as f:
        f.write(f"Rapport de Classification - {nom_modele}\n")
        f.write("=" * 60 + "\n\n")
        f.write(rapport)
    
    return chemin


# ============================================================================
# SECTION 4 : ENTRAÎNEMENT ET LOGGING D'UN MODÈLE
# ============================================================================

def entrainer_modele(nom_modele, modele, X_train, X_test, y_train, y_test, 
                     target_names, feature_names, df_train, model_index):
    """
    Entraîne un modèle et log tous les résultats dans MLflow.
    
    Cette fonction démontre le logging complet avec MLflow :
    1. Paramètres : hyperparamètres du modèle
    2. Métriques : performances mesurées
    3. Tags : métadonnées descriptives
    4. Artefacts : fichiers (graphiques, rapports)
    5. Dataset : données d'entraînement (traçabilité)
    6. Modèle : modèle sérialisé réutilisable
    
    Args:
        nom_modele: Nom descriptif du modèle (ex: "Random Forest")
        modele: Instance du modèle sklearn (non entraîné)
        X_train, X_test, y_train, y_test: Données train/test
        target_names: Noms des classes
        feature_names: Noms des features
        df_train: DataFrame du training set pour le logging
        model_index: Index numérique du modèle (pour faciliter la comparaison)
    
    Returns:
        dict: Dictionnaire des métriques calculées
    """
    # ========================================================================
    # DÉMARRAGE D'UN RUN MLFLOW
    # ========================================================================
    # mlflow.start_run() crée un nouveau run dans l'expérience active
    # run_name : nom affiché dans l'interface (optionnel mais recommandé)
    # Le context manager (with) garantit que le run est fermé proprement
    
    with mlflow.start_run(run_name=nom_modele):
        print(f"\n🔄 Entraînement: {nom_modele}")
        
        # ====================================================================
        # ÉTAPE 1 : LOGGING DU DATASET (INPUT)
        # ====================================================================
        # mlflow.log_input() enregistre le dataset utilisé pour l'entraînement
        # Cela apparaît dans l'onglet "Overview" du run dans l'UI
        # 
        # Pourquoi c'est important ?
        # - Traçabilité : savoir exactement quelles données ont été utilisées
        # - Reproductibilité : pouvoir recréer les conditions d'entraînement
        # - Debugging : identifier si un problème vient des données
        
        dataset = mlflow.data.from_pandas(
            df_train,
            source="sklearn.datasets.load_wine",  # Source des données
            name="wine_training_data"  # Nom du dataset
        )
        mlflow.log_input(dataset, context="training")
        
        # ====================================================================
        # ÉTAPE 2 : ENTRAÎNEMENT DU MODÈLE
        # ====================================================================
        # fit() entraîne le modèle sur les données
        # Après cette étape, le modèle peut faire des prédictions
        
        modele.fit(X_train, y_train)
        
        # Prédictions sur les deux ensembles
        y_pred_train = modele.predict(X_train)
        y_pred_test = modele.predict(X_test)
        
        # ====================================================================
        # ÉTAPE 3 : CALCUL DES MÉTRIQUES
        # ====================================================================
        # Les métriques quantifient la performance du modèle
        # 
        # accuracy : % de prédictions correctes
        # precision : parmi les positifs prédits, combien sont vrais positifs
        # recall : parmi les vrais positifs, combien sont détectés
        # f1_score : moyenne harmonique de precision et recall
        # 
        # average='weighted' : pondère par le nombre d'échantillons par classe
        
        metriques = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test, average='weighted'),
            "recall": recall_score(y_test, y_pred_test, average='weighted'),
            "f1_score": f1_score(y_test, y_pred_test, average='weighted')
        }
        
        # Validation croisée : évalue la robustesse du modèle
        # cv=5 : divise les données en 5 folds
        # Chaque fold est utilisé une fois comme validation
        cv_scores = cross_val_score(modele, X_train, y_train, cv=5)
        metriques["cv_mean"] = cv_scores.mean()
        metriques["cv_std"] = cv_scores.std()
        
        # ====================================================================
        # ÉTAPE 4 : LOGGING DES PARAMÈTRES
        # ====================================================================
        # MODIFICATION UTILISATEUR :
        # On ne log plus automatiquement tous les paramètres (modele.get_params())
        # pour éviter de polluer l'interface avec des valeurs par défaut.
        
        # On log explicitement le nom du run comme paramètre pour qu'il soit
        # sélectionnable dans les axes des graphiques de comparaison.
        mlflow.log_param("run_name", nom_modele)
        
        # On peut aussi logger le type de modèle
        mlflow.log_param("model_type", type(modele).__name__)
        
        # Paramètre supplémentaire : index du modèle
        # Utile pour la comparaison dans les graphiques MLflow
        # (les axes ne supportent que les paramètres/métriques numériques)
        mlflow.log_param("model_index", model_index)
        
        # ====================================================================
        # ÉTAPE 5 : LOGGING DES MÉTRIQUES
        # ====================================================================
        # log_metrics() enregistre plusieurs métriques d'un coup
        # Ces valeurs apparaissent dans l'onglet "Metrics" et sont utilisées
        # pour la comparaison entre runs
        
        mlflow.log_metrics(metriques)
        
        # ====================================================================
        # ÉTAPE 6 : LOGGING DES TAGS
        # ====================================================================
        # Les tags sont des métadonnées textuelles
        # Ils ne sont PAS utilisables pour les graphiques de comparaison
        # mais sont utiles pour le filtrage et la documentation
        
        mlflow.set_tag("modele", nom_modele)
        mlflow.set_tag("dataset", "wine")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("task", "classification")
        
        # ====================================================================
        # ÉTAPE 7 : SAUVEGARDE DU MODÈLE
        # ====================================================================
        # mlflow.sklearn.log_model() sérialise le modèle entraîné
        # 
        # artifact_path : dossier dans les artefacts du run
        # Le modèle peut ensuite être chargé avec mlflow.sklearn.load_model()
        # 
        # Note : registered_model_name=None car on enregistrera le meilleur
        # modèle plus tard dans le Model Registry
        
        mlflow.sklearn.log_model(
            modele, 
            artifact_path="model",
            registered_model_name=None,
            input_example=X_train[:5]  # Exemple pour inférer la signature
        )
        
        # ====================================================================
        # ÉTAPE 8 : LOGGING DES ARTEFACTS
        # ====================================================================
        # Les artefacts sont des fichiers associés au run
        # artifact_path="reports" : les place dans un sous-dossier 'reports'
        # 
        # IMPORTANT : Les noms standardisés permettent la comparaison
        # entre runs dans l'interface MLflow
        
        # Matrice de confusion : log_figure enregistre la figure à la racine
        # des artefacts du run, sous le nom donné par artifact_file
        fig_cm = creer_graphique_confusion(
            y_test, y_pred_test, target_names, nom_modele
        )
        mlflow.log_figure(fig_cm, artifact_file="confusion_matrix.png")
        plt.close(fig_cm)  # Libération mémoire
        
        # Rapport de classification
        chemin_rapport = creer_rapport_classification(
            y_test, y_pred_test, target_names, nom_modele
        )
        mlflow.log_artifact(chemin_rapport, artifact_path="reports")
        os.remove(chemin_rapport)
        
        # ====================================================================
        # AFFICHAGE DES RÉSULTATS
        # ====================================================================
        print(f"   ✅ Accuracy (test): {metriques['test_accuracy']:.4f}")
        print(f"   ✅ F1-Score: {metriques['f1_score']:.4f}")
        print(f"   ✅ CV Score: {metriques['cv_mean']:.4f} (±{metriques['cv_std']:.4f})")
        
        # Récupération de l'ID du run pour le Model Registry
        run_id = mlflow.active_run().info.run_id
        
        return metriques, run_id


# ============================================================================
# SECTION 5 : COMPARAISON DE PLUSIEURS MODÈLES
# ============================================================================

def comparer_modeles():
    """
    Compare plusieurs modèles ML et log les résultats dans MLflow.
    
    Cette fonction :
    1. Crée/sélectionne une expérience MLflow
    2. Charge et prépare les données
    3. Entraîne plusieurs modèles différents
    4. Compare leurs performances
    5. Enregistre le meilleur modèle dans le Model Registry
    
    Returns:
        tuple: (resultats, X_test, y_test, target_names)
               Les données de test servent à la démonstration du chargement
    """
    # ========================================================================
    # CRÉATION DE L'EXPÉRIENCE
    # ========================================================================
    # Une expérience regroupe tous les runs liés à un même objectif
    # Si l'expérience existe déjà, elle est réutilisée
    
    experiment_name = "comparaison-modeles"
    mlflow.set_experiment(experiment_name)
    print(f"\n🧪 Expérience: {experiment_name}")
    
    # Chargement des données
    X_train, X_test, y_train, y_test, feature_names, target_names, df_train = charger_donnees()
    
    # ========================================================================
    # DÉFINITION DES MODÈLES À COMPARER
    # ========================================================================
    # Chaque entrée contient :
    # - Clé : nom descriptif du modèle
    # - Valeur : instance du modèle sklearn avec ses hyperparamètres
    # 
    # random_state=42 : garantit la reproductibilité des résultats
    
    modeles = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,      # Nombre d'arbres dans la forêt
            max_depth=5,           # Profondeur maximale de chaque arbre
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,         # Nombre max d'itérations pour converger
            random_state=42
        ),
        "SVM RBF": SVC(
            kernel='rbf',          # Kernel gaussien (RBF)
            C=1.0,                 # Paramètre de régularisation
            random_state=42
        ),
        "Random Forest Large": RandomForestClassifier(
            n_estimators=200,      # Plus d'arbres
            max_depth=10,          # Arbres plus profonds
            min_samples_split=5,   # Min échantillons pour diviser un noeud
            random_state=42
        )
    }
    
    # ========================================================================
    # ENTRAÎNEMENT ET COMPARAISON
    # ========================================================================
    resultats = {}
    run_ids = {}
    
    print("\n" + "=" * 60)
    print("   📊 Comparaison des modèles")
    print("=" * 60)
    
    # Entraînement de chaque modèle avec un index pour la comparaison
    for index, (nom, modele) in enumerate(modeles.items(), start=1):
        metriques, run_id = entrainer_modele(
            nom, modele, 
            X_train, X_test, y_train, y_test, 
            target_names, feature_names, df_train,
            model_index=index  # Index numérique pour les graphiques de comparaison
        )
        resultats[nom] = metriques
        run_ids[nom] = run_id
    
    # ========================================================================
    # RÉSUMÉ DES RÉSULTATS
    # ========================================================================
    print("\n" + "=" * 60)
    print("   📊 Résumé des performances")
    print("=" * 60)
    
    # Création d'un DataFrame pour un affichage propre
    df_resultats = pd.DataFrame(resultats).T
    df_resultats = df_resultats.sort_values("test_accuracy", ascending=False)
    
    print("\n" + df_resultats[["test_accuracy", "f1_score", "cv_mean"]].to_string())
    
    # Identification du meilleur modèle
    meilleur_nom = df_resultats.index[0]
    meilleur_run_id = run_ids[meilleur_nom]
    
    print(f"\n🏆 Meilleur modèle: {meilleur_nom}")
    print(f"   Accuracy: {df_resultats.loc[meilleur_nom, 'test_accuracy']:.4f}")
    
    # ========================================================================
    # ENREGISTREMENT DANS LE MODEL REGISTRY
    # ========================================================================
    enregistrer_meilleur_modele(meilleur_run_id, meilleur_nom)
    
    return resultats, X_test, y_test, target_names


# ============================================================================
# SECTION 6 : MODEL REGISTRY (MISE EN PRODUCTION)
# ============================================================================

def enregistrer_meilleur_modele(run_id, nom_modele):
    """
    Enregistre le meilleur modèle dans le Model Registry de MLflow.
    
    Le Model Registry est un composant central pour la gestion des modèles :
    
    1. VERSIONING : Chaque enregistrement crée une nouvelle version
       - Version 1, 2, 3... automatiquement incrémenté
       - Historique complet des modèles
    
    2. ALIAS : Étiquette mobile posée sur une version
       - @champion : le modèle de référence actuel
       - @challenger : un candidat à évaluer
       Remplacent les stages (None/Staging/Production/Archived), dépréciés
       depuis MLflow 2.9
    
    3. DESCRIPTION : Documentation du modèle
       - Ajout de notes et descriptions
       - Traçabilité des changements
    
    4. CHARGEMENT SIMPLIFIÉ :
       - models:/nom-modele@champion : charge la version portant l'alias
       - models:/nom-modele/1 : charge une version spécifique
    
    Args:
        run_id: ID du run contenant le modèle à enregistrer
        nom_modele: Nom du modèle (pour la description)
    
    Returns:
        ModelVersion: Objet représentant la version enregistrée
    """
    print("\n" + "=" * 60)
    print("   🏭 Enregistrement dans le Model Registry")
    print("=" * 60)
    
    # Client MLflow pour les opérations avancées
    client = MlflowClient()
    
    # URI du modèle dans les artefacts du run
    # Format : runs:/<run_id>/<artifact_path>
    model_uri = f"runs:/{run_id}/model"
    
    # Nom du modèle dans le Registry
    # Ce nom doit être unique et descriptif
    registry_name = "wine-classifier"
    
    try:
        # ====================================================================
        # ENREGISTREMENT DU MODÈLE
        # ====================================================================
        # register_model() :
        # - Crée le modèle dans le Registry s'il n'existe pas
        # - Ajoute une nouvelle version si le modèle existe
        
        print(f"   📦 Enregistrement du modèle: {registry_name}")
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registry_name
        )
        
        print(f"   ✅ Version créée: {model_version.version}")
        
        # ====================================================================
        # AJOUT D'UNE DESCRIPTION
        # ====================================================================
        # La description aide à documenter le modèle pour les autres utilisateurs
        
        client.update_model_version(
            name=registry_name,
            version=model_version.version,
            description=f"Modèle {nom_modele} entraîné sur le dataset Wine. "
                        f"Meilleur modèle de l'expérience de comparaison."
        )
        
        # ====================================================================
        # ATTRIBUTION DE L'ALIAS "CHAMPION"
        # ====================================================================
        # set_registered_model_alias() pose l'alias sur cette version
        # 
        # Un alias ne désigne qu'une seule version à la fois : il quitte
        # automatiquement la version précédente, qui reste disponible
        
        print(f"   🚀 Attribution de l'alias 'champion'...")
        client.set_registered_model_alias(
            name=registry_name,
            alias="champion",
            version=model_version.version
        )
        
        print(f"   ✅ Modèle '{registry_name}' v{model_version.version} désormais @champion!")
        print(f"\n   💡 Pour charger ce modèle:")
        print(f"      model = mlflow.sklearn.load_model('models:/{registry_name}@champion')")
        
        return model_version
        
    except Exception as e:
        print(f"   ⚠️  Erreur lors de l'enregistrement: {e}")
        print("   Note: Le Model Registry nécessite que le serveur MLflow soit accessible.")
        return None


# ============================================================================
# SECTION 7 : DÉMONSTRATION DU CHARGEMENT D'UN MODÈLE
# ============================================================================

def demo_chargement_modele(X_test, y_test, target_names):
    """
    Démontre comment charger un modèle depuis le Model Registry.
    
    Cette fonction montre les différentes façons de charger un modèle :
    1. Depuis un run spécifique : runs:/<run_id>/model
    2. Depuis le Registry par version : models:/<name>/<version>
    3. Depuis le Registry par alias : models:/<name>@<alias>
    
    Args:
        X_test: Features de test, déjà normalisées comme à l'entraînement
        y_test: Labels réels
        target_names: Noms des classes
    """
    print("\n" + "=" * 60)
    print("   📥 Démonstration du chargement de modèle")
    print("=" * 60)
    
    registry_name = "wine-classifier"
    
    try:
        # Chargement de la version portant l'alias "champion"
        model_uri = f"models:/{registry_name}@champion"
        print(f"   🔄 Chargement depuis: {model_uri}")
        
        model = mlflow.sklearn.load_model(model_uri)
        print(f"   ✅ Modèle chargé: {type(model).__name__}")
        
        # Test de prédiction
        # Le modèle attend des données normalisées : le StandardScaler n'est pas
        # enregistré avec lui, on utilise donc X_test et non les données brutes
        predictions = model.predict(X_test[:5])
        
        print(f"   🎯 Prédictions sur 5 échantillons: {predictions}")
        print(f"   📝 Classes prédites: {[str(target_names[p]) for p in predictions]}")
        print(f"   📝 Classes réelles:  {[str(target_names[t]) for t in y_test[:5]]}")
        
    except Exception as e:
        print(f"   ⚠️  Le modèle n'est pas encore enregistré: {e}")
        print("   Exécutez d'abord la comparaison des modèles.")


# ============================================================================
# SECTION 8 : POINT D'ENTRÉE PRINCIPAL
# ============================================================================

def main():
    """
    Point d'entrée principal du script.
    
    Exécute le workflow complet :
    1. Configuration de MLflow
    2. Comparaison des modèles
    3. Enregistrement du meilleur modèle
    4. Démonstration du chargement
    """
    print("=" * 60)
    print("   🚀 MLflow + Scikit-Learn")
    print("   📊 Comparaison de modèles avec Model Registry")
    print("=" * 60)
    
    # Configuration de la connexion MLflow
    configurer_mlflow()
    
    # Comparaison des modèles (inclut l'enregistrement dans le Registry)
    _, X_test, y_test, target_names = comparer_modeles()
    
    # Démonstration du chargement d'un modèle
    demo_chargement_modele(X_test, y_test, target_names)
    
    # ========================================================================
    # INSTRUCTIONS POUR EXPLORER LES RÉSULTATS
    # ========================================================================
    print("\n" + "=" * 60)
    print("   🎉 Tutoriel terminé!")
    print("=" * 60)
    print("\n   📍 Consultez les résultats sur http://localhost:5000")
    print("\n   📌 Points à explorer dans l'interface MLflow:")
    print("      1. Expérience 'comparaison-modeles' : voir tous les runs")
    print("      2. Cliquez sur un run → Overview : voir le dataset loggé")
    print("      3. Cliquez sur un run → Artifacts : matrice de confusion et rapport")
    print("      4. Sélectionnez plusieurs runs → Compare : graphiques de comparaison")
    print("      5. Menu 'Models' : voir le Model Registry")
    print("\n   💡 Astuce pour la comparaison:")
    print("      Utilisez 'model_index' comme paramètre sur l'axe X")
    print("      pour différencier les modèles dans les graphiques.")


if __name__ == "__main__":
    main()
