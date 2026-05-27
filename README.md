# ✈️ Projet Fondements de l'IA : Analyse, Classification et Prédiction des Prix de Vols

**Auteur** : Eliott, Mayeul, Noah (A3 - ALT TDC)

**Cours** : Fondements de l'IA - ESILV (2025-2026)

---

## 📌 Présentation du Projet

Ce projet a été réalisé à partir du jeu de données *EaseMyTrip* (environ 300 000 lignes) concernant les réservations de vols entre les principales métropoles en Inde. L'objectif principal est de mettre en pratique nos compétences en Data Science et Machine Learning pour traiter deux problématiques majeures :
1. **Régression** : Prédire le prix exact d'un billet de vol (variable continue).
2. **Classification** : Déterminer la catégorie d'un vol (par exemple, sa classe ou son niveau de prix) à des fins d'analyse de marché.

---

## 🛠️ Structure et Étapes du Code

Le projet est rigoureusement segmenté selon le cahier des charges et suit le déroulement précis de notre Notebook `Projet.ipynb` :

### 1. Nettoyage et Prétraitement des Données
Pour garantir la qualité des données avant de les injecter dans nos modèles, nous avons mis en place un pipeline robuste :
* **Nettoyage initial** : Traduction et renommage des colonnes en français pour plus de clarté, suppression de la colonne `flight` (identifiant unique non prédictif), gestion des doublons et conversion de la variable `duree` au format numérique.
* **Pipeline Automatisé (`ColumnTransformer`)** :
    * *Variables numériques* (`duree`, `jours_avant_depart`) : Imputation des valeurs manquantes par la médiane et normalisation via un `StandardScaler`.
    * *Variables ordinales* (`escales`, `classe`) : Encodage logique respectant l'ordre des catégories via `OrdinalEncoder`.
    * *Variables nominales* (`compagnie`, `ville_depart`, `ville_arrivee`, etc.) : Imputation par la valeur la plus fréquente (mode) et encodage via `OneHotEncoder`.

### 2. Analyse Exploratoire des Données (EDA)
Cette étape a permis de dégager les premières statistiques clés du dataset :
* **Distribution des classes** : On observe une forte asymétrie avec environ 68,8 % de billets en classe *Économique* contre 31,1 % en classe *Affaires*.
* **Matrice de corrélation** : Calculée et affichée pour mesurer l'intensité et le sens des relations linéaires entre les variables clés (le prix, la durée du vol et le délai de réservation).

### 3. Visualisation des Données
À l'aide des bibliothèques `Matplotlib` et `Seaborn`, nous avons réalisé au moins 4 visualisations clés pour valider nos intuitions :
* **Boîtes à moustaches (Boxplots)** : Elles ont mis en évidence que la variable `classe` est le facteur le plus discriminant. La médiane de la classe Business est près de 10 fois supérieure à celle de la classe Économie.
* **Diagrammes en barres** : Analyse des parts de marché et des tarifs moyens par compagnie (Vistara et Air India se partagent l'exclusivité des vols Business).
* **Évolution temporelle** : Analyse de l'impact des jours restants avant le départ, montrant une hausse exponentielle des prix dans les 10 à 15 derniers jours avant le vol.

### 4 & 5. Sélection des Caractéristiques et Modèles
* **Justification méthodologique** : Les algorithmes non supervisés (K-Means, HCA) ont été écartés car notre problématique est purement supervisée. Le modèle KNN a également été exclu en raison de son coût de calcul trop élevé sur un volume de 300 000 lignes.
* **Modèles retenus** :
    * *Pour la Régression* : Une Régression Linéaire (modèle de référence/*baseline*) confrontée à un algorithme non linéaire puissant, le `GradientBoostingRegressor`.
    * *Pour la Classification* : Une Régression Logistique (configurée avec `max_iter=1000` pour assurer la convergence) comparée à un `GradientBoostingClassifier`.

### 6 & 7. Entraînement et Évaluation
Pour éviter toute fuite de données (*Data Leakage*), deux pipelines de traitement distincts ont été créés selon la nature de la variable cible. Les données ont été segmentées en ensembles d'entraînement (80 %) et de test (20 %).
* **Résultats de la Régression** : Le modèle de Gradient Boosting surpasse largement la régression linéaire, affichant une erreur quadratique moyenne (RMSE) considérablement réduite.
* **Résultats de la Classification** : Les modèles ont été évalués avec les métriques d'Accuracy, Recall et F1-Score (particulièrement adapté à notre déséquilibre de classe). Le Gradient Boosting s'est à nouveau imposé comme le modèle le plus performant.

### 8. Optimisation, Déploiement et Maintenance
* **Fine-Tuning** : Optimisation des hyperparamètres du modèle gagnant à l'aide de `GridSearchCV` (ajustement du `learning_rate` à 0.2 et de la profondeur `max_depth` à 3).
* **Méthodes d'Ensemble** : Expérimentation de techniques avancées de combinaison de modèles. Le `StackingClassifier` a écrasé la concurrence en atteignant une précision (Accuracy) de **99,98 %**.
* **Déploiement** : Exportation et sérialisation du modèle final d'empilement (*Stacking*) ainsi que du pipeline de prétraitement associé sous forme de fichiers binaires `.pkl` avec la bibliothèque `joblib`.
* **Maintenance et Suivi de Production** : Rédaction d'une stratégie de surveillance continue pour contrer la dérive des données (*Data Drift*). Cette section inclut la mise en place de seuils d'alerte critiques et d'une routine de réentraînement planifié du modèle.

---

## 💻 Environnement et Stack Technique

Le projet respecte les technologies préconisées dans le cadre du cours :
* **Langage de programmation** : Python 3
* **Traitement de données & ML** : `numpy`, `pandas`, `scikit-learn`, `joblib`
* **Visualisation graphique** : `matplotlib`, `seaborn`

---

## 📂 Contenu des Livrables

* `Projet.ipynb` : Le code source complet sous forme de Notebook Jupyter, structuré en sections, nettoyé et largement commenté pour expliciter chaque choix technique.
