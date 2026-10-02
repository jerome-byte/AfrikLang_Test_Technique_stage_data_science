# Classification de commentaires citoyens ,Test technique NLP (TAISS 2026 * Afriklang)

## Approche, choix et résultats
- **Données** : 150 commentaires équilibrés (50 Satisfaction / 50 Insatisfaction / 50 Suggestion), en français ; seuls 3 commentaires sont mixtes éwé/mina.
- **Nettoyage** : minuscules, suppression de la ponctuation, des chiffres et des symboles, tokenisation Unicode ; accents conservés (neutralisés ensuite dans le vectoriseur).
- **Stopwords** : liste maison (108 mots grammaticaux) qui **conserve** les négations, intensifieurs et marqueurs de suggestion (`pas`, `aucune`, `sans`, `trop`, `serait`…), car ils portent le sens.
- **Éwé/mina** : `akpe`→`merci`, `nyuie`→`bien`, mots-outils (`na`, `wò`) supprimés, autres termes conservés. Les 3 stratégies testées donnent des scores équivalents (trop peu d'exemples mixtes).
- **Vectorisation** : TF-IDF sur n-grammes de caractères (2–5), retenu par validation croisée sur le train (F1 macro ≈ 0,66 contre 0,63–0,65 pour les variantes à base de mots, écarts non significatifs).
- **Modèles** : Logistic Regression, SVM linéaire, Naive Bayes, Random Forest ; split 80/20 stratifié, `random_state=42`.
- **Résultats** : sur le test (30 exemples) accuracy 0,80–0,83 ; en validation croisée 5×3, F1 macro **0,72** (LR et SVM), 0,67 (RF et NB). La valeur de validation croisée est la plus fiable.
- **Erreurs** : surtout `Satisfaction`  `Insatisfaction` (23 des 44 erreurs hors-échantillon) : négations non gérées, mots rares, thèmes partagés, chiffres supprimés.
- **Limites** : 150 textes seulement (test de 30 : un exemple = 3,3 points), peu de données éwé/mina, sac de mots sans contexte, hyperparamètres non optimisés.
- **Pistes** : fine-tuning de CamemBERT / XLM-R ; gestion de la négation et des nombres ; plus de commentaires mixtes avec lexique validé par des locuteurs.
- **Bonus — interface web (Streamlit)** : le modèle (`modele_classification.joblib`, entraîné sur les 150 textes) est branché à `app.py`. Démo en ligne : https://afriklangteststagedatascience-dnjdu5i5nawsgfrj5elezw.streamlit.app/



## Installation et exécution en local

**Prérequis** : Git et **Python 3.11 ou plus récent**. Vérifier avec `python --version`.

### 1. Récupérer le dépôt
```bash
git clone URL
cd afriklang_test_stage_data_science
```

### 2. Créer et activer un environnement virtuel
Un environnement virtuel isole les bibliothèques du projet de celles du reste de l'ordinateur.

<table>
    <tr>
      <th>Système</th>
      <th>Création</th>
      <th>Activation</th>
    </tr>
    <tr>
      <td><strong>Windows (PowerShell)</strong></td>
      <td><code>python -m venv .venv</code></td>
      <td><code>.venv\Scripts\Activate.ps1</code></td>
    </tr>
    <tr>
      <td><strong>Windows (cmd)</strong></td>
      <td><code>python -m venv .venv</code></td>
      <td><code>.venv\Scripts\activate.bat</code></td>
    </tr>
    <tr>
      <td><strong>macOS / Linux</strong></td>
      <td><code>python3 -m venv .venv</code></td>
      <td><code>source .venv/bin/activate</code></td>
    </tr>
</table>
`

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```
Les versions sont épinglées dans `requirements.txt` : on retrouve le même environnement que celui utilisé pour le rendu.

### 4. Exécuter le notebook
```bash
jupyter notebook
```
Ouvrir `classification_commentaires.ipynb` puis **Kernel → Restart & Run All**. Les graines sont fixées (`SEED = 42`) : on obtient les mêmes chiffres et les mêmes figures. Le fichier `dataset_nlp_test_tal.csv` doit rester dans le même dossier que le notebook.
La dernière cellule régénère `modele_classification.joblib` et `metriques_modele.json`.

### 5. Lancer l'interface Streamlit
```bash
streamlit run app.py
```
L'application s'ouvre sur http://localhost:8501. Fichiers nécessaires dans le même dossier : `app.py`, `pretraitement.py`, `modele_classification.joblib`, `metriques_modele.json`.

### 6. Quitter l'environnement
```bash
deactivate
```
Pour y revenir plus tard : refaire seulement l'activation (étape 2), sans réinstaller.
