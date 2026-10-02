# Interface Streamlit : tester le modèle de classification de commentaires citoyens.
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from pretraitement import CATEGORIES, detecter_ewe_mina, pretraiter_texte

DOSSIER = Path(__file__).parent
EMOJIS = {"Satisfaction": "😊", "Insatisfaction": "😞", "Suggestion": "💡"}
EXEMPLES = {
    "😊 Satisfaction": "Service très rapide, je suis satisfait de l'accueil.",
    "😞 Insatisfaction": "Personne ne répond au téléphone et on me fait revenir sans cesse.",
    "💡 Suggestion": "Il faudrait ajouter plus de guichets le samedi matin.",
    "🇹🇬 Mixte français/éwé": "Akpe na wò, service nyuie !",
}

st.set_page_config(page_title="Commentaires citoyens classification", page_icon="🗣️", layout="centered")


@st.cache_resource
def charger_modele():
    return joblib.load(DOSSIER / "modele_classification.joblib")


@st.cache_data
def charger_metriques():
    with open(DOSSIER / "metriques_modele.json", encoding="utf-8") as fichier:
        return json.load(fichier)


def utiliser_exemple(texte):
    # Callback des boutons d'exemples : remplit la zone de saisie.
    st.session_state["commentaire"] = texte


modele = charger_modele()
metriques = charger_metriques()

# En-tête
st.title("🗣️ Classifier un commentaire citoyen")
st.write("Saisissez un retour sur un service public (en français, éventuellement avec des mots éwé/mina). "
         "Le modèle prédit s'il s'agit d'une **satisfaction**, d'une **insatisfaction** ou d'une **suggestion**.")

# Exemples + saisie
st.caption("Essayer un exemple :")
colonnes = st.columns(len(EXEMPLES))
for colonne, (libelle, texte) in zip(colonnes, EXEMPLES.items()):
    colonne.button(libelle, on_click=utiliser_exemple, args=(texte,), use_container_width=True)

commentaire = st.text_area("Commentaire à analyser", key="commentaire", height=110,
                           placeholder="Ex. : Mon dossier traîne depuis des semaines sans réponse.")

if st.button("Prédire la catégorie", type="primary"):
    if not commentaire.strip():
        st.warning("Veuillez saisir un commentaire.")
    else:
        texte_nettoye = pretraiter_texte(commentaire)
        if not texte_nettoye:
            st.warning("Après nettoyage, il ne reste aucun mot exploitable. Essayez une phrase plus complète.")
        else:
            probabilites = pd.Series(modele.predict_proba([texte_nettoye])[0], index=modele.classes_)
            probabilites = probabilites.reindex(CATEGORIES)
            categorie = probabilites.idxmax()
            confiance = probabilites.max()

            st.subheader(f"{EMOJIS[categorie]} {categorie}")
            st.progress(float(confiance), text=f"Confiance du modèle : {confiance:.0%}")
            if confiance < 0.5:
                st.info("Confiance faible : le commentaire est ambigu ou éloigné des exemples d'entraînement. "
                        "Interprétez ce résultat avec prudence.")

            st.bar_chart(probabilites.rename("Probabilité"), horizontal=True)

            with st.expander("Voir le prétraitement appliqué"):
                st.write("**Texte nettoyé :**", texte_nettoye)
                termes_ewe = detecter_ewe_mina(commentaire)
                if termes_ewe:
                    st.write("**Termes éwé/mina détectés :**", ", ".join(termes_ewe))
                else:
                    st.write("Aucun terme éwé/mina détecté.")

# Informations sur le modèle
with st.sidebar:
    st.header("À propos du modèle")
    st.write("**Algorithme :** régression logistique")
    st.write("**Représentation :** TF-IDF sur n-grammes de caractères (2 à 5)")
    st.write(f"**Entraînement :** {metriques['nb_commentaires_entrainement']} commentaires")
    st.metric("F1 macro (validation croisée)",
              f"{metriques['f1_macro_cv_moyenne']:.2f} ± {metriques['f1_macro_cv_ecart_type']:.2f}")
    st.caption("Limites : jeu de données très petit, négations mal gérées, très peu d'exemples éwé/mina. "
               "Les prédictions sont indicatives.")
    st.caption("Test technique NLP — TAISS 2026 × Afriklang")
