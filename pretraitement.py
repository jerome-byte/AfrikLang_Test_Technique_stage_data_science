"""Prétraitement des commentaires citoyens (identique à celui du notebook).
Ce module est partagé par le script d'entraînement et par l'application Streamlit,
afin que le texte soit nettoyé exactement de la même façon à l'entraînement et en production.
"""
import re
import unicodedata

CATEGORIES = ["Satisfaction", "Insatisfaction", "Suggestion"]


def sans_accents(mot):
    # Retire les diacritiques (é -> e, è -> e, ...) ; sert uniquement aux comparaisons de listes.
    return "".join(c for c in unicodedata.normalize("NFD", mot) if unicodedata.category(c) != "Mn")


# Lexique éwé/mina repéré dans le corpus (clés sans accents)
LEXIQUE_EWE_MINA = {"akpe", "nyuie", "hafi", "mele", "megbe", "yevu", "wo", "na"}
EWE_VERS_FRANCAIS = {"akpe": "merci", "nyuie": "bien"}   
EWE_MOTS_OUTILS = {"na", "wo"}                            

STOPWORDS_FR = {
    # articles et déterminants
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "au", "aux", "ce", "cet", "cette", "ces",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
    # prépositions
    "à", "a", "en", "dans", "par", "pour", "sur", "sous", "avec", "chez", "entre", "vers",
    "pendant", "depuis", "dès", "lors", "avant", "hors", "selon", "via",
    # pronoms
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "me", "m", "te", "t", "se", "s",
    "lui", "eux", "y", "qui", "que", "quoi", "où", "dont", "c", "ça", "cela", "soi",
    # conjonctions
    "et", "ou", "mais", "donc", "or", "car", "comme", "si", "quand",
    # auxiliaires être / avoir (formes fréquentes)
    "est", "sont", "suis", "es", "était", "étaient", "été", "être", "ai", "as", "ont", "avait", "avoir", "eu",
    # mots vides divers
    "tout", "tous", "toute", "toutes", "j", "qu", "n",
}
STOPWORDS_SANS_ACCENTS = {sans_accents(mot) for mot in STOPWORDS_FR}


def pretraiter_texte(texte, strategie_ewe="normaliser"):
    # Nettoie un commentaire. strategie_ewe : "conserver" 
    texte = unicodedata.normalize("NFC", texte.lower())   
    texte = re.sub(r"\d+", " ", texte)                    
    texte = re.sub(r"[^\w\s]|_", " ", texte)              
    jetons_finaux = []
    for jeton in texte.split():                           
        jeton_sans_accent = sans_accents(jeton)
        if jeton_sans_accent in LEXIQUE_EWE_MINA:         
            if strategie_ewe == "supprimer":
                continue
            if strategie_ewe == "normaliser":
                if jeton_sans_accent in EWE_MOTS_OUTILS:
                    continue
                jeton = EWE_VERS_FRANCAIS.get(jeton_sans_accent, jeton)
        elif jeton_sans_accent in STOPWORDS_SANS_ACCENTS or len(jeton) < 2:   
            continue
        jetons_finaux.append(jeton)
    return " ".join(jetons_finaux)


def detecter_ewe_mina(texte):
    # Retourne la liste des termes éwé/mina présents dans un commentaire.
    jetons = re.sub(r"[^\w\s]|\d|_", " ", texte.lower()).split()
    return [jeton for jeton in jetons if sans_accents(jeton) in LEXIQUE_EWE_MINA]
