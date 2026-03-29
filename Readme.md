# 📝 TAL-Projet : Analyse de Sentiments & Stylométrie Politique

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626)

Ce dépôt contient les travaux réalisés par **Isaac KINANE** et **Samy MANCER** dans le cadre du projet de **Traitement Automatique du Langage (TAL)**. 

Le projet est divisé en deux volets majeurs explorant l'évolution des techniques NLP, des modèles statistiques classiques (TF-IDF, SVM) jusqu'aux architectures profondes de pointe (Transformers, Bi-RNN).

---

## 🚀 Aperçu du Projet

### 🎬 Partie 1 : Analyse de Sentiments (Critiques de Films)
L'objectif est de prédire la polarité (Positif/Négatif) de critiques cinématographiques.
* **Fichier principal :** `movies_part.ipynb`
* **Approches classiques :** Comparaison de Bag of Words (BoW) et TF-IDF avec des classifieurs tels que SVM, Régression Logistique et XGBoost.
* **Deep Learning & Transformers :** Utilisation d'Embeddings denses (Sentence-BERT) et *fine-tuning* de RoBERTa (Siebert).
* **✨ Innovation technique :** Implémentation d'une troncature asymétrique **"Head + Tail"** (25% intro / 75% conclusion) pour contourner la limite des 512 tokens des Transformers tout en préservant le verdict émotionnel du spectateur.

### 🏛️ Partie 2 : Identification d'Auteurs (Chirac vs Mitterrand)
L'objectif est de classifier des segments de discours politiques pour identifier le locuteur, en gérant un fort déséquilibre de classes (Ratio 1:6.6).
* **Fichiers principaux :** Dossier `President_Task/`, `4a-RNNs.ipynb`, `4b_transformers_my_correction_fine_tuning.ipynb`.
* **Modélisation séquentielle :** Utilisation de **Bi-RNN (GRU)** pour capturer la dynamique temporelle du dialogue (le contexte des phrases précédentes et suivantes).
* **Modèle de Langage :** *Fine-tuning* de **CamemBERT** avec une ingénierie de données avancée (fenêtrage contextuel et masquage dynamique).
* **✨ Post-traitement markovien :** Application d'un algorithme de lissage *Forward-Backward* via des **Modèles de Markov Cachés (HMM)** pour stabiliser les prédictions conversationnelles.
* *Note : Les plongements lexicaux denses pour cette partie utilisent le modèle FastText `cc.fr.300.vec`.*

---

## 📂 Structure du Dépôt

L'arborescence du projet s'organise autour de plusieurs Notebooks Jupyter retraçant la progression de nos expérimentations :

```text
📦 TAL-Projet
 ┣ 📂 President_Task/                # Données/Scripts liés à la classification politique
 ┣ 📜 movies_part.ipynb              # Pipeline complet Analyse de Sentiments (Partie 1)
 ┣ 📜 2a-Sequences.ipynb             # Modélisations séquentielles basiques
 ┣ 📜 2b-Clustering.ipynb            # Expérimentations de clustering NLP
 ┣ 📜 3a-representationLearning.ipynb# Apprentissage de représentations (FastText, TF-IDF)
 ┣ 📜 3b-Sequences2.ipynb            # Approfondissement sur les séquences
 ┣ 📜 4a-RNNs.ipynb                  # Modèles Bi-RNN / GRU (Partie 2)
 ┣ 📜 4b_transformers_my_correction_fine_tuning.ipynb # Fine-tuning CamemBERT (Partie 2)
 ┣ 📜 test.py                        # Scripts de tests / fonctions utilitaires
 ┗ 📜 README.md                      # Documentation du projet
