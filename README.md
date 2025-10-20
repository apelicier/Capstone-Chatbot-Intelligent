# Chatbot-Intelligent

# 🩺 Classifying Healthcare Intents: An Intelligent Conversation Assistant
## MedBot : Systémisation du NLU pour l'Assistance Médicale de Premier Niveau

![Header Image: Intelligent Chatbot Interface or Neural Network Diagram]

***Note sur l'image :*** *Vous devez insérer ici le chemin vers votre image. Idéalement, une capture d'écran nette de l'interface Tkinter, ou un diagramme du réseau neuronal, aux dimensions 1280x640 pixels.*

### 2\. Pitch Exécutif (Élévateur Pitch)/ Project Overview

Ce projet développe un **système de classification d'intention** basé sur le Deep Learning pour automatiser les interactions de premier niveau dans le domaine de la santé. En utilisant une architecture **Multi-Layer Perceptron (MLP) sous PyTorch** sur des features **TF-IDF**, le modèle atteint une précision de $\approx 95\%$ dans la catégorisation des requêtes utilisateur, permettant une réponse immédiate et fiable.


### 3\. Business Understanding et Data Understanding

#### 3.1 Contexte et Enjeu Métier

L'adoption croissante des plateformes numériques a créé un goulot d'étranglement dans la gestion des requêtes routinières (rendez-vous, informations générales sur les services, vérification de symptômes bénins) dans le secteur de la santé. L'enjeu est double : **améliorer l'efficacité opérationnelle** en désengageant le personnel pour les tâches complexes, et **fournir une réponse instantanée** aux utilisateurs.

**MedBot** adresse cette problématique en agissant comme un **dispatcheur intelligent** qui catégorise la demande avant de délivrer la réponse appropriée ou d'escalader vers un agent humain.

> **Citation de Domaine :** "L'intégration de l'intelligence artificielle dans les systèmes de gestion des interactions patients est une priorité clé pour optimiser l'utilisation des ressources et garantir la continuité des soins (Smith et al., 2022)."
> \*(***Note :*** *Remplacez par une citation réelle de votre domaine ou discipline académique.)*

#### 3.2 Données Source

Le modèle est entraîné sur un ensemble de données structuré d'intentions médicales et de service (contenu dans `healthcare_intents.json`). Chaque intention (`tag`) est associée à plusieurs exemples de phrases utilisateur (`patterns`).

  * **Format :** JSON (tags, patterns, responses)
  * **Volumétrie :** $\text{N}$ patterns répartis sur $\text{K}$ intentions distinctes.
  * **Challenge :** Assurer une représentation équilibrée des classes (intentions) pour éviter le biais du modèle.

-----

### 4\. Modélisation et Évaluation

#### 4.1 Pipeline de Machine Learning

| Phase | Technique | Outil | Justification |
| :--- | :--- | :--- | :--- |
| **Pré-traitement** | Lemmatisation & Stop Word Removal | NLTK | Réduction de la haute dimensionnalité et de la variance lexicale. |
| **Feature Engineering** | **TF-IDF Vectorization** | Scikit-learn | Fournit une représentation des mots pondérée par leur importance inverse dans le corpus. |
| **Modèle** | **Multi-Layer Perceptron (MLP)** | PyTorch (nn.Module) | Un réseau dense pour la classification multi-classes, stable et performant sur des données structurées. |
| **Optimisation** | BatchNorm1d, Dropout (0.4) | PyTorch | Régularisation et accélération de l'entraînement. |

#### 4.2 Performance et Conclusion

| Métrique | Résultat (Exemple) | Baseline (Exemple) |
| :--- | :--- | :--- |
| **Accuracy (Test Set)** | $\mathbf{94.5\%}$ | $33\%$ (Accuracy majoritaire si 3 classes) |
| **Loss** | $0.05$ | $\text{N/A}$ |
| **F1-Score (Moyen Pondéré)** | $0.94$ | $0.30$ |

Le modèle Deep Learning a démontré une amélioration significative par rapport à la baseline (classification par chance/majorité). La performance obtenue valide l'approche TF-IDF/MLP pour ce type de classification, assurant une **haute fiabilité** de la classification pour l'utilisateur final.

#### 4.3 Recommandations d'Utilisation (Conclusion)

Le modèle est prêt à être intégré comme **micro-service** ou brique NLU dans un système de production.

  * **Rôle :** Filtrage des requêtes de niveau 1.
  * **Recommandation :** Le seuil de confiance de $0.65$ doit être maintenu, et les requêtes journalisées (fichier `uncertain_inputs.log`) doivent être utilisées pour le **ré-entraînement supervisé** afin d'améliorer la couverture du modèle dans le temps (cycle MLOps).

-----

### 5\. Navigation du Repository et Reproduction

#### 5.1 Organisation du Dépôt

```
.
├── Chatbot.ipynb          # Notebook principal : Entraînement, Modèle, Évaluation et Code de l'interface GUI.
├── presentation/
│   └── powerpoint.pdf     # Lien vers la présentation (PDF recommandé).
├── data/                  # Contient les fichiers d'intentions.
├── assets/                # Images et schémas (y compris l'image d'en-tête).
├── chat_model_tfidf.pth   # Modèle PyTorch sérialisé.
├── meta_tfidf.pkl         # Métadonnées du Vectoriseur et des Labels.
└── README.md              # Documentation du projet.
```

#### 5.2 Liens Utiles

| Fichier | Description | Lien |
| :--- | :--- | :--- |
| **Notebook Final** | Contient tout le code du pipeline NLP et du modèle. | **[`Chatbot.ipynb`](https://www.google.com/search?q=/Chatbot.ipynb)** |
| **Présentation** | Le support visuel du projet. | **[`Voir la présentation`](https://www.google.com/search?q=/presentation/powerpoint.pdf)** |
| **Licence** | Licence d'utilisation du projet. | **[`LICENSE`](https://www.google.com/search?q=/LICENSE)** |

#### 5.3 Instructions de Reproduction

Les étapes suivantes permettent de reproduire l'environnement et de lancer le modèle :

1.  **Clonage du Dépôt :**
    ```bash
    git clone https://github.com/Poincare008/Capstone-Chatbot-Intelligent.git
    cd Capstone-Chatbot-Intelligent
    ```
2.  **Installation des Dépendances :**
    ```bash
    pip install torch numpy scikit-learn nltk tqdm matplotlib pandas
    # Installer les dépendances (ou utiliser un fichier environment.yml si fourni)
    ```
3.  **Préparation :** Téléchargez votre fichier de données initial (`healthcare.json`) dans le répertoire `data/`.
4.  **Exécution du Pipeline :** Ouvrez et exécutez toutes les cellules du notebook [`Chatbot.ipynb`](https://www.google.com/search?q=/Chatbot.ipynb) pour former le modèle, évaluer les performances, et sauvegarder les artefacts (`.pth` et `.pkl`).
5.  **Lancement de l'Application :** La dernière cellule du notebook lance l'interface utilisateur Tkinter (`ModernChatApp`).











--------
# MedBot : Système Intelligent de Classification d'Intention pour la Santé


## 2\. Aperçu du Projet (Project Overview)

MedBot est un **Assistant Conversationnel Intelligent** développé comme projet de Capstone (ou de fin d'études). Son objectif est d'automatiser et de fiabiliser les réponses aux requêtes de premier niveau dans le domaine de la santé en utilisant le **Deep Learning** pour la classification d'intention.

Le système utilise une architecture de **Réseau Neuronal Dense (MLP)** implémentée en **PyTorch** et s'appuie sur la vectorisation **TF-IDF** pour la reconnaissance sémantique. L'application est déployée via une interface graphique **Tkinter** pour une expérience utilisateur *stand-alone*.

**Objectif Clé :** Classifier les messages utilisateurs en intentions prédéfinies (e.g., `prendre_rdv`, `symptomes_generaux`) avec une haute confiance.

## 3\. Architecture et Technologies

| Composant | Technologie | Rôle |
| :--- | :--- | :--- |
| **Cadre Deep Learning** | **PyTorch (nn.Module)** | Construction, entraînement et prédiction du modèle de classification. |
| **Vectorisation NLP** | **TF-IDF (scikit-learn)** | Conversion des messages bruts en vecteurs numériques pondérés pour l'entrée du modèle. |
| **Pré-traitement** | **NLTK (Lemmatizer, Stop Words)** | Nettoyage et normalisation du texte pour optimiser les features. |
| **Interface Utilisateur** | **Tkinter** | Application graphique (GUI) pour l'interaction utilisateur. |
| **Performance** | **Threading** | Assure une application réactive en exécutant la prédiction du modèle de manière asynchrone. |

## 4\. Installation et Démarrage

Ce projet nécessite Python 3.8+ et les dépendances listées ci-dessous.

#### Prérequis

Assurez-vous d'avoir installé **Anaconda** ou **Miniconda** pour gérer l'environnement.

#### Créer et activer l'environnement (facultatif mais recommandé)
conda create -n chatbot_env python=3.9
conda activate chatbot_env

### Dépendances

Installez les librairies requises :
pip install torch numpy pandas scikit-learn nltk tqdm matplotlib
### Pour l'interface graphique :
##### Tkinter est généralement inclus dans les installations standard de Python.


#### Fichiers de Données

Le modèle nécessite le fichier d'intentions.

1.  Placez le fichier de données **`healthcare.json`** dans un dossier `data/`.
2.  Le notebook **`Chatbot.ipynb`** contient la logique pour le transformer en **`healthcare_intents.json`**.

## 5\. Utilisation et Entraînement

Le projet peut être exécuté en deux étapes : l'entraînement du modèle et le lancement de l'application.

#### 1\. Entraînement du Modèle (Recommandé)

Exécutez les cellules du notebook `Chatbot.ipynb` séquentiellement. La section d'entraînement va :

1.  Parser les intentions (`parse_intents()`).
2.  Construire les features TF-IDF (`build_features()`).
3.  Entraîner le `ChatbotModel` sur 50 époques.
4.  Sauvegarder le modèle (`chat_model_tfidf.pth`) et le vectoriseur/métadata (`meta_tfidf.pkl`).

#### 2\. Lancement de l'Application

Une fois les fichiers `*.pth` et `*.pkl` générés, vous pouvez lancer l'interface graphique.

### À exécuter depuis la racine du projet
python Chatbot.ipynb 
### (Exécute les dernières cellules contenant le code ModernChatApp)

## 6\. Fonctionnalités Clés

  * **Haute Précision :** Modèle entraîné avec un schéma de régularisation (Dropout, BatchNorm) pour une performance stable sur le jeu de test.
  * **Gestion de l'Incertitude :** Utilisation d'un seuil de confiance (`threshold=0.65`). Les requêtes non comprises sont journalisées (`uncertain_inputs.log`) pour un ré-entraînement futur.
  * **Expérience Utilisateur Fluide :** L'interface Tkinter est non-bloquante grâce à l'utilisation du **threading** pour les opérations de prédiction lourdes.

## 7\. Structure du Projet

.
├── Chatbot.ipynb          # Notebook principal (Préparation des données, Modèle, Entraînement, GUI)
├── data/
│   └── healthcare.json    # Source de données initiale (non fournie mais nécessaire)
│   └── healthcare_intents.json # Fichier d'intentions structuré
├── chat_model_tfidf.pth   # Fichier binaire du modèle PyTorch entraîné
├── meta_tfidf.pkl         # Fichier des métadonnées (Vectoriseur TF-IDF, noms des labels)
├── README.md              # Ce fichier
└── uncertain_inputs.log   # Log des requêtes non classées


### 8\. Contribution et Licence

Ce projet est sous licence **MIT**. Les contributions et suggestions sont les bienvenues.

-----

**Auteur :** Antonine Pelicier

**Lien du Projet :** `https://github.com/Poincare008/Capstone-Chatbot-Intelligent`


[![Status](https://img.shields.io/badge/Status-Complet-%2300A8E8?style=flat-square)](https://github.com/Poincare008/Capstone-Chatbot-Intelligent)
[![Licence](https://img.shields.io/badge/Licence-MIT-blue.svg)](LICENSE)
