# Reconnaissance des émotions faciales avec un CNN

## 1. Démarche

L'objectif de ce projet est de développer un modèle de **Deep Learning capable de reconnaître l'émotion présente sur une image de visage**.

Le problème est traité comme une tâche de **classification multi-classes** avec 7 émotions :

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

Les images du dataset sont des images en niveaux de gris de taille **48 × 48 pixels**.

La démarche suivie est la suivante :

1. **Exploration et préparation des données**

   * Analyse de la répartition des différentes classes.
   * Redimensionnement des images en 48 × 48 pixels.
   * Conversion en niveaux de gris.
   * Normalisation des pixels entre 0 et 1.
   * Séparation des données en ensembles d'entraînement et de validation.

2. **Création d'un modèle de référence**

   * Un premier réseau simple est utilisé afin d'avoir une base de comparaison.

3. **Création d'un CNN**

   * Un réseau de neurones convolutionnel est développé pour mieux exploiter les caractéristiques spatiales des images.

4. **Analyse des performances**

   * Comparaison des performances sur les ensembles d'entraînement, de validation et de test.
   * Analyse du surapprentissage et des erreurs de classification.

5. **Amélioration du modèle**

   * Plusieurs expériences sont réalisées avec différentes techniques de régularisation et d'optimisation.

---

## 2. Architectures

### Modèle de référence

Le premier modèle est volontairement simple :

```text
Image 48 × 48 × 1
        ↓
     Flatten
        ↓
   Dense 128
      ReLU
        ↓
   Dense 7
    Softmax
```

Ce modèle sert principalement de **baseline** pour pouvoir mesurer l'apport du CNN.

---

### CNN

Le modèle principal utilise plusieurs couches convolutives :

```text
Image 48 × 48 × 1
        ↓
Conv2D - 32 filtres
        ↓
MaxPooling
        ↓
Conv2D - 64 filtres
        ↓
MaxPooling
        ↓
Conv2D - 128 filtres
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense 128
        ↓
Dense 7 - Softmax
```

Le nombre de filtres augmente progressivement :

**32 → 64 → 128**

Les couches convolutionnelles permettent d'extraire progressivement des caractéristiques visuelles de plus en plus complexes.

La dernière couche contient **7 neurones**, correspondant aux 7 émotions à reconnaître.

---

## 3. Expériences

Le CNN de base présente un problème important de **surapprentissage (overfitting)** : les performances sur les données d'entraînement deviennent très bonnes alors que les performances sur les données de validation restent beaucoup plus faibles.

Plusieurs expériences ont donc été réalisées.

### Expérience 1 — Dropout

Un **Dropout de 0,5** est ajouté après la couche Dense.

Le Dropout désactive aléatoirement une partie des neurones pendant l'entraînement afin d'éviter que le modèle mémorise trop les données d'entraînement.

**Objectif :** améliorer la généralisation du modèle.

---

### Expérience 2 — Data Augmentation + Dropout

La deuxième expérience ajoute de la **data augmentation** au modèle précédent.

Les images sont légèrement modifiées pendant l'entraînement :

* rotations ;
* translations ;
* zoom ;
* retournement horizontal.

L'objectif est de créer davantage de variations des images et ainsi de limiter le surapprentissage.

---

### Expérience 3 — Data Augmentation + Dropout + Early Stopping

La dernière expérience conserve les améliorations précédentes et ajoute :

* **Early Stopping** : arrêt de l'entraînement lorsque les performances de validation n'améliorent plus ;
* **ReduceLROnPlateau** : diminution du learning rate lorsque l'apprentissage stagne.

L'objectif est de rendre l'entraînement plus stable et d'éviter de continuer à entraîner le modèle une fois que celui-ci commence à surapprendre.

---

## 4. Résultats

Les différentes expériences ont donné les résultats suivants :

| Modèle           | Modifications                                     | Accuracy validation | Loss validation |
| ---------------- | ------------------------------------------------- | ------------------: | --------------: |
| CNN de référence | Aucune                                            |             53,72 % |          1,2222 |
| Expérience 1     | Dropout                                           |         **55,44 %** |          1,2038 |
| Expérience 2     | Data Augmentation + Dropout                       |             52,85 % |          1,2262 |
| Expérience 3     | Data Augmentation + Dropout + Early Stopping + LR |             55,36 % |      **1,1670** |

Le CNN de base obtient environ **49,38 % d'accuracy sur le jeu de test**.

### Analyse

L'ajout du **Dropout** permet d'obtenir la meilleure accuracy de validation avec **55,44 %**.

La dernière expérience obtient une accuracy très proche (**55,36 %**), mais possède la **meilleure loss de validation (1,1670)**.

La data augmentation seule n'a pas apporté l'amélioration attendue dans cette configuration.

Ces résultats montrent surtout que le principal problème du modèle est le **surapprentissage**. Les techniques de régularisation permettent donc d'améliorer la généralisation du réseau.

---

---

## 5. Ajout de Yolo

Utilisation d'un modèle pré entrainée de Yolo pour la détection de visage sur une image contenant plusieurs personne puis utilisation de notre modèle pour prédire les différentes émotions. 

---

## Conclusion

Ce projet a permis de mettre en place un pipeline complet de reconnaissance d'émotions :

**Préparation des données → CNN → Entraînement → Évaluation → Amélioration**

Les expériences montrent que l'utilisation de techniques comme le **Dropout**, l'**Early Stopping** et l'ajustement du **learning rate** peut améliorer la capacité du modèle à généraliser sur de nouvelles images.

Le meilleur résultat en validation est obtenu avec le **Dropout**, avec une accuracy de **55,44 %**.
