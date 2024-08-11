Algorithme de propagation de la récompense dans un espace de coordonnées cartésiennes pour un agent effectuant des actions discrètes (déplacement gauche, droite, haut, bas). Cet algorithme est utilisé dans le cadre d'un apprentissage par renforcement, où il n'y a pas d'état terminal.

### Explication de l'algorithme

L'objectif de l'algorithme est de propager la valeur de la récompense (notée $\mathbb{Z}$) depuis un point de récompense donné $(R(x), R(y))$ sur tout le territoire $T$. Pour chaque état $(x, y)$ de ce territoire, la valeur d'état $V_e$ est mise à jour en fonction d'un facteur d'escompte $\gamma$ (ici $\gamma = 0.90$).

#### Règles de mise à jour

Pour chaque état $V_e$ dans $T$, tant que $V_e$ n'est pas égal à la récompense $\mathbb{Z}$, la mise à jour se fait selon les conditions suivantes :

1. **Si $y \leq R(y)$ et $x \leq R(x)$** :
   $
   (R(y) - y, R(x) - x) = \gamma^{x+y}
$

2. **Si $y \leq R(y)$ et $x \leq \text{argmax}(x) - R(x)$** :
   $
   (R(y) - y, R(x) + x) = \gamma^{x+y}
$

3. **Si $y \leq \text{argmax}(y) - R(y)$ et $x \leq R(x)$** :
   $
   (R(y) + y, R(x) - x) = \gamma^{x+y}
$

4. **Si $y \leq \text{argmax}(y) - R(y)$ et $x \leq \text{argmax}(x) - R(x)$** :
   $
   (R(y) + y, R(x) + x) = \gamma^{x+y}
$

### Équation Générale

À partir de ces règles, l'équation générale pour la valeur d'état $V_e$ peut être écrite comme une fonction de la distance par rapport au point de récompense $(R(x), R(y))$, pondérée par le facteur d'escompte $\gamma$. Si l'on généralise les conditions :

$$
V_e(x, y) = \gamma^{|x - R(x)| + |y - R(y)|} \cdot \mathbb{Z}
$$

***

Pour introduire plusieurs récompenses dans l'équation de la valeur d'état, nous devons prendre en compte les contributions de chacune des récompenses potentielles, en fonction de leur position respective et de leur valeur. L'idée est de calculer la contribution de chaque récompense séparément en fonction de la distance à l'état considéré, puis de les additionner pour obtenir la valeur totale de l'état.

### Équation pour plusieurs récompenses

Supposons que nous ayons plusieurs récompenses situées à différents points $(R_1(x), R_1(y))$, $(R_2(x), R_2(y))$, ..., $(R_n(x), R_n(y))$ dans l'espace. Les valeurs de ces récompenses sont $\mathbb{Z}_1$, $\mathbb{Z}_2$, ..., $\mathbb{Z}_n$ respectivement.

La valeur d'état $V_e(x, y)$ en un point $(x, y)$ sera alors la somme des contributions de chacune de ces récompenses, pondérées par le facteur d'escompte $\gamma$ en fonction de la distance à chaque récompense.

L'équation générale devient :

$$
V_e(x, y) = \sum_{i=1}^{n} \gamma^{|x - R_i(x)| + |y - R_i(y)|} \cdot \mathbb{Z}_i
$$

### Explication

- $n$ : Nombre de récompenses.
- $(R_i(x), R_i(y))$ : Coordonnées de la $i$-ème récompense.
- $\mathbb{Z}_i$ : Valeur de la $i$-ème récompense.
- $\gamma$ : Facteur d'escompte.
- $|x - R_i(x)| + |y - R_i(y)|$ : Distance entre le point $(x, y)$ et la $i$-ème récompense.

### Exemple de calcul

Supposons que :
- Il y a deux récompenses : 
  - $(R_1(x), R_1(y)) = (2, 3)$ avec $\mathbb{Z}_1 = 50$
  - $(R_2(x), R_2(y)) = (5, 6)$ avec $\mathbb{Z}_2 = 30$
- Le facteur d'escompte est $\gamma = 0.9$.
- Nous voulons calculer la valeur $V_e(x, y)$ pour le point $(x, y) = (3, 4)$.

**Étapes** :

1. **Calcul de la contribution de la première récompense** :
   $$
   d_1 = |3 - 2| + |4 - 3| = 1 + 1 = 2
$$
   $$
   \text{Contribution de } \mathbb{Z}_1 = \gamma^2 \cdot 50 = 0.9^2 \cdot 50 = 0.81 \cdot 50 = 40.5
$$

2. **Calcul de la contribution de la deuxième récompense** :
   $$
   d_2 = |3 - 5| + |4 - 6| = 2 + 2 = 4
$$
   $$
   \text{Contribution de } \mathbb{Z}_2 = \gamma^4 \cdot 30 = 0.9^4 \cdot 30 = 0.6561 \cdot 30 = 19.683
$$

3. **Calcul de la valeur totale $V_e(x, y)$** :
   $$
   V_e(3, 4) = 40.5 + 19.683 = 60.183
$$

### Résultat

La valeur d'état $V_e(3, 4)$ en prenant en compte les deux récompenses est donc de 60,183.

### Interprétation

Cette approche permet de prendre en compte plusieurs récompenses et leurs effets combinés sur la valeur d'un état donné. Plus une récompense est proche d'un point $(x, y)$, plus son impact sera important, grâce à la pondération par $\gamma$.

***

Il serait possible d'utiliser ce procédé pour faire de la reconnaissance de texte : chaque point de récompense représente un pixel par exemple d'une lettre manuscrite. La lettre peut être écrite de différente manière, il faudrait "entraîner" notre matrice de récompenses (avec un calcul de densité de présence corrélée à la quantité de cas d'entraînement) dont les variations de la propagation de γ créeraient des zones qui seraient tantôt plus chaudes (correspondantes à une probabilité d'apparition élevée d'un pixel de la lettre) et donc plus froides à mesure que l'on s'éloigne d'elles.
Ici, $\gamma^{|x - R(x)| + |y - R(y)|}$ représente la diminution de la valeur de la récompense à mesure que l'on s'éloigne du point $(R(x), R(y))$, et $\mathbb{Z}$ est la valeur maximale de la récompense au point $(R(x), R(y))$.

Ce modèle assume que l'agent suit des règles de propagation selon la distance en termes de coordonnées cartésiennes par rapport au point de récompense, avec une décroissance exponentielle déterminée par $\gamma$.
