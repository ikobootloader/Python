Pour modéliser le mouvement de l'agent en suivant une spirale d'Archimède sur un repère cartésien, nous pouvons utiliser les équations paramétriques de la spirale d'Archimède adaptées pour une grille cartésienne. La spirale d'Archimède est définie par l'équation polaire $r = a + b\theta$, où $r$ est le rayon, $\theta$ l'angle, et $a$ et $b$ sont des constantes.

Pour un déplacement en coordonnées cartésiennes, on peut utiliser les équations suivantes :

$x(\theta) = (a + b\theta) \cdot \cos(\theta)$ <br/>
$y(\theta) = (a + b\theta) \cdot \sin(\theta)$

Cependant, dans le cas d'un mouvement en grille comme dans l'image, on doit adapter cette approche à un système de grille cartésien où l'agent se déplace par des étapes discrètes (unité par unité).

### Équation de Déplacement en Spirale sur une Grille

Soit $n$ le nombre d'étapes que l'agent a effectuées. Le déplacement en spirale peut être décomposé en segments droits de longueurs successives qui augmentent après chaque changement de direction.

Les coordonnées $(x_n, y_n)$ de l'agent après $n$ étapes peuvent être définies comme suit :

1. **Initialisation** : L'agent commence à la position centrale de la grille $(x_0, y_0) = (0, 0)$.

2. **Déplacement en Spirale** : À chaque étape, l'agent se déplace dans l'une des quatre directions (droite, haut, gauche, bas) et change de direction après avoir parcouru une certaine distance.

L'idée est d'utiliser des segments de longueur fixe, puis d'incrémenter cette longueur après deux segments (deux changements de direction).

#### Position $(x_n, y_n)$ après $n$ étapes :
Les coordonnées de l'agent peuvent être calculées de manière itérative, mais pour une approche plus mathématique :

$(x_n, y_n) = \left(x_{n-1} + \Delta x, y_{n-1} + \Delta y\right)$

Où :
- $\Delta x$ et $\Delta y$ dépendent de la direction dans laquelle l'agent se déplace :
  - **Droite** : $\Delta x = 1$, $\Delta y = 0$
  - **Haut** : $\Delta x = 0$, $\Delta y = 1$
  - **Gauche** : $\Delta x = -1$, $\Delta y = 0$
  - **Bas** : $\Delta x = 0$, $\Delta y = -1$
  
Le changement de direction et la longueur de chaque segment peuvent être déterminés comme suit :
- **Changement de direction** : tous les $m$ pas, où $m$ est la longueur du segment actuel.
- **Longueur du segment** : après deux changements de direction, la longueur augmente de 1.

### Formule de position à l'étape $n$ :

Pour formaliser :

1. **Étape initiale** : $n = 0$ (centre de la grille)
   $x_0 = 0, \quad y_0 = 0$
   
2. **À chaque étape $n$** :
   - Identifier la direction en fonction du nombre de segments parcourus.
   - Ajouter l'incrément correspondant aux coordonnées précédentes $(x_{n-1}, y_{n-1})$.

### Implémentation Mathématique

Le nombre de segments $k$ nécessaires pour atteindre $n$ peut être trouvé via :

$k = \left\lfloor \frac{n + 1}{2} \right\rfloor$

La longueur du segment est donnée par :

$L = \frac{k + 1}{2}$

Ainsi, la direction (droite, haut, gauche, bas) change tous les $L$ pas.

L'équation mathématique exacte pour chaque pas est complexe à généraliser directement à partir d'une seule formule en raison de la nature itérative du mouvement (changement de direction tous les $L$ pas). C’est pourquoi une approche itérative est souvent plus pratique en programmation pour simuler ce déplacement, comme montré dans le code Python.

**Résumé** :
- Le déplacement en spirale nécessite de savoir quelle direction est en cours et combien de segments ont été parcourus pour ajuster la direction.
- Il est plus naturel de décrire le mouvement de manière itérative avec un algorithme, qui est ce qui est typiquement fait en programmation.
