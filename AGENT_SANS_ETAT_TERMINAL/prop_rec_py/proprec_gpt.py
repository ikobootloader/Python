import numpy as np
import matplotlib.pyplot as plt

# Définir les paramètres
R_x, R_y = 2, 3  # Coordonnées de la source de récompense
Z = 100  # Valeur de la récompense
gamma = 0.9  # Facteur d'escompte

# Créer une grille de points (x, y)
x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)
X, Y = np.meshgrid(x, y)

# Calculer la distance de Manhattan à partir du point de récompense
D_manhattan = np.abs(X - R_x) + np.abs(Y - R_y)

# Calculer la valeur de l'état en chaque point
V_e = Z * (gamma ** D_manhattan)

# Créer la figure
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, V_e, cmap='viridis', levels=50)
plt.colorbar(cp, label='Valeur d\'état $V_e(x, y)$')
plt.plot(R_x, R_y, 'ro', markersize=10, label='Source de récompense')
plt.title("Propagation carrée de la récompense")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
