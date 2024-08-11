import numpy as np
import matplotlib.pyplot as plt

def create_reward_matrix(letter_image, gamma=0.9):
    """Crée une matrice de récompenses pour une lettre en propagant la récompense des pixels actifs."""
    reward_matrix = np.zeros_like(letter_image, dtype=float)
    indices = np.argwhere(letter_image > 0)  # Trouver tous les pixels actifs
    
    for index in indices:
        x, y = index
        for i in range(letter_image.shape[0]):
            for j in range(letter_image.shape[1]):
                distance = abs(x - i) + abs(y - j)  # Utilisation de la distance de Manhattan
                reward_matrix[i, j] += (gamma ** distance) * letter_image[x, y]
    
    return reward_matrix

# Exemple de lettre manuscrite (simplifié)
letter_A = np.array([
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1]
])

# Créer la matrice de récompenses
reward_matrix_A = create_reward_matrix(letter_A, gamma=0.9)

# Visualiser la matrice de récompenses
plt.figure(figsize=(6, 6))
plt.imshow(reward_matrix_A, cmap='hot', interpolation='nearest')
plt.title("Matrice de récompenses pour la lettre A")
plt.colorbar(label='Intensité de la récompense')
plt.show()
