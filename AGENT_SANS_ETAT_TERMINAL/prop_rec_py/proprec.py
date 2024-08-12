import numpy as np
import matplotlib.pyplot as plt

def calculate_value(x, y, reward_x, reward_y, Z, gamma):
    if x == reward_x and y == reward_y:
        return Z
    return gamma ** (abs(x - reward_x) + abs(y - reward_y)) * Z

def generate_reward_map(size, reward_x, reward_y, Z, gamma):
    reward_map = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            reward_map[y, x] = calculate_value(x, y, reward_x, reward_y, Z, gamma)
    return reward_map

# Paramètres
size = 5
reward_x, reward_y = 2, 2
Z = 100
gamma = 0.9

# Générer la carte de récompense
reward_map = generate_reward_map(size, reward_x, reward_y, Z, gamma)

# Afficher la carte de récompense
plt.figure(figsize=(10, 8))
plt.imshow(reward_map, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Valeur')
plt.title('Propagation de la récompense')
plt.xlabel('X')
plt.ylabel('Y')

# Ajouter les valeurs sur la carte
for x in range(size):
    for y in range(size):
        plt.text(x, y, f'{reward_map[y, x]:.2f}', ha='center', va='center', color='white')

plt.tight_layout()
plt.show()

# Afficher les valeurs numériques
print("Carte des valeurs:")
print(reward_map)