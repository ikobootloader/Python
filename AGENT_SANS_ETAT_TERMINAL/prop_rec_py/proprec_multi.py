import numpy as np
import matplotlib.pyplot as plt

def calculate_value(x, y, rewards, gamma):
    values = []
    for rx, ry in rewards:
        if x == rx and y == ry:
            values.append(100)  # Toutes les récompenses ont une valeur de 100
        else:
            values.append(gamma ** (abs(x - rx) + abs(y - ry)) * 100)
    return max(values)

def generate_reward_map(size, rewards, gamma):
    reward_map = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            reward_map[y, x] = calculate_value(x, y, rewards, gamma)
    return reward_map

# Paramètres
size = 5
rewards = [(1, 1), (3, 3)]  # (x, y) des récompenses
gamma = 0.9

# Générer la carte de récompense
reward_map = generate_reward_map(size, rewards, gamma)

# Afficher la carte de récompense
plt.figure(figsize=(10, 8))
plt.imshow(reward_map, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Valeur')
plt.title('Propagation de multiples récompenses (valeur 100)')
plt.xlabel('X')
plt.ylabel('Y')

# Ajouter les valeurs sur la carte
for x in range(size):
    for y in range(size):
        plt.text(x, y, f'{reward_map[y, x]:.1f}', ha='center', va='center', color='white')

# Marquer les positions des récompenses
for rx, ry in rewards:
    plt.plot(rx, ry, 'r*', markersize=15)
    plt.text(rx, ry, 'R:100', ha='center', va='bottom', color='red', fontweight='bold')

plt.tight_layout()
plt.show()

print("Carte des valeurs:")
print(reward_map)