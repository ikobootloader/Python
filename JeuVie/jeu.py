import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Dimensions de la grille
N = 100
M = 100

# Fonction pour initialiser la grille aléatoirement
def initial_state(N, M):
    state = np.random.randint(2, size=(N, M))
    return state

# Fonction pour calculer l'état suivant du jeu de la vie
def next_state(state):
    neighbors_count = sum(np.roll(np.roll(state, i, 0), j, 1)
                          for i in (-1, 0, 1) for j in (-1, 0, 1)
                          if (i != 0 or j != 0))
    return (neighbors_count == 3) | (state & (neighbors_count == 2))

# Initialisation de l'état initial
state = initial_state(N, M)

# Mise en place du graphique
fig, ax = plt.subplots()
img = ax.imshow(state, cmap='binary', interpolation='nearest')
plt.axis('off')

# Fonction d'animation pour mettre à jour l'état du jeu
def update(frameNum, img, state):
    new_state = next_state(state)
    img.set_data(new_state)
    state[:] = new_state[:]
    return img,

# Animation
ani = animation.FuncAnimation(fig, update, frames=100, fargs=(img, state), interval=50)
plt.show()
