import time
import os

def clear_console():
    # Fonction pour effacer la console pour simuler une animation
    os.system('cls' if os.name == 'nt' else 'clear')

def spiral_movement(n, delay=0.1):
    x, y = n // 2, n // 2  # Point de départ au centre de la grille
    dx, dy = 1, 0  # Mouvement initial vers la droite
    length = 1  # Longueur initiale de la branche
    steps = 0  # Nombre de pas effectués dans la direction courante
    direction_changes = 0  # Nombre de changements de direction
    
    grid = [[0 for _ in range(n)] for _ in range(n)]
    grid[y][x] = 1  # Marque la position initiale

    while True:
        clear_console()  # Efface la console pour afficher la grille mise à jour
        # Afficher la grille
        for row in grid:
            print(' '.join(str(cell) for cell in row))
        time.sleep(delay)  # Attendre un moment avant la prochaine étape

        # Effectuer un déplacement
        x += dx
        y += dy
        steps += 1

        if 0 <= x < n and 0 <= y < n:
            grid[y][x] = 1  # Marque la position de l'agent
        else:
            break  # Arrêter si l'agent sort de la grille

        if steps == length:
            # Changer de direction (droite -> haut -> gauche -> bas -> droite)
            dx, dy = -dy, dx
            steps = 0
            direction_changes += 1

            if direction_changes % 2 == 0:
                # Augmenter la longueur après deux changements de direction
                length += 1

# Exemple d'utilisation
n = 11  # Taille de la grille
spiral_movement(n, delay=0.5)  # Délai de 0.5 seconde entre chaque étape
