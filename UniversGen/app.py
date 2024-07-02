import pygame
import numpy as np
import sys
import random

# Initialisation de Pygame
pygame.init()

# Dimensions de la fenêtre
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Simulation de l\'Univers')

# Couleurs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Constantes de l'Univers
H0 = 70  # Constante de Hubble (km/s)/Mpc
Omega_m = 0.3
Omega_lambda = 0.7
G = 6.67430e-11  # Constante gravitationnelle en m^3 kg^-1 s^-2

# Paramètres de temps
t = 0  # Temps initial (en milliards d'années)
dt = 0.1  # Pas de temps initial

# Facteur de zoom initial
zoom_factor = 1.0

# Fonction pour calculer le facteur d'échelle
def scale_factor(t, H0, Omega_m, Omega_lambda):
    a = (Omega_m / Omega_lambda)**(1/3) * (np.sinh(1.5 * np.sqrt(Omega_lambda) * H0 * t / 3e5))**(2/3)
    return a

# Création d'un ensemble de galaxies aléatoires
num_galaxies = 1000
galaxies = np.random.rand(num_galaxies, 2) * 2 - 1  # Coordonnées x, y entre -1 et 1
galaxy_colors = [WHITE for _ in range(num_galaxies)]
selected_galaxy = None

# Fonction pour simuler une supernova
def trigger_supernova(index):
    galaxy_colors[index] = RED
    pygame.time.set_timer(pygame.USEREVENT + index, 2000)

# Fonction pour gérer la fin de la supernova
def end_supernova(index):
    galaxy_colors[index] = WHITE

# Fonction pour calculer la force gravitationnelle entre deux galaxies
def gravitational_force(pos1, pos2):
    r = np.linalg.norm(pos2 - pos1)
    if r == 0:
        return np.array([0.0, 0.0])
    F = G / r**2
    return F * (pos2 - pos1) / r

# Boucle principale
running = True
paused = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_UP:
                dt *= 2  # Augmenter la vitesse
            elif event.key == pygame.K_DOWN:
                dt /= 2  # Diminuer la vitesse
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                zoom_factor *= 1.1  # Zoomer
            elif event.key == pygame.K_MINUS or event.key == pygame.K_UNDERSCORE:
                zoom_factor /= 1.1  # Dézoomer
            elif event.key == pygame.K_s:
                # Déclencher une supernova aléatoire
                random_index = random.randint(0, num_galaxies - 1)
                trigger_supernova(random_index)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Sélectionner une galaxie avec le bouton gauche de la souris
                mouse_x, mouse_y = event.pos
                for i, galaxy in enumerate(galaxies):
                    x = int(width / 2 + galaxy[0] * width / 2 * scale_factor(t, H0, Omega_m, Omega_lambda) * zoom_factor)
                    y = int(height / 2 + galaxy[1] * height / 2 * scale_factor(t, H0, Omega_m, Omega_lambda) * zoom_factor)
                    if (x - mouse_x)**2 + (y - mouse_y)**2 < 100:  # Détecter un clic proche de la galaxie
                        selected_galaxy = i
                        break
            elif event.button == 3:  # Déclencher une collision avec le bouton droit de la souris
                mouse_x, mouse_y = event.pos
                for i, galaxy in enumerate(galaxies):
                    x = int(width / 2 + galaxy[0] * width / 2 * scale_factor(t, H0, Omega_m, Omega_lambda) * zoom_factor)
                    y = int(height / 2 + galaxy[1] * height / 2 * scale_factor(t, H0, Omega_m, Omega_lambda) * zoom_factor)
                    if (x - mouse_x)**2 + (y - mouse_y)**2 < 100:  # Détecter un clic proche de la galaxie
                        galaxy_colors[i] = YELLOW  # Marquer la galaxie comme entrant en collision
                        break
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and selected_galaxy is not None:  # Déplacer la galaxie sélectionnée
                mouse_x, mouse_y = event.pos
                galaxies[selected_galaxy] = [
                    (mouse_x - width / 2) / (width / 2 * scale_factor(t, H0, Omega_m, Omega_lambda) * zoom_factor),
                    (mouse_y - height / 2) / (height / 2 * scale_factor(t, H0, Omega_m, Omega_lambda) * zoom_factor)
                ]
                selected_galaxy = None
        elif event.type >= pygame.USEREVENT and event.type < pygame.USEREVENT + num_galaxies:
            end_supernova(event.type - pygame.USEREVENT)

    if not paused:
        # Mise à jour du temps
        t += dt

        # Calcul du facteur d'échelle
        a = scale_factor(t, H0, Omega_m, Omega_lambda)

        # Effacement de l'écran
        screen.fill(BLACK)

        # Mise à jour des positions des galaxies
        new_positions = np.copy(galaxies)
        for i in range(num_galaxies):
            net_force = np.array([0.0, 0.0])
            for j in range(num_galaxies):
                if i != j:
                    net_force += gravitational_force(galaxies[i], galaxies[j])
            acceleration = net_force  # Accélération due à la force gravitationnelle
            new_positions[i] += acceleration * dt**2  # Mise à jour de la position
        galaxies = new_positions

        # Affichage des galaxies
        for i, galaxy in enumerate(galaxies):
            x = int(width / 2 + galaxy[0] * width / 2 * a * zoom_factor)
            y = int(height / 2 + galaxy[1] * height / 2 * a * zoom_factor)
            pygame.draw.circle(screen, galaxy_colors[i], (x, y), 2)

        # Mise à jour de l'affichage
        pygame.display.flip()

    pygame.time.delay(50)  # Délai pour ralentir l'animation

pygame.quit()
sys.exit()
