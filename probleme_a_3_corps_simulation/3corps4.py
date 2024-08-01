import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Constantes
G_CONST = 6.67430e-11  # Constante gravitationnelle en m^3 kg^-1 s^-2
TIME_STEP = 1e2  # Pas de temps en secondes (réduit pour plus de précision)

# Définir les propriétés des corps
bodies = {
    "Body1": {"position": np.array([1.0e11, 0.0, 0.0]), "velocity": np.array([0.0, 3.0e4, 1.0e4]), "mass": 1.0e30},
    "Body2": {"position": np.array([-1.0e11, 0.0, 0.0]), "velocity": np.array([0.0, -3.0e4, -0.5e4]), "mass": 1.0e30},
    "Body3": {"position": np.array([0.0, 1.5e11, 1.0e11]), "velocity": np.array([2.5e4, 0.0, -1.0e4]), "mass": 1.0e30},
}

def compute_forces(bodies):
    forces = {body: np.zeros(3) for body in bodies}
    potential_energy = 0
    for i, (body1, prop1) in enumerate(bodies.items()):
        for body2, prop2 in list(bodies.items())[i+1:]:
            r_vec = prop2['position'] - prop1['position']
            r_mag = np.linalg.norm(r_vec)
            if r_mag > 0:
                force_mag = G_CONST * prop1['mass'] * prop2['mass'] / r_mag**2
                force_vec = force_mag * r_vec / r_mag
                forces[body1] += force_vec
                forces[body2] -= force_vec
                potential_energy -= G_CONST * prop1['mass'] * prop2['mass'] / r_mag
    return forces, potential_energy

def update_positions_velocities(bodies, forces, dt):
    for body, properties in bodies.items():
        acceleration = forces[body] / properties['mass']
        properties['velocity'] += acceleration * dt
        properties['position'] += properties['velocity'] * dt + 0.5 * acceleration * dt**2

def calculate_total_energy(bodies, potential_energy):
    kinetic_energy = sum(0.5 * prop['mass'] * np.dot(prop['velocity'], prop['velocity']) for prop in bodies.values())
    return kinetic_energy + potential_energy

def calculate_angular_momentum(bodies):
    return sum(prop['mass'] * np.cross(prop['position'], prop['velocity']) for prop in bodies.values())

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
lines = {body: ax.plot([], [], [], 'o-', label=body, markersize=8)[0] for body in bodies}
ax.legend()

history = {body: {'x': [], 'y': [], 'z': []} for body in bodies}
energy_history = []
angular_momentum_history = []

def init():
    for line in lines.values():
        line.set_data([], [])
        line.set_3d_properties([])
    return lines.values()

def animate(frame):
    forces, potential_energy = compute_forces(bodies)
    update_positions_velocities(bodies, forces, TIME_STEP)

    total_energy = calculate_total_energy(bodies, potential_energy)
    angular_momentum = calculate_angular_momentum(bodies)
    
    energy_history.append(total_energy)
    angular_momentum_history.append(np.linalg.norm(angular_momentum))

    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    for body, line in lines.items():
        pos = bodies[body]['position']
        history[body]['x'].append(pos[0])
        history[body]['y'].append(pos[1])
        history[body]['z'].append(pos[2])
        
        line.set_data(history[body]['x'], history[body]['y'])
        line.set_3d_properties(history[body]['z'])

        x_min, x_max = min(x_min, pos[0]), max(x_max, pos[0])
        y_min, y_max = min(y_min, pos[1]), max(y_max, pos[1])
        z_min, z_max = min(z_min, pos[2]), max(z_max, pos[2])
    
    ax.set_xlim(x_min * 1.1, x_max * 1.1)
    ax.set_ylim(y_min * 1.1, y_max * 1.1)
    ax.set_zlim(z_min * 1.1, z_max * 1.1)
    
    ax.view_init(elev=20, azim=frame/2)

    if frame % 10 == 0:
        print(f"Frame {frame}: Energy = {total_energy:.2e}, Angular Momentum = {np.linalg.norm(angular_momentum):.2e}")

    return lines.values()

ani = FuncAnimation(fig, animate, frames=500, init_func=init, interval=50, blit=False)
plt.show()

# Afficher les graphiques de l'énergie totale et du moment cinétique
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(energy_history)
plt.title("Énergie totale au fil du temps")
plt.ylabel("Énergie (J)")
plt.subplot(212)
plt.plot(angular_momentum_history)
plt.title("Norme du moment cinétique au fil du temps")
plt.ylabel("Moment cinétique (kg⋅m²/s)")
plt.xlabel("Pas de temps")
plt.tight_layout()
plt.show()

print("Animation terminée. Vérifiez les graphiques d'énergie et de moment cinétique.")