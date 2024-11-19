import numpy as np
from sklearn.datasets import make_classification, make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class HierarchicalAttractorML:
    def __init__(self, n_levels=3, n_attractors_per_level=5, dim=None, learning_rate=0.1):
        self.n_levels = n_levels
        self.n_attractors_per_level = n_attractors_per_level
        self.dim = dim
        self.learning_rate = learning_rate
        self.hierarchy = {}  # {level: {class_label: [(position, strength, attention), ...]}}
        self.pca_models = {}
        
    def _init_hierarchy(self, X):
        if self.dim is None:
            self.dim = X.shape[1]
            
        dims_per_level = [max(2, self.dim // (2**i)) for i in range(self.n_levels)]
        
        for level in range(self.n_levels):
            if level not in self.pca_models and dims_per_level[level] < X.shape[1]:
                self.pca_models[level] = PCA(n_components=dims_per_level[level])
                self.pca_models[level].fit(X)
                
    def _transform_data(self, X, level):
        if level in self.pca_models:
            return self.pca_models[level].transform(X)
        return X
        
    def _create_attractor(self, data_point, label, level):
        if level not in self.hierarchy:
            self.hierarchy[level] = {}
            
        if label not in self.hierarchy[level]:
            self.hierarchy[level][label] = []
            
        # Ajoute l'attracteur avec attention dynamique
        self.hierarchy[level][label].append({
            'position': data_point.copy(),
            'strength': 1.0,
            'attention': 1.0,
            'count': 1,
            'velocity': np.zeros_like(data_point),
            'acceleration': np.zeros_like(data_point)
        })
        
        self._merge_close_attractors(label, level)
        self._update_attention(label, level)
        
    def _merge_close_attractors(self, label, level, threshold=0.5):
        if len(self.hierarchy[level][label]) < 2:
            return
            
        i = 0
        while i < len(self.hierarchy[level][label]):
            j = i + 1
            while j < len(self.hierarchy[level][label]):
                dist = np.linalg.norm(
                    self.hierarchy[level][label][i]['position'] - 
                    self.hierarchy[level][label][j]['position']
                )
                
                if dist < threshold:
                    # Fusion avec conservation de momentum
                    count_i = self.hierarchy[level][label][i]['count']
                    count_j = self.hierarchy[level][label][j]['count']
                    total_count = count_i + count_j
                    
                    # Moyenne pondérée
                    for key in ['position', 'velocity', 'acceleration']:
                        self.hierarchy[level][label][i][key] = (
                            self.hierarchy[level][label][i][key] * count_i +
                            self.hierarchy[level][label][j][key] * count_j
                        ) / total_count
                    
                    # Mise à jour des propriétés
                    self.hierarchy[level][label][i]['count'] = total_count
                    self.hierarchy[level][label][i]['strength'] = max(
                        self.hierarchy[level][label][i]['strength'],
                        self.hierarchy[level][label][j]['strength']
                    )
                    self.hierarchy[level][label][i]['attention'] = max(
                        self.hierarchy[level][label][i]['attention'],
                        self.hierarchy[level][label][j]['attention']
                    )
                    
                    self.hierarchy[level][label].pop(j)
                else:
                    j += 1
            i += 1
            
    def _update_attention(self, label, level):
        if len(self.hierarchy[level][label]) <= 1:
            return
            
        # Calcul des distances relatives
        positions = np.array([a['position'] for a in self.hierarchy[level][label]])
        distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
        
        # Mise à jour de l'attention basée sur la centralité
        for i, attractor in enumerate(self.hierarchy[level][label]):
            mean_dist = np.mean(distances[i][distances[i] > 0])
            attractor['attention'] = 1.0 / (1.0 + mean_dist)
            
    def _update_dynamics(self, label, level):
        dt = 0.1  # Pas de temps
        damping = 0.1  # Coefficient d'amortissement
        
        for attractor in self.hierarchy[level][label]:
            # Mise à jour position
            attractor['position'] += attractor['velocity'] * dt
            
            # Mise à jour vitesse avec amortissement
            attractor['velocity'] += attractor['acceleration'] * dt
            attractor['velocity'] *= (1 - damping)
            
            # Mise à jour accélération (force aléatoire faible)
            attractor['acceleration'] = np.random.randn(*attractor['position'].shape) * 0.01
            
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._init_hierarchy(X)
        
        for level in range(self.n_levels):
            X_transformed = self._transform_data(X, level)
            
            for x, label in zip(X_transformed, y):
                self._create_attractor(x, label, level)
                self._update_dynamics(label, level)
                
    def _calculate_attraction(self, x, label, level):
        total_force = 0
        attention_sum = 0
        
        for attractor in self.hierarchy[level][label]:
            dist = np.linalg.norm(x - attractor['position'])
            force = (attractor['strength'] * attractor['attention'] * 
                    np.exp(-dist))
            total_force += force
            attention_sum += attractor['attention']
            
        return total_force / (attention_sum + 1e-10)
        
    def predict(self, X):
        predictions = []
        
        for x in X:
            forces = {}
            for label in self.classes_:
                label_force = 0
                for level in range(self.n_levels):
                    x_transformed = self._transform_data(np.array([x]), level)[0]
                    level_force = self._calculate_attraction(x_transformed, label, level)
                    label_force += level_force * (2 ** level)  # Plus de poids aux niveaux supérieurs
                forces[label] = label_force
                
            pred_label = max(forces.keys(), key=lambda k: forces[k])
            predictions.append(pred_label)
            
        return np.array(predictions)
        
    def visualize(self, X, y, title="Visualisation hiérarchique des attracteurs"):
        n_rows = (self.n_levels + 1) // 2
        n_cols = 2
        plt.figure(figsize=(15, 5*n_rows))
        
        for level in range(self.n_levels):
            plt.subplot(n_rows, n_cols, level + 1)
            
            X_transformed = self._transform_data(X, level)
            if X_transformed.shape[1] > 2:
                X_transformed = PCA(n_components=2).fit_transform(X_transformed)
                
            # Affiche les points
            for label in self.classes_:
                mask = y == label
                plt.scatter(X_transformed[mask, 0], X_transformed[mask, 1],
                          alpha=0.5, label=f'Class {label}')
                          
                # Affiche les attracteurs
                if level in self.hierarchy and label in self.hierarchy[level]:
                    for attractor in self.hierarchy[level][label]:
                        pos = attractor['position']
                        if len(pos) > 2:
                            # Ajout de points aléatoires pour satisfaire PCA
                            random_points = np.random.randn(3, len(pos))
                            points_with_noise = np.vstack([pos, random_points])
                            pos = PCA(n_components=2).fit_transform(points_with_noise)[0]
                        
                        size = 200 * attractor['strength'] * attractor['attention']
                        plt.scatter(pos[0], pos[1], marker='*', s=size,
                                  color='red', alpha=0.7)
                        circle = plt.Circle((pos[0], pos[1]), 
                                         0.2 * attractor['attention'],
                                         color='red', fill=False, alpha=0.3)
                        plt.gca().add_artist(circle)
                        
                        # Affiche le vecteur vitesse
                        if len(attractor['velocity']) <= 2:
                            plt.quiver(pos[0], pos[1],
                                     attractor['velocity'][0],
                                     attractor['velocity'][1],
                                     color='blue', alpha=0.5)
                                     
            plt.title(f'Niveau {level}')
            plt.legend()
            plt.grid(True)
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

def test_hierarchical_attractor_ml():
    # Test sur make_moons
    print("Test sur make_moons...")
    X, y = make_moons(n_samples=200, noise=0.15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = HierarchicalAttractorML(n_levels=3, n_attractors_per_level=5, dim=2)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Précision sur make_moons : {accuracy:.2f}")
    
    model.visualize(X_train, y_train, "Make_moons avec attracteurs hiérarchiques")
    
    # Test sur MNIST
    print("\nTest sur MNIST...")
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target
    
    # Normalisation
    scaler = StandardScaler()
    X_digits_scaled = scaler.fit_transform(X_digits)
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X_digits_scaled, y_digits, test_size=0.2)
    
    # Création et entraînement du modèle
    model_digits = HierarchicalAttractorML(n_levels=4, n_attractors_per_level=10, dim=64)
    model_digits.fit(X_train, y_train)
    
    # Évaluation
    predictions = model_digits.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Précision sur MNIST : {accuracy:.2f}")
    
    # Visualisation
    model_digits.visualize(X_train, y_train, "MNIST avec attracteurs hiérarchiques")

if __name__ == "__main__":
    test_hierarchical_attractor_ml()