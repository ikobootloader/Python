from dataclasses import dataclass

@dataclass
class MLConfig:
    """Configuration pour les modèles de Machine Learning."""
    sequence_length: int = 15          # Augmenté pour plus de contexte
    hidden_size: int = 256            # Doublé
    num_layers: int = 3               # Augmenté
    dropout: float = 0.3              # Augmenté légèrement
    learning_rate: float = 0.0005     # Réduit pour plus de stabilité
    batch_size: int = 16              # Réduit pour mieux gérer la complexité
    num_epochs: int = 200             # Doublé
    validation_split: float = 0.15    # Ajusté
    temperature: float = 0.5          # Augmenté pour plus d'exploration

@dataclass
class ValidationConfig:
    """Configuration optimisée pour la validation."""
    n_splits: int = 10                 # Augmenté pour plus de robustesse
    test_size: int = 30               # Agrandi pour mieux évaluer
    validation_metric: str = 'matches'
    confidence_threshold: float = 0.6  # Réduit pour plus de sensibilité
    min_training_size: int = 300      # Augmenté pour plus de données d'apprentissage

    def __post_init__(self):
        if self.min_training_size < 100:
            raise ValueError("La taille minimale d'entraînement doit être d'au moins 100 tirages")
        if self.min_training_size > 500:
            raise ValueError("La taille maximale d'entraînement ne doit pas dépasser 500 tirages")

@dataclass
class ProcessingConfig:
    """Configuration pour le post-traitement des prédictions."""
    temperature: float = 0.3
    min_number: int = 1
    max_number: int = 49
    min_bonus: int = 1
    max_bonus: int = 10
    required_numbers: int = 5