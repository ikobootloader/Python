from typing import Literal
import torch.nn as nn
from partie_3B import LSTMPredictor
from config.settings import MLConfig

class ModelFactory:
    """Fabrique de modèles de prédiction."""
    
    def create_model(self, 
                    model_type: Literal['lstm', 'transformer', 'ensemble'],
                    input_size: int,
                    config: MLConfig) -> nn.Module:
        """Crée et retourne un modèle selon le type spécifié."""
        if model_type == 'lstm':
            return LSTMPredictor(input_size, config)
        elif model_type == 'transformer':
            raise NotImplementedError("Transformer pas encore implémenté")
        elif model_type == 'ensemble':
            raise NotImplementedError("Ensemble pas encore implémenté")
        else:
            raise ValueError(f"Type de modèle inconnu: {model_type}")