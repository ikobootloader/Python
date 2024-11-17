import random
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    """Configuration pour le post-traitement des prédictions."""
    temperature: float = 0.3
    min_number: int = 1
    max_number: int = 49
    min_bonus: int = 1
    max_bonus: int = 10
    required_numbers: int = 5

class PredictionPostProcessor:
    def __init__(self, config: ProcessingConfig = None):
        """Initialise le post-processeur avec une configuration."""
        self.config = config or ProcessingConfig()
    
    def ensure_unique_numbers(self, predicted_numbers: List[int]) -> List[int]:
        """Assure que les numéros prédits sont uniques et valides."""
        # Filtrage des numéros valides
        valid_numbers = [n for n in predicted_numbers 
                        if self.config.min_number <= n <= self.config.max_number]
        
        # Création d'un ensemble unique
        unique_numbers = list(set(valid_numbers))
        
        # Complétion si nécessaire
        while len(unique_numbers) < self.config.required_numbers:
            new_num = random.randint(self.config.min_number, self.config.max_number)
            if new_num not in unique_numbers:
                unique_numbers.append(new_num)
        
        return sorted(unique_numbers[:self.config.required_numbers])
    
    def evaluate_prediction_quality(self, numbers: List[int], bonus: int) -> Dict[str, float]:
        """Évalue la qualité d'une prédiction."""
        # Validité des numéros
        validity_score = 1.0 if (
            len(set(numbers)) == self.config.required_numbers and
            all(self.config.min_number <= n <= self.config.max_number for n in numbers) and
            self.config.min_bonus <= bonus <= self.config.max_bonus
        ) else 0.0
        
        # Équilibre bas/haut
        mid_point = (self.config.max_number + self.config.min_number) / 2
        low_high_ratio = len([n for n in numbers if n <= mid_point]) / len(numbers)
        
        # Score de diversité (écart entre les numéros)
        sorted_nums = sorted(numbers)
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
        diversity_score = np.mean(gaps) / self.config.max_number
        
        # Score de distribution
        distribution_score = 1 - abs(0.5 - low_high_ratio)
        
        # Score global
        global_score = (validity_score * 0.4 + 
                       distribution_score * 0.3 + 
                       diversity_score * 0.3)
        
        return {
            'global_score': global_score,
            'validity_score': validity_score,
            'low_high_ratio': low_high_ratio,
            'diversity_score': diversity_score,
            'distribution_score': distribution_score
        }
    
    def print_prediction_quality(self, metrics: Dict[str, float]):
        """Affiche une évaluation détaillée de la qualité de la prédiction."""
        print("\n=== Qualité de la Prédiction ===")
        print(f"Score global: {metrics['global_score']:.2%}")
        print(f"Validité: {'✓' if metrics['validity_score'] == 1.0 else '✗'}")
        print(f"Distribution: {metrics['distribution_score']:.2%}")
        print(f"Diversité: {metrics['diversity_score']:.2%}")
        print(f"Équilibre bas/haut: {abs(0.5 - metrics['low_high_ratio'])*100:.1f}% de déviation")
    
    def process_prediction(self, raw_numbers: List[int], bonus: int) -> Tuple[List[int], int, Dict[str, float]]:
        """Traitement complet d'une prédiction."""
        # Assure l'unicité des numéros
        processed_numbers = self.ensure_unique_numbers(raw_numbers)
        
        # Valide et ajuste le numéro bonus si nécessaire
        validated_bonus = min(max(bonus, self.config.min_bonus), self.config.max_bonus)
        
        # Évalue la qualité
        quality_metrics = self.evaluate_prediction_quality(processed_numbers, validated_bonus)
        
        # Affiche les résultats
        self.print_prediction_quality(quality_metrics)
        
        return processed_numbers, validated_bonus, quality_metrics