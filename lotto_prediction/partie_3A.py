import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta

from partie_1 import LottoDataAnalyzer  # Import ajouté
from partie_2 import LottoOptimizer, OptimizationConfig    # Import modifié

@dataclass
class ValidationConfig:
    """Configuration pour la validation des prédictions."""
    n_splits: int = 5
    test_size: int = 20
    validation_metric: str = 'matches'  # 'matches' ou 'partial_matches'
    confidence_threshold: float = 0.7
    min_training_size: int = 100

class LottoValidator:
    def __init__(self, analyzer, optimizer, config=None):
        """Initialise le validateur avec les analyseurs et optimiseurs."""
        self.analyzer = analyzer
        self.optimizer = optimizer
        self.config = config or ValidationConfig()
        self.validation_history = []

    def _create_temp_analyzer(self, train_df):
        """Crée un analyseur temporaire à partir d'un DataFrame."""
        if train_df.empty:
            raise ValueError("DataFrame d'entraînement vide")
            
        temp_analyzer = LottoDataAnalyzer.__new__(LottoDataAnalyzer)
        temp_analyzer.df = train_df.copy()  # Création d'une copie explicite
        temp_analyzer.number_range = range(1, 50)
        temp_analyzer.bonus_range = range(1, 11)
        return temp_analyzer

    def _create_temp_optimizer(self, temp_analyzer):
        """Crée un optimiseur temporaire avec un analyseur temporaire."""
        try:
            # Créer l'optimiseur temporaire
            temp_optimizer = LottoOptimizer.__new__(LottoOptimizer)
            temp_optimizer.analyzer = temp_analyzer
            temp_optimizer.config = OptimizationConfig()
            
            # Calcul des attributs nécessaires
            temp_optimizer.frequencies = temp_analyzer.compute_frequency_analysis()
            temp_optimizer.delays = temp_analyzer.compute_delay_analysis()
            temp_optimizer.correlation_matrix = temp_analyzer.get_correlation_matrix()
            
            # Initialisation des poids
            temp_optimizer.pattern_weights = {
                'frequency': 0.25,
                'delay': 0.20,
                'correlation': 0.15,
                'distribution': 0.20,
                'historical': 0.20
            }
            
            return temp_optimizer
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la création de l'optimiseur temporaire: {str(e)}")

    def _calculate_prediction_score(self, predicted, actual):
        """Calcule le score d'une prédiction par rapport au tirage réel avec gestion d'erreurs."""
        try:
            pred_numbers, pred_bonus = predicted
            
            # Vérification des dimensions
            if len(pred_numbers) != 5:
                raise ValueError(f"Nombre incorrect de numéros prédits: {len(pred_numbers)}")
                
            actual_numbers = []
            for i in range(1, 6):
                col = f'boule_{i}'
                if col not in actual:
                    raise KeyError(f"Colonne {col} manquante dans les données actuelles")
                actual_numbers.append(actual[col])
                
            if 'numero_chance' not in actual:
                raise KeyError("Colonne numero_chance manquante dans les données actuelles")
            actual_bonus = actual['numero_chance']
            
            # Calcul des scores avec vérifications
            correct_numbers = len(set(pred_numbers) & set(actual_numbers))
            bonus_correct = int(pred_bonus == actual_bonus)
            
            scores = {
                'matching_numbers': correct_numbers,
                'bonus_match': bonus_correct,
                'full_match': int(correct_numbers == 5 and bonus_correct),
                'partial_score': (correct_numbers / 5 + (bonus_correct * 0.2)),
                'position_independent_score': correct_numbers / 5,
                'bonus_score': float(bonus_correct)
            }
            
            return scores
        except Exception as e:
            print(f"Erreur lors du calcul du score: {str(e)}")
            return None

    def perform_backtesting(self, start_date=None, end_date=None):
        """Effectue un backtesting des prédictions sur une période donnée avec gestion d'erreurs améliorée."""
        print("Démarrage du backtesting...")
        
        try:
            # Préparation des données avec vérification
            df = self.analyzer.df.copy()
            if df.empty:
                raise ValueError("Dataset vide")
                
            if start_date:
                df = df[df['date_de_tirage'] >= start_date]
            if end_date:
                df = df[df['date_de_tirage'] <= end_date]
                
            if len(df) < self.config.min_training_size:
                raise ValueError(f"Dataset trop petit: {len(df)} tirages < {self.config.min_training_size} minimum requis")
                
            print(f"Nombre total de tirages pour le backtesting: {len(df)}")
            
            results = []
            window_size = self.config.min_training_size
            total_windows = len(df) - window_size
            
            print(f"Taille de la fenêtre d'entraînement: {window_size}")
            print(f"Nombre total de fenêtres: {total_windows}")
            
            for i in range(total_windows):
                try:
                    if i % 10 == 0:
                        print(f"Processing window {i}/{total_windows}")
                        
                    # Création des ensembles d'entraînement et de test
                    train_df = df.iloc[i:i+window_size].copy()
                    test_row = df.iloc[i+window_size].copy()
                    
                    # Création des analyseurs et optimiseurs temporaires
                    temp_analyzer = self._create_temp_analyzer(train_df)
                    temp_optimizer = self._create_temp_optimizer(temp_analyzer)
                    
                    # Génération de la prédiction
                    predicted = temp_optimizer.optimize_numbers()
                    scores = self._calculate_prediction_score(predicted, test_row)
                    
                    if scores is not None:
                        results.append(scores)
                        
                except Exception as e:
                    print(f"Erreur lors de la prédiction pour la fenêtre {i}: {str(e)}")
                    continue
            
            # Calcul des métriques agrégées
            if not results:
                print("Aucun résultat de backtesting disponible")
                return {}
                
            aggregated_metrics = {
                'average_matching_numbers': np.mean([r['matching_numbers'] for r in results]),
                'bonus_match_rate': np.mean([r['bonus_match'] for r in results]),
                'full_match_rate': np.mean([r['full_match'] for r in results]),
                'average_partial_score': np.mean([r['partial_score'] for r in results]),
                'position_independent_accuracy': np.mean([r['position_independent_score'] for r in results])
            }
            
            # Ajout de métriques avancées
            aggregated_metrics.update({
                'consistency_score': self._calculate_consistency_score(results),
                'prediction_stability': self._calculate_stability_score(results),
                'effectiveness_ratio': self._calculate_effectiveness_ratio(results)
            })
            
            print("\nRésultats du backtesting:")
            for metric, value in aggregated_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            return aggregated_metrics
            
        except Exception as e:
            print(f"Erreur lors du backtesting: {str(e)}")
            return {}
            
    def _calculate_consistency_score(self, results):
        """Calcule un score de cohérence des prédictions avec gestion d'erreurs."""
        try:
            scores = [r['partial_score'] for r in results]
            if not scores:
                return 0.0
            return 1 - np.std(scores) / (np.mean(scores) + 1e-10)
        except Exception:
            return 0.0

    def _calculate_stability_score(self, results):
        """Évalue la stabilité des prédictions dans le temps avec gestion d'erreurs."""
        try:
            scores = [r['partial_score'] for r in results]
            if len(scores) < 2:
                return 0.0
            diffs = np.diff(scores)
            return 1 / (1 + np.std(diffs))
        except Exception:
            return 0.0

    def _calculate_effectiveness_ratio(self, results):
        """Calcule le ratio d'efficacité des prédictions avec gestion d'erreurs."""
        try:
            total_matches = sum(r['matching_numbers'] for r in results)
            total_possible = len(results) * 5
            return total_matches / total_possible if total_possible > 0 else 0.0
        except Exception:
            return 0.0

    def analyze_prediction_patterns(self, n_predictions: int = 100) -> Dict:
        """
        Analyse les patterns dans les prédictions pour détecter les biais potentiels.
        
        Args:
            n_predictions: Nombre de prédictions à analyser
            
        Returns:
            Dict contenant l'analyse des patterns
        """
        predictions = []
        for _ in range(n_predictions):
            numbers, bonus = self.optimizer.optimize_numbers()
            predictions.append((numbers, bonus))
        
        analysis = {
            'number_frequency': np.zeros(49),
            'bonus_frequency': np.zeros(10),
            'consecutive_pairs': 0,
            'low_high_ratio': [],
            'odd_even_ratio': [],
            'sum_distribution': []
        }
        
        for numbers, bonus in predictions:
            # Fréquence des numéros
            for num in numbers:
                analysis['number_frequency'][num-1] += 1
            analysis['bonus_frequency'][bonus-1] += 1
            
            # Paires consécutives
            sorted_numbers = sorted(numbers)
            analysis['consecutive_pairs'] += sum(1 for i in range(len(sorted_numbers)-1)
                                              if sorted_numbers[i+1] - sorted_numbers[i] == 1)
            
            # Ratio bas/haut
            low_count = sum(1 for n in numbers if n <= 25)
            analysis['low_high_ratio'].append(low_count / len(numbers))
            
            # Ratio pair/impair
            even_count = sum(1 for n in numbers if n % 2 == 0)
            analysis['odd_even_ratio'].append(even_count / len(numbers))
            
            # Distribution des sommes
            analysis['sum_distribution'].append(sum(numbers))
        
        # Normalisation et statistiques
        analysis['number_frequency'] /= n_predictions
        analysis['bonus_frequency'] /= n_predictions
        analysis['consecutive_pairs'] /= n_predictions
        
        # Ajout des statistiques descriptives
        for key in ['low_high_ratio', 'odd_even_ratio', 'sum_distribution']:
            analysis[f'{key}_mean'] = np.mean(analysis[key])
            analysis[f'{key}_std'] = np.std(analysis[key])
        
        return analysis