import pandas as pd
import numpy as np
import torch
import torch.nn as nn  # Ajout de cet import
import torch.nn.functional as F
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

from config.settings import MLConfig, ValidationConfig, ProcessingConfig
from src.post_processing.prediction_processor import PredictionPostProcessor
from partie_1 import LottoDataAnalyzer
from partie_2 import LottoOptimizer
from partie_3A import LottoValidator
from partie_3B import FeatureExtractor
from src.models.model_factory import ModelFactory

class LottoPredictionSystem:
    """Système complet de prédiction de loterie."""
    
    def __init__(self, filepath: str = 'dataset.csv', 
                 ml_config: MLConfig = None,
                 validation_config: ValidationConfig = None,
                 processing_config: ProcessingConfig = None):
        """Initialise le système avec tous les composants nécessaires."""
        # Configurations
        self.ml_config = ml_config or MLConfig()
        self.validation_config = validation_config or ValidationConfig()
        self.processing_config = processing_config or ProcessingConfig()
        
        # Composants principaux
        self.analyzer = LottoDataAnalyzer(filepath)
        self.optimizer = LottoOptimizer(self.analyzer)
        self.validator = LottoValidator(self.analyzer, self.optimizer, self.validation_config)
        self.feature_extractor = FeatureExtractor(self.ml_config)
        self.post_processor = PredictionPostProcessor(self.processing_config)
        
        # Modèles
        self.model_factory = ModelFactory()
        self._initialize_models()
        
        # Configuration de la température si applicable
        if hasattr(self.model, 'adjust_temperature'):
            self.model.adjust_temperature(self.ml_config.temperature)

    def _initialize_models(self):
        """Initialise et configure les modèles."""
        input_size = self.feature_extractor.get_feature_size()
        
        # Création du modèle principal
        self.model = self.model_factory.create_model(
            "lstm", 
            input_size, 
            self.ml_config
        )
        
        # Configuration de la température
        if hasattr(self.model, 'adjust_temperature'):
            self.model.adjust_temperature(self.ml_config.temperature)

    def _verify_dimensions(self, features, targets):
        """Vérifie la compatibilité des dimensions des données."""
        print(f"Dimension des features : {features.shape}")
        print(f"Dimension des targets : {targets.shape}")
        print(f"Taille d'entrée du modèle : {self.feature_extractor.get_feature_size()}")
        
        expected_feature_size = self.feature_extractor.get_feature_size()
        actual_feature_size = features.shape[-1] if len(features.shape) > 1 else features.shape[0]
        
        if actual_feature_size != expected_feature_size:
            raise ValueError(
                f"Incompatibilité des dimensions. "
                f"Attendu : {expected_feature_size}, "
                f"Obtenu : {actual_feature_size}"
            )
            
    def train_system(self):
        """Entraîne le système complet."""
        print("Début de l'entraînement du système...")
        
        # Préparation des données
        features = self.feature_extractor.extract_sequence_features(self.analyzer.df)
        targets = self._prepare_targets()
        
        # Vérification des dimensions
        self._verify_dimensions(features, targets)
        
        # Entraînement du modèle
        self._train_model(features, targets)
        
        # Validation
        self._validate_system()

    def predict_next_draw(self) -> Tuple[List[int], Dict]:
        """Prédit et analyse le prochain tirage."""
        print("\nGénération de la prédiction pour le prochain tirage...")
        
        # Extraction des caractéristiques récentes
        recent_data = self.analyzer.df.head(self.ml_config.sequence_length)
        features = self.feature_extractor.extract_sequence_features(recent_data)
        
        # Prédiction brute
        with torch.no_grad():
            main_logits, bonus_logits = self.model(torch.FloatTensor(features).unsqueeze(0))
            raw_numbers = main_logits.argmax(dim=2)[0].numpy() + 1
            raw_bonus = bonus_logits.argmax(dim=1).item() + 1
        
        # Post-traitement
        processed_numbers, processed_bonus, quality_metrics = self.post_processor.process_prediction(
            raw_numbers.tolist(), 
            raw_bonus
        )
        
        # Analyse complète
        analysis = self._analyze_prediction(processed_numbers, processed_bonus)
        analysis['quality_metrics'] = quality_metrics
        
        return (processed_numbers, processed_bonus), analysis

    def _analyze_prediction(self, numbers: List[int], bonus: int) -> Dict:
        """Analyse détaillée d'une prédiction."""
        analysis = {
            'statistical_analysis': self.analyzer.perform_statistical_tests(),
            'pattern_analysis': self.analyzer.analyze_patterns(numbers),
            'optimization_score': self.optimizer._evaluate_combination(numbers, bonus),
            'historical_analysis': {
                'frequency_analysis': self.analyzer.compute_frequency_analysis(),
                'delay_analysis': self.analyzer.compute_delay_analysis(),
                'cycles': self.analyzer.identify_cycles()
            }
        }
        
        return analysis

    def _prepare_targets(self) -> np.ndarray:
        """Prépare les cibles pour l'entraînement."""
        targets = []
        for _, row in self.analyzer.df.iterrows():
            numbers = [row[f'boule_{i}']-1 for i in range(1, 6)]
            bonus = row['numero_chance']-1
            targets.append(numbers + [bonus])
        return np.array(targets)

    def _prepare_data(self, features: np.ndarray, targets: np.ndarray):
        """
        Prépare les données pour l'entraînement en s'assurant que les dimensions correspondent
        et que toutes les valeurs sont valides.
        """
        print("Dimensions originales:")
        print(f"Features: {features.shape}")
        print(f"Targets: {targets.shape}")
        print(f"Premiers targets: {targets[0]}")
        
        # S'assure que nous avons le même nombre d'exemples
        min_size = min(len(features), len(targets))
        features = features[:min_size]
        targets = targets[:min_size].astype(np.int64)  # Conversion explicite en int64
        
        # Vérifie et corrige les valeurs des targets
        print("\nVérification des valeurs targets avant correction:")
        print(f"Min target principal: {targets[:, :5].min()}")
        print(f"Max target principal: {targets[:, :5].max()}")
        print(f"Min target chance: {targets[:, 5].min()}")
        print(f"Max target chance: {targets[:, 5].max()}")
        
        # Corrige les valeurs non valides
        for i in range(len(targets)):
            for j in range(5):  # Pour les numéros principaux
                if targets[i, j] < 1 or targets[i, j] > 49:
                    targets[i, j] = 1  # Valeur par défaut sûre
            # Pour le numéro chance
            if targets[i, 5] < 1 or targets[i, 5] > 10:
                targets[i, 5] = 1  # Valeur par défaut sûre
        
        # Convertit les targets en indices 0-based
        adjusted_targets = np.copy(targets)
        adjusted_targets[:, :5] = targets[:, :5] - 1  # Indices 0-48 pour les numéros principaux
        adjusted_targets[:, 5] = targets[:, 5] - 1    # Indices 0-9 pour le numéro chance
        
        print("\nVérification des valeurs après ajustement:")
        print(f"Min target principal: {adjusted_targets[:, :5].min()}")
        print(f"Max target principal: {adjusted_targets[:, :5].max()}")
        print(f"Min target chance: {adjusted_targets[:, 5].min()}")
        print(f"Max target chance: {adjusted_targets[:, 5].max()}")
        
        # Calcule les indices de séparation
        train_size = int(min_size * (1 - self.ml_config.validation_split))
        
        # Sépare en ensembles d'entraînement et de validation
        train_features = features[:train_size]
        train_targets = adjusted_targets[:train_size]
        val_features = features[train_size:]
        val_targets = adjusted_targets[train_size:]
        
        # Vérification finale des valeurs
        assert train_targets[:, :5].min() >= 0, "Train targets négatifs trouvés"
        assert train_targets[:, :5].max() < 49, "Train targets trop grands trouvés"
        assert train_targets[:, 5].min() >= 0, "Train bonus négatifs trouvés"
        assert train_targets[:, 5].max() < 10, "Train bonus trop grands trouvés"
        
        # Conversion en tenseurs
        train_features = torch.FloatTensor(train_features)
        train_targets = torch.LongTensor(train_targets)
        val_features = torch.FloatTensor(val_features)
        val_targets = torch.LongTensor(val_targets)
        
        print("\nDimensions finales:")
        print(f"Train features: {train_features.shape}")
        print(f"Train targets: {train_targets.shape}")
        print(f"Val features: {val_features.shape}")
        print(f"Val targets: {val_targets.shape}")
        
        return train_features, train_targets, val_features, val_targets

    def _train_model(self, features: np.ndarray, targets: np.ndarray):
        """Entraîne le modèle principal avec vérification des données."""
        try:
            # Préparation des données avec vérification
            train_features, train_targets, val_features, val_targets = self._prepare_data(features, targets)
            
            # Optimiseur et critère
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.ml_config.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Boucle d'entraînement
            for epoch in range(self.ml_config.num_epochs):
                self.model.train()
                optimizer.zero_grad()
                
                # Forward pass
                main_logits, bonus_logits = self.model(train_features)
                
                # Reshape pour le calcul de la perte
                batch_size = train_features.size(0)
                main_logits_reshaped = main_logits.reshape(-1, 49)
                main_targets_reshaped = train_targets[:, :5].reshape(-1)
                
                # Vérification des valeurs avant le calcul de la perte
                assert main_targets_reshaped.min() >= 0, f"Target négatif trouvé: {main_targets_reshaped.min()}"
                assert main_targets_reshaped.max() < 49, f"Target trop grand trouvé: {main_targets_reshaped.max()}"
                
                # Calcul de la perte
                main_loss = criterion(main_logits_reshaped, main_targets_reshaped)
                bonus_loss = criterion(bonus_logits, train_targets[:, 5])
                loss = main_loss + bonus_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Validation périodique
                if (epoch + 1) % 10 == 0:
                    self._validate_epoch(epoch, val_features, val_targets, loss.item())
                    
        except Exception as e:
            print(f"Erreur pendant l'entraînement: {str(e)}")
            print(f"Type d'erreur: {type(e)}")
            raise

    def _validate_epoch(self, epoch: int, val_features: torch.Tensor, val_targets: torch.Tensor, train_loss: float):
        """Valide le modèle sur une époque avec les indices corrects."""
        self.model.eval()
        with torch.no_grad():
            # Forward pass
            main_logits, bonus_logits = self.model(val_features)
            
            # Reshape pour le calcul des métriques
            batch_size = val_features.size(0)
            main_logits_reshaped = main_logits.reshape(-1, 49)
            main_targets_reshaped = val_targets[:, :5].reshape(-1)
            
            # Calcul des prédictions
            main_preds = main_logits_reshaped.argmax(dim=1)
            bonus_preds = bonus_logits.argmax(dim=1)
            
            # Calcul des métriques
            main_correct = (main_preds == main_targets_reshaped).sum().item()
            main_total = main_targets_reshaped.size(0)
            main_accuracy = main_correct / main_total if main_total > 0 else 0
            
            bonus_correct = (bonus_preds == val_targets[:, 5]).sum().item()
            bonus_total = val_targets[:, 5].size(0)
            bonus_accuracy = bonus_correct / bonus_total if bonus_total > 0 else 0
            
            print(f"\nÉpoque {epoch+1}:")
            print(f"  Perte d'entraînement: {train_loss:.4f}")
            print(f"  Précision numéros principaux: {main_accuracy:.2%}")
            print(f"  Précision numéro chance: {bonus_accuracy:.2%}")
            
            # Affiche quelques prédictions pour vérification
            if epoch % 50 == 0:
                for i in range(min(3, batch_size)):
                    pred_numbers = main_logits[i].reshape(5, 49).argmax(dim=1) + 1
                    true_numbers = val_targets[i, :5] + 1
                    pred_bonus = bonus_preds[i] + 1
                    true_bonus = val_targets[i, 5] + 1
                    print(f"\nExemple {i+1}:")
                    print(f"  Prédit: {pred_numbers.tolist()} + {pred_bonus}")
                    print(f"  Réel  : {true_numbers.tolist()} + {true_bonus}")

    def _validate_system(self):
        """Valide le système complet."""
        print("\nValidation du système...")
        results = self.validator.perform_backtesting()
        
        print("\n=== Résultats de la validation ===")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
            

def main():
    """Point d'entrée principal."""
    # Initialisation
    system = LottoPredictionSystem('dataset.csv')
    
    # Entraînement
    system.train_system()
    
    # Prédiction
    prediction, analysis = system.predict_next_draw()
    
    # Affichage des résultats
    print("\n=== Prédiction pour le prochain tirage ===")
    print(f"Numéros: {sorted(prediction[0])}")
    print(f"Numéro chance: {prediction[1]}")
    
    print("\n=== Analyse détaillée ===")
    for key, value in analysis['quality_metrics'].items():
        print(f"{key}: {value:.3f}")

if __name__ == "__main__":
    main()