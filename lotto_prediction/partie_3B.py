import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

@dataclass
class MLConfig:
    """Configuration pour les modèles de Machine Learning."""
    sequence_length: int = 10
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    validation_split: float = 0.2

class LottoDataset(Dataset):
    """Dataset personnalisé pour les données de loterie."""
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class FeatureExtractor:
    """Extracteur de caractéristiques pour les données de loterie."""
    def __init__(self, config: MLConfig):
        self.config = config
        self.scaler = StandardScaler()
        self._initialize_feature_size()

    def _initialize_feature_size(self):
        """Initialise et calcule la taille des features."""
        # Pour chaque sequence:
        # - 5 stats de base par tirage (mean, std, min, max, median)
        # - 2 ratios (low_numbers, even_numbers)
        # - 3 gaps features (mean, std, max)
        basic_features_per_sequence = (5 + 2 + 3) * self.config.sequence_length
        
        # Features temporelles:
        # - 4 caractéristiques cycliques (2 pour jour, 2 pour mois)
        # - 3 différences temporelles (mean, std, max)
        temporal_features = 4 + 3
        
        # Features statistiques:
        # - 10 bins d'histogramme normalisé
        # - 5 valeurs d'autocorrélation
        # - 1 valeur d'entropie
        statistical_features = 10 + 5 + 1
        
        self.feature_size = basic_features_per_sequence + temporal_features + statistical_features

    def get_feature_size(self) -> int:
        """Retourne la taille du vecteur de caractéristiques."""
        return self.feature_size

    def extract_sequence_features(self, df) -> np.ndarray:
        """Extrait des caractéristiques des séquences de tirages."""
        features = []
        
        for i in range(len(df) - self.config.sequence_length + 1):
            sequence = df.iloc[i:i+self.config.sequence_length]
            
            # Combine toutes les caractéristiques
            combined_features = np.concatenate([
                self._extract_basic_features(sequence),
                self._extract_temporal_features(sequence),
                self._extract_statistical_features(sequence)
            ])
            
            features.append(combined_features)
        
        features = np.array(features)
        
        # Normalisation
        if len(features) > 0:
            features = self.scaler.fit_transform(features)
        
        return features
    
    def _extract_basic_features(self, sequence) -> np.ndarray:
        """Extrait les caractéristiques de base d'une séquence."""
        features = []
        
        for _, row in sequence.iterrows():
            numbers = [row[f'boule_{i}'] for i in range(1, 6)]
            
            # Statistiques de base
            features.extend([
                np.mean(numbers),
                np.std(numbers),
                np.min(numbers),
                np.max(numbers),
                np.median(numbers)
            ])
            
            # Ratios et proportions
            low_numbers = sum(1 for n in numbers if n <= 25)
            even_numbers = sum(1 for n in numbers if n % 2 == 0)
            features.extend([
                low_numbers / 5,
                even_numbers / 5
            ])
            
            # Écarts entre les nombres
            sorted_numbers = sorted(numbers)
            gaps = [sorted_numbers[i+1] - sorted_numbers[i] 
                   for i in range(len(sorted_numbers)-1)]
            features.extend([
                np.mean(gaps),
                np.std(gaps),
                max(gaps)
            ])
            
        return np.array(features)
    
    def _extract_temporal_features(self, sequence) -> np.ndarray:
        """Extrait les caractéristiques temporelles d'une séquence."""
        features = []
        
        # Conversion des dates en caractéristiques cycliques
        dates = pd.to_datetime(sequence['date_de_tirage'])
        
        # Caractéristiques cycliques pour le jour de la semaine (0-6)
        day_sin = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
        day_cos = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
        
        # Caractéristiques cycliques pour le mois (1-12)
        month_sin = np.sin(2 * np.pi * dates.dt.month / 12)
        month_cos = np.cos(2 * np.pi * dates.dt.month / 12)
        
        # Calcul des moyennes pour avoir une seule valeur par caractéristique
        features.extend([
            np.mean(day_sin),
            np.mean(day_cos),
            np.mean(month_sin),
            np.mean(month_cos)
        ])
        
        # Différences temporelles entre les tirages
        time_diffs = dates.diff().dt.total_seconds() / (24 * 3600)  # en jours
        
        # Ajout des statistiques temporelles (en retirant les valeurs NaN)
        features.extend([
            np.nanmean(time_diffs),
            np.nanstd(time_diffs) if len(time_diffs) > 1 else 0,
            np.nanmax(time_diffs) if len(time_diffs) > 0 else 0
        ])
        
        return np.array(features)
    
    def _extract_statistical_features(self, sequence) -> np.ndarray:
        """Extrait les caractéristiques statistiques d'une séquence."""
        features = []
        all_numbers = []
        
        # Collecte de tous les numéros de la séquence
        for _, row in sequence.iterrows():
            numbers = [row[f'boule_{i}'] for i in range(1, 6)]
            all_numbers.extend(numbers)
        
        # Distribution des nombres (histogramme normalisé)
        hist, _ = np.histogram(all_numbers, bins=10, range=(1, 50))
        features.extend(hist / len(all_numbers))
        
        # Autocorrélation (5 premières valeurs)
        autocorr = np.correlate(all_numbers, all_numbers, mode='full')
        center = len(autocorr) // 2
        features.extend(autocorr[center:center+5] / autocorr[center])
        
        # Entropie de la distribution
        hist = np.histogram(all_numbers, bins=10)[0]
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        features.append(entropy)
        
        return np.array(features)
        
class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, config):
        super().__init__()
        self.config = config
        self.temperature = 0.5
        self.input_size = input_size
        
        # Dimensions de base
        self.hidden_size = config.hidden_size
        
        # Normalisation de l'entrée
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Projection de l'entrée
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # LSTM principal
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Dimension de sortie du LSTM (x2 car bidirectionnel)
        lstm_output_size = self.hidden_size * 2
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Couche de réduction de dimension post-attention
        self.post_attention = nn.Sequential(
            nn.Linear(lstm_output_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Réseau pour les numéros principaux
        self.main_numbers_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, 5 * 49)
        )
        
        # Réseau pour le numéro chance
        self.bonus_number_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.BatchNorm1d(self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, 10)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialisation des poids."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'linear' in name:
                    nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Assurer la dimension de séquence
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        
        # Normalisation de l'entrée
        x_flat = x.reshape(-1, self.input_size)
        x_norm = self.input_norm(x_flat)
        x = x_norm.reshape(batch_size, -1, self.input_size)
        
        # Projection initiale
        x = self.input_projection(x.view(-1, self.input_size))
        x = x.view(batch_size, -1, self.hidden_size)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Réduction de dimension
        features = self.post_attention(attn_out[:, -1])
        
        # Prédiction des numéros principaux
        main_logits = self.main_numbers_head(features)
        main_logits = main_logits.reshape(batch_size, 5, 49) / self.temperature
        
        # Prédiction du numéro chance
        bonus_logits = self.bonus_number_head(features) / self.temperature
        
        return main_logits, bonus_logits

    def adjust_temperature(self, new_temp: float):
        """Ajuste la température du softmax."""
        self.temperature = max(0.1, min(1.0, new_temp))
        
    def predict_proba(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcule les probabilités de sortie."""
        with torch.no_grad():
            main_logits, bonus_logits = self.forward(x)
            main_probs = F.softmax(main_logits, dim=-1)
            bonus_probs = F.softmax(bonus_logits, dim=-1)
        return main_probs, bonus_probs