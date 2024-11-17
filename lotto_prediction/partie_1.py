import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime

class LottoDataAnalyzer:
    def __init__(self, filepath: str = 'dataset.csv'):
        """Initialise l'analyseur de données Lotto."""
        self.df = self._load_dataset(filepath)
        self.number_range = range(1, 50)
        self.bonus_range = range(1, 11)
        
    def _load_dataset(self, filepath: str) -> pd.DataFrame:
        """Charge et prépare le dataset."""
        try:
            df = pd.read_csv(filepath, parse_dates=['date_de_tirage'], dayfirst=True)
            df = df.sort_values('date_de_tirage', ascending=False)
            return df
        except Exception as e:
            raise Exception(f"Erreur lors du chargement du dataset: {e}")

    def _runs_test(self, sequence: str) -> Tuple[float, float]:
        """
        Implémente le test des runs (séquences).
        
        Args:
            sequence: Chaîne binaire de '0' et '1'
        
        Returns:
            Tuple contenant la statistique du test et la p-value
        """
        n1 = sequence.count('1')
        n0 = sequence.count('0')
        
        # Compter le nombre de runs
        runs = 1
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                runs += 1
        
        # Calcul de la moyenne et de la variance attendues
        n = n0 + n1
        mean_runs = 1 + (2 * n0 * n1) / n
        var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
        
        if var_runs == 0:
            return 0.0, 1.0
        
        # Calcul de la statistique Z
        z = (runs - mean_runs) / np.sqrt(var_runs)
        
        # Calcul de la p-value (test bilatéral)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value

    def perform_statistical_tests(self) -> Dict[str, float]:
        """Effectue des tests statistiques sur les tirages."""
        stats_results = {
            'chi_square_pvalue': 0.0,
            'ks_test_pvalue': 0.0,
            'runs_test_pvalue': 0.0
        }
        
        # Test du chi-carré pour l'uniformité
        observed_freq = np.zeros(49)
        for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']:
            for num in self.df[col]:
                observed_freq[num-1] += 1
        expected_freq = np.ones(49) * (len(self.df) * 5 / 49)
        chi2, pvalue = stats.chisquare(observed_freq, expected_freq)
        stats_results['chi_square_pvalue'] = pvalue
        
        # Test de Kolmogorov-Smirnov
        all_numbers = self.df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].values.ravel()
        ks_stat, ks_pvalue = stats.kstest(all_numbers, 'uniform', args=(1, 49))
        stats_results['ks_test_pvalue'] = ks_pvalue
        
        # Test des runs pour la randomisation
        median = np.median(all_numbers)
        runs = ''.join('1' if x > median else '0' for x in all_numbers)
        _, runs_pvalue = self._runs_test(runs)
        stats_results['runs_test_pvalue'] = runs_pvalue
        
        return stats_results

    def analyze_patterns(self, numbers: List[int]) -> Dict[str, float]:
        """Analyse les patterns dans les numéros prédits."""
        sorted_numbers = sorted(numbers)
        
        # Calcul des différentes métriques
        analysis = {
            'low_high_balance': sum(1 for n in numbers if n <= 25) / len(numbers),
            'even_odd_balance': sum(1 for n in numbers if n % 2 == 0) / len(numbers),
            'sum_range': sum(numbers),
            'max_gap': max(b-a for a, b in zip(sorted_numbers, sorted_numbers[1:])),
            'spread_score': (max(numbers) - min(numbers)) / 48.0,  # Normalisé par la plage possible
        }
        
        # Analyse des séquences historiques similaires
        historical_matches = []
        for _, row in self.df.iterrows():
            draw_numbers = [row[f'boule_{i}'] for i in range(1, 6)]
            common_numbers = len(set(numbers) & set(draw_numbers))
            if common_numbers >= 3:  # Si au moins 3 numéros en commun
                historical_matches.append(common_numbers)
        
        analysis['historical_similarity'] = len(historical_matches) / len(self.df) if historical_matches else 0
        analysis['max_historical_match'] = max(historical_matches) if historical_matches else 0
        
        return analysis

    def compute_frequency_analysis(self) -> Dict[str, np.ndarray]:
        """Analyse les fréquences d'apparition des numéros."""
        frequencies = {
            'main_numbers': np.zeros(49),
            'bonus_numbers': np.zeros(10),
            'normalized_frequencies': np.zeros(49)
        }
        
        # Calcul des fréquences pour les numéros principaux
        for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']:
            for num in self.df[col]:
                frequencies['main_numbers'][num-1] += 1
                
        # Calcul des fréquences pour les numéros chance
        for num in self.df['numero_chance']:
            frequencies['bonus_numbers'][num-1] += 1
            
        # Normalisation des fréquences
        total_tirages = len(self.df) * 5  # 5 numéros par tirage
        frequencies['normalized_frequencies'] = frequencies['main_numbers'] / total_tirages
        
        return frequencies

    def identify_cycles(self, window_size: int = 10) -> Dict[str, List[float]]:
        """Identifie les cycles potentiels dans les tirages."""
        cycles = {
            'autocorrelation': [],
            'periodicity_scores': []
        }
        
        for num in self.number_range:
            time_series = self.df.apply(
                lambda row: 1 if num in [row['boule_1'], row['boule_2'], 
                                       row['boule_3'], row['boule_4'], 
                                       row['boule_5']] else 0, 
                axis=1
            ).values
            
            # Calcul de l'autocorrélation
            autocorr = np.correlate(time_series, time_series, mode='full')
            normalized_autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
            cycles['autocorrelation'].append(normalized_autocorr[:window_size])
            
            # Calcul du score de périodicité
            fft = np.fft.fft(time_series)
            power_spectrum = np.abs(fft)**2
            cycles['periodicity_scores'].append(np.max(power_spectrum[1:]) / np.mean(power_spectrum[1:]))
            
        return cycles

    def compute_delay_analysis(self) -> Dict[str, np.ndarray]:
        """Analyse les retards entre les apparitions des numéros."""
        delays = {
            'current_delays': np.zeros(49),
            'average_delays': np.zeros(49),
            'max_delays': np.zeros(49)
        }
        
        for num in self.number_range:
            last_appearance = next(
                (idx for idx, row in self.df.iterrows() 
                 if num in [row['boule_1'], row['boule_2'], row['boule_3'],
                           row['boule_4'], row['boule_5']]),
                None
            )
            delays['current_delays'][num-1] = 0 if last_appearance is None else last_appearance
            
            appearances = []
            last_idx = None
            for idx, row in self.df.iterrows():
                if num in [row['boule_1'], row['boule_2'], row['boule_3'],
                          row['boule_4'], row['boule_5']]:
                    if last_idx is not None:
                        appearances.append(idx - last_idx)
                    last_idx = idx
            
            if appearances:
                delays['average_delays'][num-1] = np.mean(appearances)
                delays['max_delays'][num-1] = np.max(appearances)
            
        return delays

    def get_correlation_matrix(self) -> np.ndarray:
        """Calcule la matrice de corrélation entre les numéros."""
        correlation_matrix = np.zeros((49, 49))
        
        for i in self.number_range:
            for j in self.number_range:
                if i < j:
                    series_i = self.df.apply(
                        lambda row: 1 if i in [row['boule_1'], row['boule_2'],
                                             row['boule_3'], row['boule_4'],
                                             row['boule_5']] else 0,
                        axis=1
                    ).values
                    
                    series_j = self.df.apply(
                        lambda row: 1 if j in [row['boule_1'], row['boule_2'],
                                             row['boule_3'], row['boule_4'],
                                             row['boule_5']] else 0,
                        axis=1
                    ).values
                    
                    corr = np.corrcoef(series_i, series_j)[0, 1]
                    correlation_matrix[i-1, j-1] = corr
                    correlation_matrix[j-1, i-1] = corr
        
        return correlation_matrix