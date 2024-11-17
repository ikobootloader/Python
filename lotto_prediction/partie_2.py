import numpy as np
from typing import List, Tuple, Dict
from itertools import combinations
from collections import defaultdict
from dataclasses import dataclass
import random

@dataclass
class OptimizationConfig:
    """Configuration pour les paramètres d'optimisation."""
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    tournament_size: int = 5
    particle_count: int = 30
    inertia_weight: float = 0.7
    cognitive_weight: float = 1.5
    social_weight: float = 1.5

class LottoOptimizer:
    def __init__(self, analyzer, config=None):
        """Initialise l'optimiseur avec des stratégies améliorées."""
        self.analyzer = analyzer
        self.config = config or OptimizationConfig()
        
        # Initialisation des attributs de base
        self.frequencies = analyzer.compute_frequency_analysis()
        self.delays = analyzer.compute_delay_analysis()
        self.correlation_matrix = analyzer.get_correlation_matrix()
        
        # Initialisation des poids par défaut
        self.pattern_weights = {
            'frequency': 0.25,
            'delay': 0.20,
            'correlation': 0.15,
            'distribution': 0.20,
            'historical': 0.20
        }
        
        # Initialisation du historique
        self.historical_success = {}
        
        # Mise à jour des poids et du historique
        self._update_weights_and_history()

    def _update_weights_and_history(self):
        """Met à jour les poids et l'historique après l'initialisation."""
        try:
            # Analyse des derniers tirages pour ajuster les poids
            recent_draws = self.analyzer.df.head(100)
            patterns_success = self._analyze_pattern_success(recent_draws)
            
            if patterns_success:
                self.pattern_weights = patterns_success
            
            # Analyse historique des combinaisons gagnantes
            self.historical_success = self._analyze_historical_success()
            
        except Exception as e:
            print(f"Avertissement lors de l'initialisation: {str(e)}")

    def _evaluate_combination(self, numbers: List[int], bonus: int) -> float:
        """Évaluation améliorée d'une combinaison avec analyse historique approfondie."""
        try:
            score = 0.0
            
            # 1. Analyse des fréquences avec fenêtre glissante
            recent_draws = self.analyzer.df.head(50)
            recent_freq = np.zeros(49)
            for _, row in recent_draws.iterrows():
                for i in range(1, 6):
                    recent_freq[row[f'boule_{i}']-1] += 1
            recent_freq /= len(recent_draws) * 5
            
            freq_score = sum(recent_freq[n-1] for n in numbers) / len(numbers)
            score += self.pattern_weights['frequency'] * freq_score
            
            # 2. Analyse des délais avec pondération exponentielle
            delays = self.delays['current_delays']
            max_delay = max(delays)
            if max_delay > 0:
                delay_scores = []
                for n in numbers:
                    delay = delays[n-1]
                    delay_score = 1 - np.exp(-delay / max_delay)
                    delay_scores.append(delay_score)
                score += self.pattern_weights['delay'] * np.mean(delay_scores)
            
            # 3. Analyse de corrélation améliorée
            correlation_penalty = 0
            for i, j in combinations(numbers, 2):
                corr = abs(self.correlation_matrix[i-1][j-1])
                if corr > 0.2:
                    correlation_penalty += (corr - 0.2) ** 2
            score -= self.pattern_weights['correlation'] * correlation_penalty
            
            # 4. Analyse de distribution
            dist_score = self._evaluate_enhanced_distribution(numbers)
            score += self.pattern_weights['distribution'] * dist_score
            
            # 5. Analyse des patterns historiques
            hist_score = self._evaluate_historical_patterns(numbers, bonus)
            score += self.pattern_weights['historical'] * hist_score
            
            return max(0.0, min(1.0, score))  # Normalisation entre 0 et 1
            
        except Exception as e:
            print(f"Erreur dans l'évaluation: {str(e)}")
            return 0.0

    def _evaluate_enhanced_distribution(self, numbers: List[int]) -> float:
        """Évaluation améliorée de la distribution des numéros."""
        try:
            score = 0.0
            
            # 1. Équilibre des zones
            zones = [0] * 4
            zone_size = 49 // 4
            for n in numbers:
                zone_idx = (n-1) // zone_size
                if zone_idx >= len(zones):
                    zone_idx = len(zones) - 1
                zones[zone_idx] += 1
            zone_balance = 1 - np.std(zones) / (np.mean(zones) + 1e-10)
            score += 0.3 * zone_balance
            
            # 2. Analyse des écarts
            sorted_nums = sorted(numbers)
            gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
            if gaps:
                gap_score = 1 - (np.std(gaps) / (np.mean(gaps) + 1e-10))
                score += 0.3 * gap_score
            
            # 3. Distribution des parités
            even_count = sum(1 for n in numbers if n % 2 == 0)
            parity_balance = 1 - abs(even_count - len(numbers)/2) / (len(numbers)/2)
            score += 0.2 * parity_balance
            
            # 4. Somme totale
            total = sum(numbers)
            sum_score = 1 - abs(total - 150) / 150  # 150 est une somme moyenne idéale
            score += 0.2 * sum_score
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"Erreur dans l'évaluation de distribution: {str(e)}")
            return 0.0
            
    def _analyze_pattern_success(self, recent_draws) -> Dict[str, float]:
        """Analyse le succès des différents patterns dans les tirages récents."""
        try:
            success_rates = {}
            total_score = 0
            
            # Analyse basée sur la fréquence
            freq_success = 0
            for _, row in recent_draws.iterrows():
                numbers = [row[f'boule_{i}'] for i in range(1, 6)]
                freq_score = sum(self.frequencies['normalized_frequencies'][n-1] for n in numbers) / 5
                if freq_score > 0.5:  # Seuil de succès
                    freq_success += 1
            success_rates['frequency'] = freq_success / len(recent_draws)
            total_score += success_rates['frequency']
            
            # Analyse basée sur les délais
            delay_success = 0
            max_delay = max(self.delays['current_delays'])
            for _, row in recent_draws.iterrows():
                numbers = [row[f'boule_{i}'] for i in range(1, 6)]
                delay_score = np.mean([self.delays['current_delays'][n-1] for n in numbers]) / max_delay
                if delay_score > 0.3:  # Seuil de succès
                    delay_success += 1
            success_rates['delay'] = delay_success / len(recent_draws)
            total_score += success_rates['delay']
            
            # Analyse basée sur la distribution
            dist_success = 0
            for _, row in recent_draws.iterrows():
                numbers = [row[f'boule_{i}'] for i in range(1, 6)]
                dist_score = self._evaluate_enhanced_distribution(numbers)
                if dist_score > 0.6:  # Seuil de succès
                    dist_success += 1
            success_rates['distribution'] = dist_success / len(recent_draws)
            total_score += success_rates['distribution']
            
            # Ajout des autres patterns avec poids par défaut
            success_rates['correlation'] = 0.15
            success_rates['historical'] = 0.20
            total_score += 0.35
            
            # Normalisation
            if total_score > 0:
                for key in success_rates:
                    success_rates[key] /= total_score
                    
            return success_rates
            
        except Exception as e:
            print(f"Erreur dans l'analyse des patterns: {str(e)}")
            return self.pattern_weights

    def _evaluate_historical_patterns(self, numbers: List[int], bonus: int) -> float:
        """Évaluation approfondie des patterns historiques."""
        try:
            score = 0.0
            recent_draws = self.analyzer.df.head(100)
            
            # Analyse des correspondances partielles
            partial_matches = []
            for _, row in recent_draws.iterrows():
                draw = [row[f'boule_{i}'] for i in range(1, 6)]
                common = len(set(numbers) & set(draw))
                partial_matches.append(common)
                
            if partial_matches:
                avg_matches = np.mean(partial_matches)
                score += 0.3 * (1 - abs(2.5 - avg_matches) / 2.5)
            
            # Analyse du bonus
            bonus_history = recent_draws['numero_chance'].value_counts()
            bonus_freq = bonus_history.get(bonus, 0) / len(recent_draws)
            score += 0.2 * (1 - bonus_freq)
            
            # Analyse des patterns de succession
            success_patterns = self._analyze_successful_patterns(recent_draws)
            pattern_score = self._match_current_patterns(numbers, success_patterns)
            score += 0.3 * pattern_score
            
            # Vérification des contraintes
            constraints_score = self._check_historical_constraints(numbers, recent_draws)
            score += 0.2 * constraints_score
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"Erreur dans l'évaluation des patterns historiques: {str(e)}")
            return 0.0

    def _analyze_successful_patterns(self, recent_draws) -> Dict:
        """Analyse détaillée des patterns qui ont eu du succès."""
        try:
            patterns = {
                'sum_ranges': defaultdict(int),
                'gap_patterns': defaultdict(int),
                'zone_patterns': defaultdict(int)
            }
            
            for _, row in recent_draws.iterrows():
                numbers = sorted([row[f'boule_{i}'] for i in range(1, 6)])
                
                # Analyse des sommes
                total = sum(numbers)
                sum_range = (total // 20) * 20
                patterns['sum_ranges'][sum_range] += 1
                
                # Analyse des écarts
                gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
                mean_gap = np.mean(gaps)
                gap_pattern = tuple(1 if gap > mean_gap else 0 for gap in gaps)
                patterns['gap_patterns'][gap_pattern] += 1
                
                # Analyse des zones
                zones = [0] * 4
                zone_size = 49 // 4
                for n in numbers:
                    zone_idx = min((n-1) // zone_size, 3)
                    zones[zone_idx] += 1
                patterns['zone_patterns'][tuple(zones)] += 1
            
            # Normalisation
            total_draws = len(recent_draws)
            for pattern_type in patterns:
                for key in patterns[pattern_type]:
                    patterns[pattern_type][key] /= total_draws
                    
            return patterns
            
        except Exception as e:
            print(f"Erreur dans l'analyse des patterns réussis: {str(e)}")
            return {'sum_ranges': {}, 'gap_patterns': {}, 'zone_patterns': {}}

    def _match_current_patterns(self, numbers: List[int], success_patterns: Dict) -> float:
        """Évalue la correspondance avec les patterns à succès."""
        try:
            score = 0.0
            numbers = sorted(numbers)
            
            # Vérification de la somme
            total = sum(numbers)
            sum_range = (total // 20) * 20
            score += success_patterns['sum_ranges'].get(sum_range, 0)
            
            # Vérification des écarts
            gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            mean_gap = np.mean(gaps)
            gap_pattern = tuple(1 if gap > mean_gap else 0 for gap in gaps)
            score += success_patterns['gap_patterns'].get(gap_pattern, 0)
            
            # Vérification des zones
            zones = [0] * 4
            zone_size = 49 // 4
            for n in numbers:
                zone_idx = min((n-1) // zone_size, 3)
                zones[zone_idx] += 1
            score += success_patterns['zone_patterns'].get(tuple(zones), 0)
            
            return score / 3
            
        except Exception as e:
            print(f"Erreur dans la correspondance des patterns: {str(e)}")
            return 0.0

    def _analyze_historical_success(self) -> Dict:
        """Analyse globale des patterns historiques réussis."""
        try:
            patterns = {}
            recent_draws = self.analyzer.df.head(200)
            
            for _, row in recent_draws.iterrows():
                draw = sorted([row[f'boule_{i}'] for i in range(1, 6)])
                
                # Patterns de répartition bas/haut
                low_high = sum(1 for n in draw if n <= 25)
                patterns[f'low_high_{low_high}'] = patterns.get(f'low_high_{low_high}', 0) + 1
                
                # Patterns de parité
                even_count = sum(1 for n in draw if n % 2 == 0)
                patterns[f'even_{even_count}'] = patterns.get(f'even_{even_count}', 0) + 1
                
                # Patterns de somme
                total = sum(draw)
                sum_range = (total // 20) * 20
                patterns[f'sum_{sum_range}'] = patterns.get(f'sum_{sum_range}', 0) + 1
            
            # Normalisation
            total_draws = len(recent_draws)
            return {k: v/total_draws for k, v in patterns.items()}
            
        except Exception as e:
            print(f"Erreur dans l'analyse historique: {str(e)}")
            return {}

    def _check_historical_constraints(self, numbers: List[int], recent_draws) -> float:
        """Vérifie le respect des contraintes historiques."""
        try:
            score = 0.0
            
            # Vérification des répétitions exactes
            for _, row in recent_draws.iterrows():
                draw = set(row[f'boule_{i}'] for i in range(1, 6))
                if draw == set(numbers):
                    return 0.0
            
            # Analyse des intervalles
            intervals = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            unique_intervals = len(set(intervals))
            score += 0.5 * (unique_intervals / (len(numbers)-1))
            
            # Vérification des tendances récentes
            recent_numbers = set()
            for _, row in recent_draws.head(10).iterrows():
                for i in range(1, 6):
                    recent_numbers.add(row[f'boule_{i}'])
            
            overlap = len(set(numbers) & recent_numbers)
            score += 0.5 * (1 - (overlap / len(numbers)))
            
            return score
            
        except Exception as e:
            print(f"Erreur dans la vérification des contraintes: {str(e)}")
            return 0.0
            
    def _genetic_algorithm(self) -> Tuple[List[int], int]:
        """Algorithme génétique optimisé pour la génération de numéros."""
        try:
            # Population initiale
            population = []
            for _ in range(self.config.population_size):
                numbers = sorted(random.sample(range(1, 50), 5))
                bonus = random.randint(1, 10)
                population.append((numbers, bonus))

            best_solution = None
            best_score = float('-inf')

            for generation in range(self.config.generations):
                try:
                    # Évaluation
                    fitness_scores = []
                    for nums, bonus in population:
                        score = self._evaluate_combination(nums, bonus)
                        fitness_scores.append(score)
                        
                        # Mise à jour de la meilleure solution
                        if score > best_score:
                            best_score = score
                            best_solution = (nums.copy(), bonus)

                    # Nouvelle population avec élitisme
                    new_population = []
                    
                    # Conservation des 2 meilleurs individus
                    sorted_pop = sorted(zip(population, fitness_scores), 
                                     key=lambda x: x[1], reverse=True)
                    new_population.extend([p[0] for p in sorted_pop[:2]])

                    # Complétion de la nouvelle population
                    while len(new_population) < self.config.population_size:
                        # Sélection par tournoi
                        tournament1 = random.sample(list(enumerate(fitness_scores)), 
                                                 min(self.config.tournament_size, len(population)))
                        tournament2 = random.sample(list(enumerate(fitness_scores)), 
                                                 min(self.config.tournament_size, len(population)))
                        
                        parent1_idx = max(tournament1, key=lambda x: x[1])[0]
                        parent2_idx = max(tournament2, key=lambda x: x[1])[0]
                        
                        parent1 = population[parent1_idx]
                        parent2 = population[parent2_idx]

                        # Croisement
                        if random.random() < 0.8:
                            split_point = random.randint(1, 4)
                            child1_numbers = parent1[0][:split_point] + \
                                          [n for n in parent2[0] if n not in parent1[0][:split_point]]
                            child2_numbers = parent2[0][:split_point] + \
                                          [n for n in parent1[0] if n not in parent2[0][:split_point]]
                            
                            # Complétion si nécessaire
                            while len(child1_numbers) < 5:
                                new_num = random.randint(1, 49)
                                if new_num not in child1_numbers:
                                    child1_numbers.append(new_num)
                            while len(child2_numbers) < 5:
                                new_num = random.randint(1, 49)
                                if new_num not in child2_numbers:
                                    child2_numbers.append(new_num)
                            
                            # Croisement du bonus
                            child1_bonus = parent1[1] if random.random() < 0.5 else parent2[1]
                            child2_bonus = parent2[1] if random.random() < 0.5 else parent1[1]
                        else:
                            child1_numbers, child1_bonus = parent1
                            child2_numbers, child2_bonus = parent2
                            child1_numbers = child1_numbers.copy()
                            child2_numbers = child2_numbers.copy()

                        # Mutation des nombres
                        for child_numbers in [child1_numbers, child2_numbers]:
                            if random.random() < self.config.mutation_rate:
                                idx = random.randint(0, 4)
                                new_number = random.randint(1, 49)
                                while new_number in child_numbers:
                                    new_number = random.randint(1, 49)
                                child_numbers[idx] = new_number

                        # Mutation des bonus
                        for bonus in [child1_bonus, child2_bonus]:
                            if random.random() < self.config.mutation_rate:
                                bonus = random.randint(1, 10)

                        # Ajout à la nouvelle population
                        new_population.append((sorted(child1_numbers[:5]), child1_bonus))
                        if len(new_population) < self.config.population_size:
                            new_population.append((sorted(child2_numbers[:5]), child2_bonus))

                    population = new_population

                except Exception as e:
                    print(f"Erreur dans la génération {generation}: {str(e)}")
                    continue

            return best_solution if best_solution else (random.sample(range(1, 50), 5), 
                                                      random.randint(1, 10))

        except Exception as e:
            print(f"Erreur dans l'algorithme génétique: {str(e)}")
            return random.sample(range(1, 50), 5), random.randint(1, 10)

    def _particle_swarm_optimization(self) -> Tuple[List[int], int]:
        """Optimisation par essaim particulaire (PSO) améliorée."""
        class Particle:
            def __init__(self, numbers, bonus):
                self.numbers = list(numbers)[:5]
                while len(self.numbers) < 5:
                    self.numbers.append(random.randint(1, 49))
                self.bonus = max(1, min(10, bonus))
                
                self.velocity_numbers = [random.uniform(-2, 2) for _ in range(5)]
                self.velocity_bonus = random.uniform(-1, 1)
                
                self.best_numbers = self.numbers.copy()
                self.best_bonus = self.bonus
                self.best_score = float('-inf')

            def update_position(self):
                """Met à jour la position de manière sécurisée."""
                # Mise à jour des nombres
                new_numbers = []
                for i, v in enumerate(self.velocity_numbers):
                    new_val = int(round(self.numbers[i] + v))
                    new_val = max(1, min(49, new_val))
                    new_numbers.append(new_val)
                
                # Assure l'unicité
                self.numbers = list(set(new_numbers))
                while len(self.numbers) < 5:
                    new_num = random.randint(1, 49)
                    if new_num not in self.numbers:
                        self.numbers.append(new_num)
                self.numbers = sorted(self.numbers[:5])
                
                # Mise à jour du bonus
                self.bonus = int(round(max(1, min(10, self.bonus + self.velocity_bonus))))

        try:
            particles = []
            global_best_numbers = None
            global_best_bonus = None
            global_best_score = float('-inf')

            # Initialisation des particules
            for _ in range(self.config.particle_count):
                init_numbers = random.sample(range(1, 50), 5)
                init_bonus = random.randint(1, 10)
                particles.append(Particle(init_numbers, init_bonus))

            # Boucle principale
            for _ in range(self.config.generations):
                for particle in particles:
                    try:
                        # Évaluation
                        current_score = self._evaluate_combination(particle.numbers, particle.bonus)

                        # Mise à jour du meilleur personnel
                        if current_score > particle.best_score:
                            particle.best_score = current_score
                            particle.best_numbers = particle.numbers.copy()
                            particle.best_bonus = particle.bonus

                        # Mise à jour du meilleur global
                        if current_score > global_best_score:
                            global_best_score = current_score
                            global_best_numbers = particle.numbers.copy()
                            global_best_bonus = particle.bonus

                        # Mise à jour des vitesses
                        for i in range(5):
                            r1, r2 = random.random(), random.random()
                            cognitive = self.config.cognitive_weight * r1 * \
                                      (particle.best_numbers[i] - particle.numbers[i])
                            social = self.config.social_weight * r2 * \
                                   (global_best_numbers[i] - particle.numbers[i])
                            
                            particle.velocity_numbers[i] = \
                                self.config.inertia_weight * particle.velocity_numbers[i] + \
                                cognitive + social

                        # Mise à jour de la vitesse du bonus
                        r1, r2 = random.random(), random.random()
                        cognitive_bonus = self.config.cognitive_weight * r1 * \
                                       (particle.best_bonus - particle.bonus)
                        social_bonus = self.config.social_weight * r2 * \
                                    (global_best_bonus - particle.bonus)
                        
                        particle.velocity_bonus = \
                            self.config.inertia_weight * particle.velocity_bonus + \
                            cognitive_bonus + social_bonus

                        # Mise à jour de la position
                        particle.update_position()

                    except Exception as e:
                        print(f"Erreur pour une particule: {str(e)}")
                        continue

            if global_best_numbers is None:
                return random.sample(range(1, 50), 5), random.randint(1, 10)

            return global_best_numbers, global_best_bonus

        except Exception as e:
            print(f"Erreur dans PSO: {str(e)}")
            return random.sample(range(1, 50), 5), random.randint(1, 10)

    def optimize_numbers(self, method: str = 'hybrid') -> Tuple[List[int], int]:
        """Méthode principale d'optimisation avec stratégie hybride."""
        try:
            if method == 'hybrid':
                candidates = []
                
                # Méthode génétique
                for _ in range(3):
                    try:
                        numbers, bonus = self._genetic_algorithm()
                        score = self._evaluate_combination(numbers, bonus)
                        candidates.append((numbers, bonus, score))
                    except Exception as e:
                        print(f"Erreur dans l'algorithme génétique: {str(e)}")
                        continue
                
                # Méthode PSO
                for _ in range(3):
                    try:
                        numbers, bonus = self._particle_swarm_optimization()
                        score = self._evaluate_combination(numbers, bonus)
                        candidates.append((numbers, bonus, score))
                    except Exception as e:
                        print(f"Erreur dans PSO: {str(e)}")
                        continue
                
                if not candidates:
                    return random.sample(range(1, 50), 5), random.randint(1, 10)
                
                best_candidate = max(candidates, key=lambda x: x[2])
                return best_candidate[0], best_candidate[1]
                
            elif method == 'genetic':
                return self._genetic_algorithm()
            else:
                return self._particle_swarm_optimization()
                
        except Exception as e:
            print(f"Erreur dans optimize_numbers: {str(e)}")
            return random.sample(range(1, 50), 5), random.randint(1, 10)