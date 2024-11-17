import logging
import argparse
from datetime import datetime
from typing import Dict, Any

from systeme import LottoPredictionSystem
from config.settings import MLConfig, ValidationConfig, ProcessingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse les arguments de la ligne de commande avec les nouvelles options."""
    parser = argparse.ArgumentParser(description='Système de prédiction Lotto amélioré')
    parser.add_argument('--data-path', default='data/dataset.csv', help='Chemin vers le dataset')
    parser.add_argument('--mode', choices=['train', 'predict', 'validate', 'analyze'], 
                       default='predict', help='Mode de fonctionnement')
    parser.add_argument('--model-type', choices=['lstm', 'transformer', 'ensemble'],
                       default='lstm', help='Type de modèle à utiliser')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Température pour le softmax (entre 0.1 et 1.0)')
    parser.add_argument('--output-format', choices=['simple', 'detailed', 'json'],
                       default='detailed', help='Format de sortie des résultats')
    return parser.parse_args()

def setup_configurations(args) -> tuple:
    """Configure les paramètres du système selon les arguments."""
    ml_config = MLConfig(temperature=args.temperature)
    validation_config = ValidationConfig()
    processing_config = ProcessingConfig(temperature=args.temperature)
    
    return ml_config, validation_config, processing_config

def format_prediction_output(prediction: tuple, analysis: Dict[str, Any], format_type: str) -> str:
    """Formate la sortie selon le format demandé."""
    numbers, bonus = prediction
    
    if format_type == 'simple':
        return f"Numéros: {sorted(numbers)}, Bonus: {bonus}"
    
    elif format_type == 'detailed':
        output = [
            "\n=== Prédiction détaillée ===",
            f"Numéros: {sorted(numbers)}",
            f"Numéro chance: {bonus}",
            "\n=== Métriques de qualité ===",
            f"Score global: {analysis['quality_metrics']['global_score']:.2%}",
            f"Validité: {'✓' if analysis['quality_metrics']['validity_score'] == 1.0 else '✗'}",
            f"Distribution: {analysis['quality_metrics']['distribution_score']:.2%}",
            f"Diversité: {analysis['quality_metrics']['diversity_score']:.2%}",
            
            "\n=== Analyse statistique ===",
            f"Ratio bas/haut: {abs(0.5 - analysis['quality_metrics']['low_high_ratio'])*100:.1f}% de déviation",
            f"Score d'optimisation: {analysis['optimization_score']:.3f}",
        ]
        return "\n".join(output)
    
    else:  # json
        import json
        return json.dumps({
            'prediction': {
                'numbers': sorted(numbers),
                'bonus': bonus
            },
            'analysis': analysis
        }, indent=2)

def run_analysis_mode(system: LottoPredictionSystem):
    """Exécute une analyse approfondie des données historiques."""
    logger.info("Analyse approfondie des données historiques...")
    
    # Analyse statistique
    stats = system.analyzer.perform_statistical_tests()
    frequencies = system.analyzer.compute_frequency_analysis()
    patterns = system.analyzer.analyze_patterns([])  # patterns globaux
    
    # Affichage des résultats
    print("\n=== Analyse statistique approfondie ===")
    print("\nTests statistiques:")
    for test, value in stats.items():
        print(f"{test}: {value:.4f}")
    
    print("\nAnalyse des fréquences:")
    print("Top 5 numéros les plus fréquents:")
    sorted_freq = sorted(enumerate(frequencies['main_numbers']), 
                        key=lambda x: x[1], reverse=True)
    for num, freq in sorted_freq[:5]:
        print(f"Numéro {num+1}: {freq:.1%}")
    
    print("\nPatterns identifiés:")
    for key, value in patterns.items():
        if not isinstance(value, (list, np.ndarray)):
            print(f"{key}: {value:.3f}")

def main():
    """Fonction principale avec les nouvelles fonctionnalités."""
    args = parse_arguments()
    
    try:
        # Configuration du système
        ml_config, validation_config, processing_config = setup_configurations(args)
        
        # Initialisation du système
        logger.info("Initialisation du système de prédiction...")
        system = LottoPredictionSystem(
            filepath=args.data_path,
            ml_config=ml_config,
            validation_config=validation_config,
            processing_config=processing_config
        )
        
        # Exécution selon le mode choisi
        if args.mode == 'train':
            logger.info("Démarrage de l'entraînement...")
            system.train_system()
            
        elif args.mode == 'predict':
            logger.info("Génération des prédictions...")
            prediction, analysis = system.predict_next_draw()
            print(format_prediction_output(prediction, analysis, args.output_format))
            
        elif args.mode == 'validate':
            logger.info("Validation du système...")
            results = system.validator.perform_backtesting()
            print("\n=== Résultats de la validation ===")
            for metric, value in results.items():
                print(f"{metric}: {value:.4f}")
                
        else:  # analyze
            run_analysis_mode(system)
            
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {str(e)}")
        raise

if __name__ == "__main__":
    main()