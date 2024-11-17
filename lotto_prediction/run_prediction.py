#!/usr/bin/env python3
import argparse
import subprocess
import sys
from typing import List, Optional

class PredictionSystemHelper:
    """Assistant pour l'utilisation du système de prédiction."""
    
    def __init__(self):
        self.commands = {
            'setup': self.setup_system,
            'train': self.train_model,
            'predict': self.make_prediction,
            'analyze': self.analyze_system,
            'validate': self.validate_system
        }

    def setup_system(self) -> None:
        """Configure le système initialement."""
        print("Installation des dépendances...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\nConfiguration terminée.")

    def train_model(self, temperature: float = 0.3) -> None:
        """Entraîne le modèle."""
        cmd = ["python", "main.py", "--mode", "train", "--temperature", str(temperature)]
        subprocess.run(cmd)

    def make_prediction(self, output_format: str = 'detailed', temperature: float = 0.3) -> None:
        """Génère une prédiction."""
        cmd = [
            "python", "main.py",
            "--mode", "predict",
            "--output-format", output_format,
            "--temperature", str(temperature)
        ]
        subprocess.run(cmd)

    def analyze_system(self) -> None:
        """Lance l'analyse du système."""
        subprocess.run(["python", "main.py", "--mode", "analyze"])

    def validate_system(self) -> None:
        """Valide le système."""
        subprocess.run(["python", "main.py", "--mode", "validate"])

    def run_interactive(self) -> None:
        """Mode interactif pour guider l'utilisateur."""
        print("Bienvenue dans l'assistant de prédiction Lotto")
        
        while True:
            print("\nQue souhaitez-vous faire ?")
            print("1. Configuration initiale")
            print("2. Entraîner le modèle")
            print("3. Faire une prédiction")
            print("4. Analyser le système")
            print("5. Valider le système")
            print("6. Quitter")
            
            choice = input("\nVotre choix (1-6) : ")
            
            if choice == '1':
                self.setup_system()
            elif choice == '2':
                temp = input("Température (0.1-1.0, défaut 0.3) : ") or "0.3"
                self.train_model(float(temp))
            elif choice == '3':
                format_choice = input("Format de sortie (simple/detailed/json) [detailed] : ") or "detailed"
                temp = input("Température (0.1-1.0, défaut 0.3) : ") or "0.3"
                self.make_prediction(format_choice, float(temp))
            elif choice == '4':
                self.analyze_system()
            elif choice == '5':
                self.validate_system()
            elif choice == '6':
                print("\nAu revoir!")
                break
            else:
                print("Choix invalide, veuillez réessayer.")

def main():
    """Point d'entrée principal."""
    helper = PredictionSystemHelper()
    
    if len(sys.argv) > 1:
        # Mode ligne de commande
        parser = argparse.ArgumentParser(description='Assistant de prédiction Lotto')
        parser.add_argument('command', choices=helper.commands.keys(),
                          help='Commande à exécuter')
        parser.add_argument('--temperature', type=float, default=0.3,
                          help='Température pour le modèle')
        parser.add_argument('--output-format', choices=['simple', 'detailed', 'json'],
                          default='detailed', help='Format de sortie')
        
        args = parser.parse_args()
        helper.commands[args.command]()
    else:
        # Mode interactif
        helper.run_interactive()

if __name__ == "__main__":
    main()