# Guide d'Utilisation du Système de Prédiction Lotto

## Table des Matières
1. Installation
2. Structure du Projet
3. Modes d'Utilisation
4. Guide Pas à Pas
5. Options Avancées
6. Dépannage
7. Bonnes Pratiques

## 1. Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Espace disque : minimum 500 Mo
- RAM recommandée : 4 Go minimum

### Étapes d'Installation
1. Clonez ou téléchargez le projet dans un dossier local
2. Ouvrez un terminal dans le dossier du projet
3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## 2. Structure du Projet

```
lotto_prediction/
│
├── config/               # Configurations du système
├── src/                 # Code source principal
├── data/                # Données d'entrée
├── partie_1.py          # Analyse des données
├── partie_2.py          # Optimisation
├── partie_3A.py         # Validation
├── partie_3B.py         # Prédiction
├── systeme.py           # Système principal
└── main.py              # Point d'entrée
```

## 3. Modes d'Utilisation

Le système offre deux modes d'utilisation :

### Mode Interactif (Recommandé pour les débutants)
Lancez l'assistant interactif :
```bash
python run_prediction.py
```

### Mode Ligne de Commande (Pour utilisateurs avancés)
Commandes principales :
```bash
python main.py --mode [train|predict|validate|analyze]
```

## 4. Guide Pas à Pas

### Étape 1 : Préparation Initiale
1. Installation des dépendances :
```bash
python run_prediction.py setup
```

### Étape 2 : Entraînement Initial
```bash
python main.py --mode train
```
Durée approximative : 10-30 minutes selon votre machine

### Étape 3 : Validation (Recommandé)
```bash
python main.py --mode validate
```
Vérifie la qualité du modèle

### Étape 4 : Prédiction
Version simple :
```bash
python main.py --mode predict
```

Version détaillée :
```bash
python main.py --mode predict --output-format detailed
```

## 5. Options Avancées

### Formats de Sortie
- simple : Affichage basique des numéros
- detailed : Analyse complète avec métriques
- json : Format structuré pour intégration

```bash
python main.py --mode predict --output-format [simple|detailed|json]
```

### Ajustement du Modèle
Température (contrôle la "créativité" du modèle) :
```bash
python main.py --mode predict --temperature 0.3
```
- Valeurs : 0.1 (très conservateur) à 1.0 (très créatif)
- Recommandé : 0.3-0.5

### Analyse Approfondie
```bash
python main.py --mode analyze
```
Fournit des statistiques détaillées sur les données

## 6. Dépannage

### Messages d'Erreur Courants

1. "Model not trained"
   - Solution : Exécutez d'abord l'entraînement
   ```bash
   python main.py --mode train
   ```

2. "Invalid temperature"
   - Solution : Utilisez une valeur entre 0.1 et 1.0

3. "Dataset not found"
   - Solution : Vérifiez que le fichier dataset.csv est présent dans le dossier data/

### Vérification de l'Installation
```bash
python run_prediction.py setup --check
```

## 7. Bonnes Pratiques

1. Entraînement
   - Réentraînez le modèle régulièrement
   - Conservez un historique des performances

2. Validation
   - Validez après chaque entraînement
   - Surveillez les métriques de qualité

3. Prédiction
   - Utilisez le format détaillé pour les analyses importantes
   - Vérifiez les scores de confiance

4. Maintenance
   - Sauvegardez vos configurations personnalisées
   - Mettez à jour les dépendances régulièrement

## Notes Importantes

1. Performance
   - Les premières prédictions peuvent être plus lentes
   - L'analyse complète demande plus de ressources

2. Limitations
   - Le système est éducatif
   - Les prédictions sont des estimations statistiques

3. Sécurité
   - Ne partagez pas vos modèles entraînés
   - Sauvegardez régulièrement vos données

## Support

Pour plus d'aide ou en cas de problèmes :
1. Vérifiez la documentation
2. Utilisez le mode --help
3. Contactez l'équipe de support

```bash
python main.py --help
```
