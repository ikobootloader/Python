# Exemple d'Utilisation Quotidienne

## Scénario : Première Utilisation

### 1. Installation initiale
```bash
# Installation des dépendances
python run_prediction.py setup

# Vérification de l'installation
python run_prediction.py setup --check
```

### 2. Premier entraînement
```bash
# Entraînement avec paramètres par défaut
python main.py --mode train

# Vérification de la qualité
python main.py --mode validate
```

### 3. Première prédiction
```bash
# Prédiction détaillée
python main.py --mode predict --output-format detailed
```

## Scénario : Utilisation Régulière

### 1. Vérification rapide
```bash
# Prédiction simple
python main.py --mode predict --output-format simple
```

### 2. Analyse approfondie
```bash
# Analyse complète
python main.py --mode analyze

# Prédiction détaillée
python main.py --mode predict --output-format detailed --temperature 0.3
```

### 3. Export des résultats
```bash
# Export en JSON
python main.py --mode predict --output-format json > resultats.json
```

## Conseils d'Utilisation Quotidienne

1. Vérification Quotidienne
   - Lancez une prédiction simple chaque matin
   - Vérifiez les scores de confiance

2. Analyse Hebdomadaire
   - Effectuez une analyse complète
   - Comparez les performances

3. Maintenance Mensuelle
   - Réentraînez le modèle
   - Sauvegardez les configurations
   - Mettez à jour les dépendances

4. Documentation
   - Notez les configurations performantes
   - Gardez un historique des prédictions
   - Documentez les anomalies

## Automatisation

Vous pouvez créer un script bash/batch pour automatiser les tâches quotidiennes :

```bash
#!/bin/bash
# daily_prediction.sh

# Date du jour
DATE=$(date +%Y-%m-%d)

# Création du dossier de résultats
mkdir -p "resultats/$DATE"

# Prédiction du jour
python main.py --mode predict --output-format json > "resultats/$DATE/prediction.json"

# Analyse
python main.py --mode analyze > "resultats/$DATE/analyse.txt"

echo "Prédiction du $DATE terminée"
```
