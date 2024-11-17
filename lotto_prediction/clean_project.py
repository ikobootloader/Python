import os
import shutil
from pathlib import Path
import sys

def clean_project(project_path: str):
    """Nettoie tous les fichiers cache Python du projet."""
    
    # Compteurs pour le rapport
    deleted_dirs = 0
    deleted_files = 0
    
    # Extensions à supprimer
    cache_extensions = ['.pyc', '.pyo', '.pyd', '.pt']
    
    # Dossiers à supprimer
    cache_dirs = ['__pycache__', '.pytest_cache', '.mypy_cache', '.hypothesis', 
                 '.coverage', '.tox', '.eggs', '*.egg-info']
    
    project_path = Path(project_path)
    print(f"Nettoyage du projet dans : {project_path.absolute()}\n")
    
    # Parcours de tous les fichiers et dossiers
    for root, dirs, files in os.walk(project_path, topdown=False):
        root_path = Path(root)
        
        # Suppression des fichiers cache
        for file in files:
            file_path = root_path / file
            if any(file.endswith(ext) for ext in cache_extensions):
                try:
                    file_path.unlink()
                    print(f"Fichier supprimé : {file_path}")
                    deleted_files += 1
                except Exception as e:
                    print(f"Erreur lors de la suppression de {file_path}: {e}")
        
        # Suppression des dossiers cache
        for dir_name in dirs:
            dir_path = root_path / dir_name
            if any(dir_name == cache_dir.strip('*') for cache_dir in cache_dirs):
                try:
                    shutil.rmtree(dir_path)
                    print(f"Dossier supprimé : {dir_path}")
                    deleted_dirs += 1
                except Exception as e:
                    print(f"Erreur lors de la suppression de {dir_path}: {e}")
    
    # Rapport final
    print(f"\nNettoyage terminé !")
    print(f"Dossiers supprimés : {deleted_dirs}")
    print(f"Fichiers supprimés : {deleted_files}")

if __name__ == "__main__":
    # Utilise le chemin fourni en argument ou le dossier courant
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."
    clean_project(project_path)