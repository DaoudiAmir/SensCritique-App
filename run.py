#!/usr/bin/env python3
"""
Script de lancement pour la démo SensCritique Streamlit
Développé par: Amir Salah Eddine Daoudi
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Vérifie que les dépendances sont installées."""
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import plotly
        print("✅ Toutes les dépendances sont installées")
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("📦 Installation des dépendances...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True

def main():
    """Lance l'application Streamlit."""
    print("🎬 SensCritique - Démo de Recommandation")
    print("=" * 50)
    print("Développé par: Amir Salah Eddine Daoudi")
    print("Email: daoudiamirsalaheddine@gmail.com")
    print("LinkedIn: https://www.linkedin.com/in/amir-salah-eddine-daoudi/")
    print("=" * 50)
    
    # Vérifier les dépendances
    if not check_dependencies():
        return
    
    # Changer vers le répertoire du script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🚀 Lancement de l'application Streamlit...")
    print("🌐 L'application sera disponible sur: http://localhost:8501")
    print("⚠️  Appuyez sur Ctrl+C pour arrêter")
    
    try:
        # Lancer Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--browser.serverAddress", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application arrêtée")

if __name__ == "__main__":
    main()
