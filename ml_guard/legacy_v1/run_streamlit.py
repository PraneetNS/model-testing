#!/usr/bin/env python3
"""
Run ML Guard Streamlit Application
Fireflink-Style ML Model Testing Platform
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        import joblib
        print("[OK] All dependencies are installed")
        return True
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run: pip install -r requirements-streamlit.txt")
        return False

def check_nlp_setup():
    """Check NLP setup (simplified version without spaCy)."""
    print("[OK] Using simplified NLP processing (no external models required)")
    return True

def main():
    """Run the Streamlit application."""
    print("ML Guard - Fireflink Style Streamlit Application")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("[ERROR] streamlit_app.py not found in current directory")
        print("Please run this script from the ml_guard directory")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check NLP setup
    if not check_nlp_setup():
        print("[WARNING] NLP setup incomplete")

    # Start Streamlit app
    print("Starting ML Guard Streamlit Application...")
    print("Open your browser to: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("=" * 50)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "true",
            "--server.address", "0.0.0.0",
            "--server.port", "8501"
        ], check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to start Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()