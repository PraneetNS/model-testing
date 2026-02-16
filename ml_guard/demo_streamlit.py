#!/usr/bin/env python3
"""
Demo script for ML Guard Streamlit UI
Shows how to use the Fireflink-style interface
"""

import os
import subprocess
import sys
from pathlib import Path

def demo():
    """Run a demonstration of the Streamlit UI."""
    print("🔥 ML Guard Streamlit UI Demo")
    print("=" * 40)

    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ Error: Please run this from the ml_guard directory")
        return

    print("📋 This demo will show you how to use the ML Guard Streamlit UI:")
    print()
    print("1. 🎨 Fireflink-inspired modern design")
    print("2. 🧠 NLP-powered test selection")
    print("3. 📤 Drag-and-drop model upload")
    print("4. 📊 Real-time test execution")
    print("5. 📈 Comprehensive results dashboard")
    print()

    print("🚀 Starting Streamlit application...")
    print("📱 Once started, open: http://localhost:8501")
    print()
    print("💡 Demo Usage Instructions:")
    print("   - Upload a model (.pkl) in the sidebar")
    print("   - Upload training/validation/test datasets")
    print("   - Try NLP testing: 'Run accuracy and bias tests'")
    print("   - Or use manual selection for specific tests")
    print("   - View results in the dashboard tab")
    print()

    # Start the Streamlit app
    try:
        subprocess.run([
            sys.executable, "run_streamlit.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Demo completed!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    demo()