from setuptools import setup, find_packages

setup(
    name="mlguard",
    version="7.2.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
        "pandas>=1.5.0",
        "joblib>=1.2.0",
        "scikit-learn>=1.1.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": ["pytest", "httpx"],
    },
    entry_points={
        "console_scripts": [
            # `pip install -e .` then just run: mlguard check
            "mlguard=ml_guard.cli:app",
        ],
    },
    author="ML Guard Team",
    description="Python SDK and CLI for ML Guard v7.2 — Enterprise AI Governance Platform",
    long_description=open("../../README.md", encoding="utf-8").read() if True else "",
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
