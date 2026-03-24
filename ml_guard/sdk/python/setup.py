from setuptools import setup, find_packages

setup(
    name="ml-guard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.1",
        "pandas>=1.2.0",
        "joblib>=1.0.1",
        "scikit-learn>=0.24.0",
    ],
    entry_points={
        "console_scripts": [
            "mlguard=ml_guard.cli:main",
        ],
    },
    author="ML Guard Team",
    description="Python SDK and CLI for ML Guard Quality Governance Platform",
    python_requires=">=3.8",
)
