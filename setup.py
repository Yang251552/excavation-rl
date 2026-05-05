"""Setup script for excavation-rl project."""

from setuptools import setup, find_packages

setup(
    name="excavation-rl",
    version="0.1.0",
    description="Reinforcement Learning for Particle-Based Excavation in Isaac Lab",
    author="ETH Zurich RSL",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
        "matplotlib>=3.7.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.8",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
        ],
    },
)
