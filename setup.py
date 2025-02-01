from setuptools import setup, find_packages

setup(
    name="omen-vectorstore",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "kbcstorage>=0.9.2",
        "qdrant-client>=1.7.0",
        "sentence-transformers>=2.2.2",
        "python-dotenv>=1.0.0",
        "msgpack>=1.0.5",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.1",
            "flake8>=6.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "mypy>=1.5.1",
            "types-requests>=2.31.0.2",
            "mkdocs>=1.5.2",
            "mkdocs-material>=9.2.0",
        ],
    },
    python_requires=">=3.11",
)
