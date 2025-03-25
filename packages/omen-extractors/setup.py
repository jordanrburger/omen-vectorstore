from setuptools import setup, find_namespace_packages

setup(
    name="omen-extractors",
    version="0.1.0",
    description="OMEN Platform - Metadata Extractors",
    author="Keboola",
    author_email="info@keboola.com",
    url="https://github.com/keboola/omen-platform",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "omen-core>=0.1.0",
        "omen-vectorstore>=0.1.0",
        "omen-ontology>=0.1.0",
    ],
    extras_require={
        "keboola": ["kbcstorage==0.9.2"],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.1.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 