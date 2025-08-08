from setuptools import setup, find_namespace_packages

setup(
    name="omen-api",
    version="0.1.1",
    description="OMEN Platform - API Server",
    author="Keboola",
    author_email="info@keboola.com",
    url="https://github.com/keboola/omen-platform",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "omen-core>=0.1.1",
        "omen-vectorstore>=0.1.1",
        "omen-ontology>=0.1.1",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "pydantic>=1.10.7",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.1.0",
            "mypy>=1.0.0",
            "httpx>=0.23.0",
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
