from setuptools import setup, find_namespace_packages

setup(
    name="omen-ontology",
    version="0.1.1",
    description="Ontology system for the OMEN platform",
    author="Keboola",
    author_email="info@keboola.com",
    url="https://github.com/keboola/omen-platform",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        "omen-core==0.1.1",
        "rdflib>=6.3.2",
        "networkx>=3.1",
        "graphviz>=0.20.1",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
