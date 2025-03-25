from setuptools import setup, find_namespace_packages

setup(
    name="omen-cli",
    version="0.1.0",
    description="OMEN Platform - Command Line Interface",
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
        "click>=8.1.3",
        "rich>=12.5.0",
    ],
    entry_points={
        "console_scripts": [
            "omen=omen.cli:cli",
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
