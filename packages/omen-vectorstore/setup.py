from setuptools import setup, find_namespace_packages

setup(
    name="omen-vectorstore",
    version="0.1.0",
    description="Vector database integration for the OMEN platform",
    author="Keboola",
    author_email="info@keboola.com",
    url="https://github.com/keboola/omen-platform",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        "omen-core==0.1.0",
        "qdrant-client>=1.7.0",
        "sentence-transformers>=2.2.2",
        "msgpack>=1.0.5",
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
