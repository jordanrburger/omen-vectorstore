from setuptools import setup, find_namespace_packages

setup(
    name="omen-core",
    version="0.1.0",
    description="Core functionality for the OMEN platform",
    author="Keboola",
    author_email="info@keboola.com",
    url="https://github.com/keboola/omen-platform",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "tqdm>=4.65.0",
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
