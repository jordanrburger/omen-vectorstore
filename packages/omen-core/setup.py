from setuptools import setup, find_packages

setup(
    name="omen-core",
    version="0.1.1",
    description="Core functionality for the OMEN platform",
    author="Keboola",
    author_email="support@keboola.com",
    url="https://github.com/keboola/omen",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pydantic>=2.0.0",
        "tenacity>=8.0.0",
        "openai>=1.0.0",
        "anthropic>=0.3.0",
        "python-dotenv>=1.0.0",
        "rich>=13.0.0",
        "typing-extensions>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
