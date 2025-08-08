"""
OMEN Platform - Ontology-powered Metadata Engine.

Namespace package bootstrap to allow subpackages from installed distributions
to coexist under the `omen` namespace inside this repo.
"""

from pkgutil import extend_path

# Enable namespace package behavior so `omen.*` from installed packages are visible
__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

# Version of the OMEN platform
__version__ = "0.1.1"