"""
OMEN Platform namespace package.

This file allows Python to recognize 'omen' as a namespace package
that can span multiple directories.
"""

__path__ = __import__('pkgutil').extend_path(__path__, __name__)
