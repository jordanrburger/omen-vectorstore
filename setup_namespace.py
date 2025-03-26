#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script fixes namespace packages by creating proper __init__.py files.
"""

import os
import sys
import argparse
from pathlib import Path

# Define the namespace package content
NAMESPACE_INIT_CONTENT = '''"""
OMEN Platform namespace package.

This file allows Python to recognize 'omen' as a namespace package
that can span multiple directories.
"""

__path__ = __import__('pkgutil').extend_path(__path__, __name__)
'''

# Directory containing all packages
PACKAGES_DIR = Path('packages')

def setup_namespace_packages():
    """Set up all necessary namespace package files."""
    print("Setting up namespace packages...")
    package_count = 0
    
    # Ensure the omen directory exists in each package
    for package_dir in PACKAGES_DIR.glob('omen-*'):
        package_count += 1
        print(f"Processing package: {package_dir.name}")
        
        src_dir = package_dir / 'src'
        if not src_dir.exists():
            print(f"  Warning: {src_dir} does not exist. Skipping.")
            continue

        # Create or update namespace package __init__.py
        omen_dir = src_dir / 'omen'
        omen_dir.mkdir(exist_ok=True)
        
        init_file = omen_dir / '__init__.py'
        if not init_file.exists() or init_file.read_text() != NAMESPACE_INIT_CONTENT:
            print(f"  Creating/updating namespace __init__.py in {omen_dir}")
            init_file.write_text(NAMESPACE_INIT_CONTENT)
        else:
            print(f"  Namespace __init__.py already correct in {omen_dir}")
        
        # Create nested module directory if needed
        module_name = package_dir.name.split('-')[1]  # e.g., extract 'core' from 'omen-core'
        module_dir = omen_dir / module_name
        module_dir.mkdir(exist_ok=True)
        
        # Ensure module has __init__.py
        module_init = module_dir / '__init__.py'
        if not module_init.exists():
            print(f"  Creating module __init__.py in {module_dir}")
            module_init.write_text(f'"""\nOMEN {module_name.capitalize()} module.\n"""\n')
        else:
            print(f"  Module __init__.py already exists in {module_dir}")
    
    if package_count == 0:
        print("No packages found. Are you running this script from the correct directory?")
        return False
        
    print(f"Namespace packages setup completed for {package_count} packages.")
    return True

def uninstall_packages():
    """Uninstall all OMEN packages to prepare for reinstallation."""
    print("\nUninstalling packages...")
    result = os.system('pip3 uninstall -y omen-core omen-vectorstore omen-ontology omen-extractors omen-cli')
    return result == 0

def reinstall_packages():
    """Reinstall all OMEN packages in the correct order."""
    packages = [
        "omen-core",
        "omen-vectorstore",
        "omen-ontology",
        "omen-extractors[keboola]",
        "omen-cli",
    ]
    
    print("\nReinstalling packages...")
    success = True
    
    for package in packages:
        print(f"\nInstalling {package}...")
        result = os.system(f'pip3 install -e "packages/{package}"')
        if result != 0:
            print(f"Failed to install {package}")
            success = False
            break
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Setup OMEN namespace packages')
    parser.add_argument('--reinstall', action='store_true', help='Uninstall and reinstall packages')
    parser.add_argument('--setup-only', action='store_true', help='Only set up namespace packages without reinstalling')
    args = parser.parse_args()
    
    success = setup_namespace_packages()
    
    if not success:
        return 1
        
    if args.reinstall:
        if uninstall_packages() and reinstall_packages():
            print("\nAll packages have been reinstalled successfully.")
            print("Try running 'omen --help' to test.")
        else:
            print("\nPackage reinstallation failed.")
            return 1
    elif not args.setup_only:
        print("\nTo reinstall packages, run this script with the --reinstall flag.")
        print("Or run these commands manually:")
        print("pip3 uninstall -y omen-core omen-vectorstore omen-ontology omen-extractors omen-cli")
        print("pip3 install -e packages/omen-core")
        print("pip3 install -e packages/omen-vectorstore")
        print("pip3 install -e packages/omen-ontology")
        print('pip3 install -e "packages/omen-extractors[keboola]"')
        print("pip3 install -e packages/omen-cli")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 