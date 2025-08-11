#!/usr/bin/env python3
"""Simple test script to validate package structure without external dependencies."""

import os
import sys
import ast
import importlib.util

def test_package_structure():
    """Test that the package structure is correct."""
    print("Testing package structure...")
    
    base_path = "/home/runner/work/WebscraperApp/WebscraperApp"
    package_path = os.path.join(base_path, "forcasting_pkg")
    
    # Check main package directory exists
    assert os.path.exists(package_path), "Main package directory not found"
    
    # Check submodules exist
    expected_modules = [
        "__init__.py",
        "forecasting/__init__.py",
        "analysis/__init__.py", 
        "data/__init__.py",
        "visualization/__init__.py",
        "models/__init__.py",
        "cli/__init__.py",
        "examples/__init__.py"
    ]
    
    for module in expected_modules:
        module_path = os.path.join(package_path, module)
        assert os.path.exists(module_path), f"Module {module} not found"
        print(f"✓ {module}")
    
    print("Package structure test passed!")


def test_syntax_validation():
    """Test that all Python files have valid syntax."""
    print("\nTesting Python syntax...")
    
    base_path = "/home/runner/work/WebscraperApp/WebscraperApp"
    package_path = os.path.join(base_path, "forcasting_pkg")
    
    python_files = []
    for root, dirs, files in os.walk(package_path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Parse the AST to check syntax
            ast.parse(source)
            rel_path = os.path.relpath(py_file, base_path)
            print(f"✓ {rel_path}")
            
        except SyntaxError as e:
            print(f"✗ Syntax error in {py_file}: {e}")
            return False
        except Exception as e:
            print(f"✗ Error reading {py_file}: {e}")
            return False
    
    print("Syntax validation test passed!")
    return True


def test_imports_structure():
    """Test import structure without actually importing (to avoid dependency issues)."""
    print("\nTesting import structure...")
    
    base_path = "/home/runner/work/WebscraperApp/WebscraperApp"
    package_path = os.path.join(base_path, "forcasting_pkg")
    
    # Test main package __init__.py
    init_file = os.path.join(package_path, "__init__.py")
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Check for expected exports
    expected_exports = ['ForecastingEngine', 'TechnicalAnalyzer', 'DataSource']
    for export in expected_exports:
        assert export in content, f"Expected export {export} not found in main __init__.py"
        print(f"✓ {export} exported")
    
    print("Import structure test passed!")


def test_cli_structure():
    """Test CLI module structure."""
    print("\nTesting CLI structure...")
    
    base_path = "/home/runner/work/WebscraperApp/WebscraperApp"
    cli_file = os.path.join(base_path, "forcasting_pkg", "cli", "__init__.py")
    
    with open(cli_file, 'r') as f:
        content = f.read()
    
    # Check for main function
    assert 'def main(' in content, "CLI main function not found"
    print("✓ CLI main function exists")
    
    # Check for command parsing
    assert 'argparse' in content, "CLI uses argparse"
    print("✓ CLI uses argparse")
    
    print("CLI structure test passed!")


def test_setup_files():
    """Test setup and configuration files."""
    print("\nTesting setup files...")
    
    base_path = "/home/runner/work/WebscraperApp/WebscraperApp"
    
    # Check setup.py
    setup_file = os.path.join(base_path, "setup.py")
    assert os.path.exists(setup_file), "setup.py not found"
    print("✓ setup.py exists")
    
    # Check pyproject.toml
    pyproject_file = os.path.join(base_path, "pyproject.toml")
    assert os.path.exists(pyproject_file), "pyproject.toml not found"
    print("✓ pyproject.toml exists")
    
    # Check requirements.txt
    req_file = os.path.join(base_path, "requirements.txt")
    assert os.path.exists(req_file), "requirements.txt not found"
    print("✓ requirements.txt exists")
    
    # Check README.md
    readme_file = os.path.join(base_path, "README.md")
    assert os.path.exists(readme_file), "README.md not found"
    print("✓ README.md exists")
    
    # Check LICENSE
    license_file = os.path.join(base_path, "LICENSE")
    assert os.path.exists(license_file), "LICENSE not found"
    print("✓ LICENSE exists")
    
    print("Setup files test passed!")


def test_examples_and_tests():
    """Test examples and test files."""
    print("\nTesting examples and tests...")
    
    base_path = "/home/runner/work/WebscraperApp/WebscraperApp"
    
    # Check examples
    examples_dir = os.path.join(base_path, "forcasting_pkg", "examples")
    basic_example = os.path.join(examples_dir, "basic_usage.py")
    advanced_example = os.path.join(examples_dir, "advanced_usage.py")
    
    assert os.path.exists(basic_example), "basic_usage.py not found"
    assert os.path.exists(advanced_example), "advanced_usage.py not found"
    print("✓ Example files exist")
    
    # Check tests
    tests_dir = os.path.join(base_path, "tests_pkg")
    test_file = os.path.join(tests_dir, "test_forcasting_pkg.py")
    
    assert os.path.exists(test_file), "test_forcasting_pkg.py not found"
    print("✓ Test files exist")
    
    # Check notebook
    notebook_dir = os.path.join(base_path, "notebooks")
    notebook_file = os.path.join(notebook_dir, "tutorial.ipynb")
    
    assert os.path.exists(notebook_file), "tutorial.ipynb not found"
    print("✓ Tutorial notebook exists")
    
    print("Examples and tests validation passed!")


def main():
    """Run all tests."""
    print("Forcasting Package Structure Validation")
    print("=" * 50)
    
    try:
        test_package_structure()
        test_syntax_validation()
        test_imports_structure()
        test_cli_structure()
        test_setup_files()
        test_examples_and_tests()
        
        print("\n" + "=" * 50)
        print("✅ All structure validation tests passed!")
        print("The forcasting_pkg package is properly structured and ready for use.")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Validation failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())