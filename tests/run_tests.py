#!/usr/bin/env python3
"""
run_tests.py

Test runner script for the MLOps testing framework.

This script provides convenient ways to run different types of tests:
- Unit tests only
- Integration tests only
- All tests with coverage
- Specific test modules
- API tests (when available)

Usage:
    python tests/run_tests.py --unit                    # Run unit tests only
    python tests/run_tests.py --integration             # Run integration tests only
    python tests/run_tests.py --coverage                # Run all tests with coverage
    python tests/run_tests.py --module test_data_loader # Run specific module
    python tests/run_tests.py --all                     # Run all tests
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def run_unit_tests():
    """Run unit tests only."""
    cmd = [
        "python", "-m", "pytest", 
        "tests/", 
        "-m", "unit",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests only."""
    cmd = [
        "python", "-m", "pytest", 
        "tests/", 
        "-m", "integration",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Integration Tests")


def run_tests_with_coverage():
    """Run all tests with coverage reporting."""
    cmd = [
        "python", "-m", "pytest", 
        "tests/",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        "-v"
    ]
    return run_command(cmd, "Tests with Coverage")


def run_specific_module(module_name):
    """Run tests for a specific module."""
    cmd = [
        "python", "-m", "pytest", 
        f"tests/{module_name}.py",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, f"Module Tests: {module_name}")


def run_all_tests():
    """Run all tests."""
    cmd = [
        "python", "-m", "pytest", 
        "tests/",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "All Tests")


def run_code_quality_checks():
    """Run code quality checks."""
    print(f"\n{'='*60}")
    print("Running Code Quality Checks")
    print(f"{'='*60}\n")
    
    # Check if black is available
    try:
        subprocess.run(["black", "--version"], check=True, capture_output=True)
        print("✅ Black (code formatter) is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Black not found. Install with: pip install black")
    
    # Check if flake8 is available
    try:
        subprocess.run(["flake8", "--version"], check=True, capture_output=True)
        print("✅ Flake8 (linter) is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Flake8 not found. Install with: pip install flake8")
    
    # Run black check
    black_cmd = ["black", "--check", "src/", "tests/"]
    black_success = run_command(black_cmd, "Code Formatting Check (Black)")
    
    # Run flake8
    flake8_cmd = ["flake8", "src/", "tests/"]
    flake8_success = run_command(flake8_cmd, "Code Linting (Flake8)")
    
    return black_success and flake8_success


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Run MLOps testing framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_tests.py --unit                    # Run unit tests only
  python tests/run_tests.py --integration             # Run integration tests only
  python tests/run_tests.py --coverage                # Run all tests with coverage
  python tests/run_tests.py --module test_data_loader # Run specific module
  python tests/run_tests.py --all                     # Run all tests
  python tests/run_tests.py --quality                 # Run code quality checks
        """
    )
    
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run unit tests only"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run integration tests only"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run all tests with coverage reporting"
    )
    parser.add_argument(
        "--module", 
        type=str, 
        help="Run tests for specific module (e.g., test_data_loader)"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all tests"
    )
    parser.add_argument(
        "--quality", 
        action="store_true", 
        help="Run code quality checks"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Error: tests directory not found. Please run from project root.")
        sys.exit(1)
    
    # Check if pytest is available
    try:
        subprocess.run(["python", "-m", "pytest", "--version"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ Error: pytest not found. Install with: pip install pytest")
        sys.exit(1)
    
    success = True
    
    if args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.coverage:
        success = run_tests_with_coverage()
    elif args.module:
        success = run_specific_module(args.module)
    elif args.quality:
        success = run_code_quality_checks()
    elif args.all:
        success = run_all_tests()
    else:
        # Default: run all tests
        success = run_all_tests()
    
    if success:
        print(f"\n{'='*60}")
        print("🎉 All tests completed successfully!")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print("❌ Some tests failed!")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main() 