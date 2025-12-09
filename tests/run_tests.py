#!/usr/bin/env python3
"""
End-to-End Test Runner for CodebaseExplainer

This script runs the complete test suite and generates detailed output
including test results, coverage reports, and error details.

Usage:
    python tests/run_tests.py
    # or
    ./tests/run_tests.py
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import shutil


class TestRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_output_dir = self.project_root / "test_output"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.test_output_dir / f"run_{self.timestamp}"
        
        # Create output directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Output files
        self.results_file = self.run_dir / "test_results.txt"
        self.coverage_file = self.run_dir / "coverage_report.txt"
        self.junit_file = self.run_dir / "junit.xml"
        self.html_coverage_dir = self.run_dir / "htmlcov"
        self.summary_file = self.run_dir / "summary.json"
        
    def print_header(self, message):
        """Print a formatted header"""
        print("\n" + "=" * 80)
        print(f"  {message}")
        print("=" * 80 + "\n")
        
    def run_command(self, cmd, description):
        """Run a command and capture output"""
        self.print_header(description)
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            output = result.stdout + result.stderr
            print(output)
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'output': output
            }
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after 300 seconds"
            print(f"ERROR: {error_msg}")
            return {
                'success': False,
                'returncode': -1,
                'output': error_msg
            }
        except Exception as e:
            error_msg = f"Failed to run command: {e}"
            print(f"ERROR: {error_msg}")
            return {
                'success': False,
                'returncode': -1,
                'output': error_msg
            }
    
    def run_pytest(self):
        """Run pytest with coverage and detailed output"""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",  # Verbose output
            "--tb=long",  # Long traceback format
            "--cov=src/codehierarchy",  # Coverage for source code
            "--cov-report=term-missing",  # Show missing lines
            "--cov-report=html:" + str(self.html_coverage_dir),
            f"--junitxml={self.junit_file}",
            "--color=yes"
        ]
        
        result = self.run_command(cmd, "Running Test Suite with Coverage")
        
        # Save detailed results
        with open(self.results_file, 'w') as f:
            f.write(result['output'])
        
        return result
    
    def run_flake8(self):
        """Run flake8 linting"""
        cmd = [
            sys.executable, "-m", "flake8",
            "src/", "tests/",
            "--count",
            "--statistics",
            "--show-source"
        ]
        
        result = self.run_command(cmd, "Running Flake8 Linting")
        
        # Save flake8 results
        flake8_file = self.run_dir / "flake8_report.txt"
        with open(flake8_file, 'w') as f:
            f.write(result['output'])
        
        return result
    
    def run_mypy(self):
        """Run mypy type checking"""
        cmd = [
            sys.executable, "-m", "mypy",
            "src/", "tests/",
            "--show-error-codes",
            "--pretty"
        ]
        
        result = self.run_command(cmd, "Running MyPy Type Checking")
        
        # Save mypy results
        mypy_file = self.run_dir / "mypy_report.txt"
        with open(mypy_file, 'w') as f:
            f.write(result['output'])
        
        return result
    
    def generate_summary(self, pytest_result, flake8_result, mypy_result):
        """Generate a summary of all test results"""
        summary = {
            'timestamp': self.timestamp,
            'run_directory': str(self.run_dir),
            'pytest': {
                'success': pytest_result['success'],
                'returncode': pytest_result['returncode']
            },
            'flake8': {
                'success': flake8_result['success'],
                'returncode': flake8_result['returncode']
            },
            'mypy': {
                'success': mypy_result['success'],
                'returncode': mypy_result['returncode']
            },
            'overall_success': all([
                pytest_result['success'],
                flake8_result['success'],
                mypy_result['success']
            ])
        }
        
        # Save JSON summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create human-readable summary
        self.print_header("TEST RUN SUMMARY")
        
        print(f"Timestamp: {self.timestamp}")
        print(f"Output Directory: {self.run_dir}\n")
        
        print("Results:")
        print(f"  ✓ PyTest:  {'PASSED' if pytest_result['success'] else 'FAILED'}")
        print(f"  ✓ Flake8:  {'PASSED' if flake8_result['success'] else 'FAILED'}")
        print(f"  ✓ MyPy:    {'PASSED' if mypy_result['success'] else 'FAILED'}")
        
        print(f"\nOverall: {'✓ ALL TESTS PASSED' if summary['overall_success'] else '✗ SOME TESTS FAILED'}")
        
        print("\nGenerated Files:")
        print(f"  - Test Results:     {self.results_file}")
        print(f"  - Coverage Report:  {self.html_coverage_dir}/index.html")
        print(f"  - JUnit XML:        {self.junit_file}")
        print(f"  - Flake8 Report:    {self.run_dir}/flake8_report.txt")
        print(f"  - MyPy Report:      {self.run_dir}/mypy_report.txt")
        print(f"  - Summary JSON:     {self.summary_file}")
        
        # Create a latest symlink for convenience
        latest_link = self.test_output_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        try:
            latest_link.symlink_to(self.run_dir.name)
            print(f"\nLatest results also available at: {latest_link}")
        except Exception:
            pass  # Symlinks might not work on all systems
        
        return summary
    
    def run_all(self):
        """Run all tests and generate reports"""
        self.print_header(f"CodebaseExplainer Test Suite - {self.timestamp}")
        
        print(f"Project Root: {self.project_root}")
        print(f"Output Directory: {self.run_dir}\n")
        
        # Run all test suites
        pytest_result = self.run_pytest()
        flake8_result = self.run_flake8()
        mypy_result = self.run_mypy()
        
        # Generate summary
        summary = self.generate_summary(pytest_result, flake8_result, mypy_result)
        
        # Return exit code based on overall success
        return 0 if summary['overall_success'] else 1


def main():
    """Main entry point"""
    runner = TestRunner()
    exit_code = runner.run_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
