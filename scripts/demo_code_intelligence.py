#!/usr/bin/env python3
"""
Demo Script for Aragora Code Intelligence Platform.

Demonstrates:
1. Codebase indexing and understanding
2. Security vulnerability scanning
3. Bug pattern detection
4. Code review with multi-agent debate
5. Test generation

Usage:
    python scripts/demo_code_intelligence.py [path_to_codebase]

Examples:
    # Analyze the Aragora codebase itself
    python scripts/demo_code_intelligence.py .

    # Analyze a specific directory
    python scripts/demo_code_intelligence.py /path/to/project
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}\n")


async def demo_codebase_understanding(path: str) -> None:
    """Demonstrate codebase understanding capabilities."""
    print_header("CODEBASE UNDERSTANDING")

    try:
        from aragora.agents.codebase_agent import CodebaseUnderstandingAgent
    except ImportError:
        print("CodebaseUnderstandingAgent not available. Skipping...")
        return

    agent = CodebaseUnderstandingAgent(
        root_path=path,
        enable_debate=False,  # Faster without debate for demo
    )

    # Index the codebase
    print_subheader("Indexing Codebase")
    index = await agent.index_codebase()
    print(f"Total files: {index.total_files}")
    print(f"Total lines: {index.total_lines}")
    print(f"Languages: {dict(index.languages)}")
    print(f"Classes: {len(index.symbols.get('classes', []))}")
    print(f"Functions: {len(index.symbols.get('functions', []))}")

    # Ask a question about the codebase
    print_subheader("Understanding Query")
    question = "What are the main modules and their purposes?"
    print(f"Question: {question}")
    understanding = await agent.understand(question, max_files=5)
    print(f"Answer: {understanding.answer[:500]}...")
    print(f"Confidence: {understanding.confidence:.1%}")
    print(f"Relevant files: {understanding.relevant_files[:5]}")


async def demo_security_scan(path: str) -> None:
    """Demonstrate security vulnerability scanning."""
    print_header("SECURITY VULNERABILITY SCAN")

    try:
        from aragora.audit.security_scanner import SecurityScanner
    except ImportError:
        print("SecurityScanner not available. Showing sample scan...")
        # Show sample output
        print("Sample security findings:")
        print("- [HIGH] Hardcoded secret detected in config.py:42")
        print("- [MEDIUM] SQL injection risk in database.py:156")
        print("- [LOW] Debug mode enabled in settings.py:12")
        return

    scanner = SecurityScanner()

    print("Scanning for vulnerabilities...")
    report = scanner.scan_directory(
        path,
        exclude_patterns=["__pycache__", ".git", "node_modules", ".venv"],
    )

    print(f"\nFiles scanned: {report.files_scanned}")
    print(f"Lines scanned: {report.lines_scanned}")
    print(f"Total findings: {len(report.findings)}")

    # Show top findings
    print_subheader("Top Security Findings")
    for finding in report.findings[:5]:
        severity = (
            finding.severity.value if hasattr(finding.severity, "value") else finding.severity
        )
        print(f"[{severity.upper()}] {finding.title}")
        print(f"  File: {finding.file_path}:{finding.line_number}")
        if finding.recommendation:
            print(f"  Fix: {finding.recommendation[:100]}...")
        print()


async def demo_bug_detection(path: str) -> None:
    """Demonstrate bug pattern detection."""
    print_header("BUG PATTERN DETECTION")

    try:
        from aragora.audit.bug_detector import BugDetector
    except ImportError:
        print("BugDetector not available. Showing sample detection...")
        print("Sample bug patterns detected:")
        print("- [WARNING] Resource leak: File opened but not closed in utils.py:78")
        print("- [WARNING] Potential null reference in handler.py:234")
        print("- [INFO] Empty except block in processor.py:156")
        return

    detector = BugDetector()

    print("Detecting bug patterns...")
    report = detector.detect_in_directory(
        path,
        exclude_patterns=["__pycache__", ".git", "node_modules", ".venv"],
    )

    print(f"\nFiles scanned: {report.files_scanned}")
    print(f"Total bugs found: {len(report.bugs)}")

    # Show top bugs
    print_subheader("Top Bug Patterns")
    for bug in report.bugs[:5]:
        severity = bug.severity.value if hasattr(bug.severity, "value") else bug.severity
        print(f"[{severity.upper()}] {bug.title}")
        print(f"  File: {bug.file_path}:{bug.line_number}")
        if bug.description:
            print(f"  Details: {bug.description[:100]}...")
        print()


async def demo_code_review(sample_code: Optional[str] = None) -> None:
    """Demonstrate code review capabilities."""
    print_header("CODE REVIEW")

    try:
        from aragora.agents.code_reviewer import CodeReviewOrchestrator, ReviewCategory
    except ImportError:
        print("CodeReviewOrchestrator not available. Showing sample review...")
        print("Sample review findings:")
        print("- [SECURITY] SQL injection vulnerability in query construction")
        print("- [PERFORMANCE] Inefficient nested loop O(n^2) complexity")
        print("- [MAINTAINABILITY] Function too long (>50 lines)")
        return

    # Sample code to review
    if sample_code is None:
        sample_code = '''
import os
import subprocess

API_KEY = "EXAMPLE_HARDCODED_SECRET"  # Hardcoded secret (demo)

def execute_command(user_input):
    """Execute a shell command."""
    result = subprocess.run(user_input, shell=True, capture_output=True)
    return result.stdout

def get_user_data(user_id):
    """Get user data from database."""
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute_query(query)

def process_items(items):
    """Process items inefficiently."""
    result = ""
    for item in items:
        result = result + str(item)  # String concatenation in loop
    return result
'''

    print("Reviewing sample code for issues...")
    print_subheader("Sample Code")
    for i, line in enumerate(sample_code.strip().split("\n")[:15], 1):
        print(f"{i:3}: {line}")
    if len(sample_code.split("\n")) > 15:
        print("     ...")

    orchestrator = CodeReviewOrchestrator()
    result = await orchestrator.review_code(
        code=sample_code,
        file_path="sample.py",
    )

    print_subheader("Review Findings")
    for finding in result.findings:
        category = (
            finding.category.value if hasattr(finding.category, "value") else finding.category
        )
        severity = (
            finding.severity.value if hasattr(finding.severity, "value") else finding.severity
        )
        print(f"[{category.upper()}] [{severity.upper()}] {finding.title}")
        if finding.description:
            print(f"  {finding.description[:100]}...")
        if finding.suggestion:
            print(f"  Fix: {finding.suggestion[:100]}...")
        print()

    print(f"\nTotal findings: {len(result.findings)}")


async def demo_test_generation() -> None:
    """Demonstrate test generation capabilities."""
    print_header("TEST GENERATION")

    try:
        from aragora.coding.test_generator import TestGenerator, TestFramework
    except ImportError:
        print("TestGenerator not available. Showing sample generation...")
        print("Sample generated test:")
        print('''
def test_calculate_total_happy_path():
    """Test calculate_total with valid inputs."""
    result = calculate_total([10, 20, 30])
    assert result == 60

def test_calculate_total_empty_list():
    """Test calculate_total with empty list."""
    result = calculate_total([])
    assert result == 0
''')
        return

    # Sample code to generate tests for
    sample_code = '''
def calculate_total(items: list[float]) -> float:
    """Calculate the total of a list of numbers.

    Args:
        items: List of numbers to sum

    Returns:
        Sum of all numbers

    Raises:
        ValueError: If items contains non-numeric values
    """
    if not items:
        return 0.0
    total = 0.0
    for item in items:
        if not isinstance(item, (int, float)):
            raise ValueError(f"Invalid item: {item}")
        total += item
    return total
'''

    print("Analyzing function for test generation...")
    print_subheader("Source Function")
    for i, line in enumerate(sample_code.strip().split("\n"), 1):
        print(f"{i:3}: {line}")

    generator = TestGenerator(framework=TestFramework.PYTEST)
    analysis = generator.analyze_function(sample_code, "calculate_total")

    print_subheader("Function Analysis")
    print(f"Parameters: {analysis.get('parameters', [])}")
    print(f"Return type: {analysis.get('return_type', 'unknown')}")
    print(f"Branches: {analysis.get('branches', 0)}")
    print(f"Raises: {analysis.get('raises', [])}")
    print(f"Complexity: {analysis.get('complexity', 1)}")

    test_cases = generator.generate_test_cases("calculate_total", analysis)

    print_subheader("Generated Test Cases")
    for tc in test_cases[:5]:
        print(f"- {tc.name}")
        print(f"  Type: {tc.test_type.value}")
        print(f"  Description: {tc.description}")
        if tc.inputs:
            print(f"  Inputs: {tc.inputs}")
        print()

    print(f"\nTotal test cases generated: {len(test_cases)}")


async def demo_audit(path: str) -> None:
    """Demonstrate comprehensive codebase audit."""
    print_header("COMPREHENSIVE AUDIT")

    try:
        from aragora.agents.codebase_agent import CodebaseUnderstandingAgent
    except ImportError:
        print("CodebaseUnderstandingAgent not available. Showing sample audit...")
        print("Sample audit results:")
        print("- Risk Score: 35/100")
        print("- Security findings: 3 (1 high, 2 medium)")
        print("- Bug findings: 5 (2 medium, 3 low)")
        print("- Quality issues: 8")
        return

    agent = CodebaseUnderstandingAgent(
        root_path=path,
        enable_debate=False,
    )

    print("Running comprehensive audit...")
    result = await agent.audit(
        include_dead_code=False,  # Skip for faster demo
        include_quality=True,
    )

    print_subheader("Audit Summary")
    print(f"Scan ID: {result.scan_id}")
    print(f"Files analyzed: {result.files_analyzed}")
    print(f"Lines analyzed: {result.lines_analyzed}")
    print(f"Risk score: {result.risk_score:.1f}/100")
    print(f"\nSecurity findings: {len(result.security_findings)}")
    print(f"Bug findings: {len(result.bug_findings)}")
    print(f"Quality issues: {len(result.quality_issues)}")

    if result.prioritized_remediations:
        print_subheader("Priority Remediations")
        for rem in result.prioritized_remediations[:5]:
            print(f"{rem['priority']}. [{rem['type'].upper()}] {rem['title']}")
            print(f"   Urgency: {rem['urgency']}")

    print_subheader("Agent Summary")
    print(result.agent_summary or "No summary generated.")


async def main(path: str, demos: Optional[list] = None) -> None:
    """Run the code intelligence demo."""
    start_time = datetime.now()

    print("\n" + "=" * 60)
    print(" ARAGORA CODE INTELLIGENCE DEMO")
    print("=" * 60)
    print(f"\nAnalyzing: {path}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_demos = ["understand", "security", "bugs", "review", "tests", "audit"]
    demos = demos or all_demos

    if "understand" in demos:
        await demo_codebase_understanding(path)

    if "security" in demos:
        await demo_security_scan(path)

    if "bugs" in demos:
        await demo_bug_detection(path)

    if "review" in demos:
        await demo_code_review()

    if "tests" in demos:
        await demo_test_generation()

    if "audit" in demos:
        await demo_audit(path)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print_header("DEMO COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Demos run: {', '.join(demos)}")
    print("\nFor more information, see docs/CODE_INTELLIGENCE_GUIDE.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo script for Aragora Code Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full demo on current directory
    python scripts/demo_code_intelligence.py .

    # Only security and bug detection
    python scripts/demo_code_intelligence.py . --demos security bugs

    # Analyze a specific project
    python scripts/demo_code_intelligence.py /path/to/project
        """,
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to codebase to analyze (default: current directory)",
    )
    parser.add_argument(
        "--demos",
        nargs="+",
        choices=["understand", "security", "bugs", "review", "tests", "audit"],
        help="Specific demos to run (default: all)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate path
    path = Path(args.path).resolve()
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    try:
        asyncio.run(main(str(path), args.demos))
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
