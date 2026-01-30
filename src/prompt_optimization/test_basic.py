#!/usr/bin/env python3
"""Basic test script to verify the agent and evaluation work.

Run this before optimization to ensure the setup is correct.

Usage:
    uv run python -m prompt_optimization.test_basic
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Check for required API key
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("\nTo run this example, you need to set your OpenAI API key:")
    print("  export OPENAI_API_KEY='your-api-key-here'")
    print("\nOr create a .env file with:")
    print("  OPENAI_API_KEY=your-api-key-here")
    sys.exit(1)

# Now import the rest
import logfire

from .evals import contact_dataset
from .task import ContactInfo, TaskInput, contact_agent, extract_contact_info

# Configure logfire
logfire.configure(
    send_to_logfire="if-token-present",
    environment="development",
    service_name="prompt-optimization-test",
)
logfire.instrument_pydantic_ai()


async def test_single_extraction():
    """Test a single extraction to verify the agent works."""
    print("Testing single extraction...")

    test_input = TaskInput(
        text="John Smith\njohn@example.com\n555-123-4567"
    )

    result = await extract_contact_info(test_input)
    print(f"Input: {test_input.text}")
    print(f"Output: {result}")
    print()

    assert result.name is not None, "Name should be extracted"
    assert result.email is not None, "Email should be extracted"
    print("Single extraction test passed!")


async def test_override_instructions():
    """Test that instruction override works."""
    print("\nTesting instruction override...")

    test_input = TaskInput(
        text="Contact: Jane Doe at jane@corp.com"
    )

    # Test with default instructions
    result1 = await extract_contact_info(test_input)
    print(f"Default instructions result: {result1}")

    # Test with overridden instructions
    custom_instructions = """Extract contact information precisely.
    Focus on finding: name, email, phone, company, and title.
    Return null for any field not found."""

    with contact_agent.override(instructions=custom_instructions):
        result2 = await extract_contact_info(test_input)
        print(f"Custom instructions result: {result2}")

    print("Instruction override test passed!")


async def test_evaluation():
    """Test running evaluation on the dataset."""
    print("\nTesting evaluation...")

    # Run evaluation on a subset
    subset_cases = contact_dataset.cases[:3]
    from pydantic_evals import Dataset

    test_dataset = Dataset(
        name="test_subset",
        cases=subset_cases,
        evaluators=contact_dataset.evaluators,
    )

    report = await test_dataset.evaluate(
        extract_contact_info,
        max_concurrency=2,
        progress=True,
    )

    print(f"\nEvaluation complete. Cases evaluated: {len(report.cases)}")
    for case_report in report.cases:
        name = case_report.name if hasattr(case_report, "name") else "unknown"
        scores = case_report.scores if hasattr(case_report, "scores") else {}
        print(f"  {name}: {scores}")

    print("Evaluation test passed!")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Running basic tests for prompt optimization example")
    print("=" * 60)

    await test_single_extraction()
    await test_override_instructions()
    await test_evaluation()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
