# Automated Prompt Optimization with GEPA, Pydantic AI, and Pydantic Evals

## TL;DR

- **Prompt engineering is tedious**: Manual iteration on prompts is time-consuming and inconsistent
- **GEPA automates the process**: Uses evolutionary algorithms to systematically improve prompts based on evaluation feedback
- **Integration with pydantic-ai**: We use `Agent.override()` to inject candidate prompts during optimization without modifying agent definitions
- **pydantic-evals provides the evaluation harness**: Parallel execution, rich metrics, and OpenTelemetry tracing make evaluation fast and observable
- **The result**: Turn prompt optimization from art into science—let the algorithm explore the prompt space while you define success criteria

---

## Introduction: The Prompt Engineering Problem

If you've built applications with large language models, you've experienced the prompt engineering cycle: write a prompt, test it on a few examples, notice it fails on edge cases, tweak the wording, repeat. This process has several fundamental problems:

**It's time-consuming.** Each iteration requires manual review of outputs and careful consideration of what changes might help. A single prompt might go through dozens of revisions.

**It's inconsistent.** Human judgment varies. The "improvement" you make at 3pm after coffee might look different than the one you'd make at 6pm when tired. There's no systematic exploration of the prompt space.

**It's limited in scope.** You can only test so many variations. The space of possible prompts is vast, and manual exploration covers only a tiny fraction.

**It doesn't scale.** When you have multiple prompts across different agents, keeping them all optimized becomes a maintenance burden.

What if we could automate this process? Instead of manually iterating, we define:
1. A task with clear success criteria
2. A dataset of test cases with expected outputs
3. An evaluation function that scores how well the agent performs

Then let an algorithm systematically explore prompt variations to maximize that score.

This is exactly what GEPA (Genetic-Pareto Prompt Evolution) does. In this repository, we demonstrate a complete prompt optimization pipeline using GEPA with pydantic-ai and pydantic-evals, showing how to turn prompt engineering from an art into a science.

---

## Quick Start

```bash
# Clone this repository
git clone https://github.com/dmontagu/prompt-optimization-example
cd prompt-optimization-example

# Set up your API key
export OPENAI_API_KEY='your-key'
# Or create a .env file:
# cp .env.example .env
# Then edit .env with your key

# Install dependencies
uv sync

# Run evaluation with initial instructions
cd src
uv run python -m prompt_optimization.run_optimization eval

# Compare initial vs expert instructions
uv run python -m prompt_optimization.run_optimization compare

# Run optimization
uv run python -m prompt_optimization.run_optimization optimize --max-calls 50
```

---

## Understanding GEPA: Evolution for Prompts

### What is GEPA?

GEPA applies evolutionary algorithms to prompt optimization. If you're familiar with genetic algorithms, the concept will feel familiar. If not, here's the intuition:

Imagine you're trying to breed the fastest racing pigeons. You start with a population of pigeons, race them, keep the fastest ones, breed them together (mixing their genetics), and occasionally introduce random mutations. Over generations, the population gets faster.

GEPA does the same thing with prompts across multiple modules:

1. **Start with seed prompts** for each "module" (e.g., agent) in your system (your initial "pigeons")
2. **Evaluate the system** against a dataset to get fitness scores
3. **Generate variations** by having an LLM propose improvements based on what went wrong
4. **Crossover**: Combine successful prompt variants from different modules to create new candidates
5. **Keep the best performers** and discard the rest (Pareto selection)
6. **Repeat** until convergence or budget exhaustion

The key insight is that GEPA optimizes *multiple prompts together*. When you have a system with several agents or modules, their prompts can interact in complex ways. GEPA's genetic crossover mixes successful prompt variants across modules, exploring combinations that manual iteration would never find. The "mutation" step isn't random either—it's guided by an LLM that analyzes failures and proposes targeted improvements.

### The GEPA Algorithm in Detail

Here's how a single optimization iteration works:

```
┌─────────────────────────────────────────────────────────────────┐
│                     GEPA Optimization Loop                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. EVALUATE: Run current candidate on training batch           │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│     │ Case 1   │    │ Case 2   │    │ Case N   │                │
│     │ Score:0.8│    │ Score:1.0│    │ Score:0.6│                │
│     └──────────┘    └──────────┘    └──────────┘                │
│                           │                                     │
│                           ▼                                     │
│  2. REFLECT: Build dataset of what went wrong                   │
│     ┌────────────────────────────────────────┐                  │
│     │ Input: "John at john@example.com"      │                  │
│     │ Expected: name="John", email="..."     │                  │
│     │ Actual: name=None, email="..."         │                  │
│     │ Feedback: "Missed extracting name"     │                  │
│     └────────────────────────────────────────┘                  │
│                           │                                     │
│                           ▼                                     │
│  3. PROPOSE: LLM generates improved prompt                      │
│     ┌────────────────────────────────────────┐                  │
│     │ "Based on the failures, the prompt     │                  │
│     │  should explicitly mention looking     │                  │
│     │  for names before email addresses..."  │                  │
│     └────────────────────────────────────────┘                  │
│                           │                                     │
│                           ▼                                     │
│  4. ACCEPT/REJECT: Compare new vs old on subsample              │
│     - If new is better: accept and continue                     │
│     - If not: keep old, try different mutation                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The algorithm maintains a "population" of candidate prompts (though in the example we'll show below, we're optimizing a single prompt, so it's a population of one). Each iteration:

1. **Samples a mini-batch** from the training set
2. **Evaluates the current best candidate** on that batch
3. **Captures trajectories**—detailed information about what happened during execution
4. **Builds a reflective dataset**—structured feedback about failures
5. **Proposes mutations**—uses an LLM to suggest improvements based on the feedback
6. **Accepts or rejects** the mutation based on whether it improves the score

This is what makes GEPA different from random search: the mutations are intelligent. The proposer LLM sees exactly what went wrong and can make targeted fixes.

### Why This Works Better Than Manual Iteration

Consider what happens when you manually iterate on prompts:

- You test on a few examples (maybe 5-10)
- You notice patterns in failures
- You make a change based on intuition
- You hope it doesn't break other cases

With GEPA:

- You test on the full dataset (or statistically significant samples)
- The algorithm systematically captures all failure patterns
- Changes are proposed by analyzing the actual error distribution
- Regression is caught automatically by the acceptance test

The benefits are threefold:

1. **Automation**: The process runs unattended—you can kick off an optimization and come back to improved prompts. Manual prompt iteration is attention-consuming; this frees you up.
2. **Efficiency**: The genetic Pareto approach evaluates candidates on mini-batches and subsamples, letting you explore many more prompt variations than you could afford if you ran every candidate against the full dataset.
3. **Fresh perspective**: The LLM proposer analyzes all failure cases together and suggests improvements without anchoring on your previous attempts. It might notice patterns you'd miss or try phrasings you wouldn't think of.

---

## Our Example: Contact Information Extraction

To demonstrate GEPA in action, we've built an agent that extracts contact information from unstructured text. This is a simple but realistic task—think of processing email signatures, LinkedIn messages, or business card scans.

### The Task Definition

We want to extract five fields from text:
- **Name**: The person's full name
- **Email**: Their email address
- **Phone**: Phone number in any format
- **Company**: Organization or company name
- **Title**: Job title or role

Here's our Pydantic model for the output:

```python
from pydantic import BaseModel, Field

class ContactInfo(BaseModel):
    """Extracted contact information from text."""

    name: str | None = Field(default=None, description="The person's full name")
    email: str | None = Field(default=None, description="Email address")
    phone: str | None = Field(default=None, description="Phone number")
    company: str | None = Field(default=None, description="Company or organization name")
    title: str | None = Field(default=None, description="Job title or role")
```

### The Agent

Using pydantic-ai, we create an agent with minimal initial instructions:

```python
from pydantic_ai import Agent

contact_agent = Agent(
    "openai:gpt-4o-mini",
    output_type=ContactInfo,
    instructions="Extract contact information from the provided text.",
    defer_model_check=True,  # Defer validation until runtime
)
```

The `defer_model_check=True` is just there so we can define the agent at module load time without requiring API credentials. The model is only instantiated when we actually run the agent.

### The Task Function

We wrap the agent in a simple async function that pydantic-evals will call:

```python
from dataclasses import dataclass

@dataclass
class TaskInput:
    """Input to the contact extraction task."""
    text: str

async def extract_contact_info(input: TaskInput) -> ContactInfo:
    """Run the contact extraction agent on the input text."""
    result = await contact_agent.run(input.text)
    return result.output
```

This separation of agent from task function is intentional—it allows us to modify how the agent is called (including overriding its instructions) without changing the task signature.

---

## Building the Evaluation Dataset

A good evaluation dataset is crucial for optimization. It should cover:
- **Happy path cases**: Standard, well-formatted inputs
- **Edge cases**: Unusual formats, missing information
- **Adversarial cases**: Noisy text, multiple contacts, ambiguous data

### Test Case Design

Here are some of our test cases, ranging from easy to hard:

```python
from pydantic_evals import Case, Dataset

# Easy: Standard email signature
Case(
    name="simple_email_signature",
    inputs=TaskInput(
        text="Best regards,\nJohn Smith\njohn.smith@example.com\n555-123-4567"
    ),
    expected_output=ContactInfo(
        name="John Smith",
        email="john.smith@example.com",
        phone="555-123-4567",
    ),
)

# Medium: Contact info embedded in prose
Case(
    name="inline_contact",
    inputs=TaskInput(
        text="For more information, contact Michael Johnson at michael.j@techcorp.io or call 1-800-TECH-123."
    ),
    expected_output=ContactInfo(
        name="Michael Johnson",
        email="michael.j@techcorp.io",
        phone="1-800-TECH-123",
    ),
)

# Hard: Noisy email thread with multiple contacts
Case(
    name="noisy_email_thread",
    inputs=TaskInput(
        text="""Re: Meeting Follow-up

Hi team,

Thanks for joining today's call. Please reach out to our new vendor contact:

Robert Chen
VP of Sales, CloudTech Solutions
r.chen@cloudtech.solutions
Mobile: +1 (650) 555-8900

He'll handle our account. The previous contact (sarah@oldvendor.com) is no longer available.

Best,
Alex"""
    ),
    expected_output=ContactInfo(
        name="Robert Chen",
        email="r.chen@cloudtech.solutions",
        phone="+1 (650) 555-8900",
        company="CloudTech Solutions",
        title="VP of Sales",
    ),
)
```

The hard cases test whether the agent can:
- Identify the *primary* contact when multiple are mentioned
- Ignore noise like sender signatures and old contacts
- Handle various phone number formats

### The Evaluator

We need a way to score how well the agent did. Our `FieldAccuracyEvaluator` computes per-field accuracy:

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

@dataclass
class FieldAccuracyEvaluator(Evaluator[TaskInput, ContactInfo, ContactCaseMetadata]):
    """Evaluates how many fields were correctly extracted."""

    def evaluate(self, ctx: EvaluatorContext) -> dict:
        expected = ctx.expected_output
        output = ctx.output

        fields = ["name", "email", "phone", "company", "title"]
        correct = 0
        total = 0

        for field in fields:
            expected_val = getattr(expected, field)
            output_val = getattr(output, field)

            # Only count fields that have expected values
            if expected_val is not None:
                total += 1
                # Flexible matching: case-insensitive, handles substrings
                expected_norm = str(expected_val).lower().strip()
                output_norm = str(output_val).lower().strip() if output_val else ""

                if expected_norm == output_norm or expected_norm in output_norm:
                    correct += 1

        accuracy = correct / total if total > 0 else 1.0

        return {
            "accuracy": accuracy,
            "fields_correct": correct,
            "fields_total": total,
        }
```

This evaluator:
- Only scores fields that have expected values (handles partial extraction cases)
- Uses flexible matching to handle minor formatting differences
- Returns multiple metrics for debugging

---

## The GEPA Adapter: Bridging Pydantic-Evals and GEPA

This is where the magic happens. The GEPA adapter is the integration layer that:
1. Takes candidate prompts from GEPA
2. Injects them into the agent using `Agent.override()`
3. Runs evaluation using pydantic-evals
4. Returns scores and feedback to GEPA

### Why We Built a Custom Adapter

GEPA provides a `GEPAAdapter` protocol that defines three methods:

1. **`evaluate()`**: Run the candidate on a batch of examples, return scores
2. **`make_reflective_dataset()`**: Build feedback data from evaluation results
3. **`propose_new_texts()`** (optional): Generate improved candidates

We could have implemented this by just calling the agent directly, but using pydantic-evals provides significant benefits:

**Parallel Evaluation**: pydantic-evals runs test cases concurrently. With 8 test cases and `max_concurrency=5`, we evaluate 5 cases in parallel, dramatically speeding up optimization.

**Rich Metrics**: The evaluation report includes not just scores but detailed per-case breakdowns, making debugging easy.

**OpenTelemetry Integration**: Every evaluation run is traced and can be viewed in Logfire, showing exactly what happened in each LLM call.

**Structured Results**: The `ReportCase` objects contain all the information we need for reflection.

### The Adapter Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     EvalsGEPAAdapter                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GEPA calls evaluate(batch, candidate)                          │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────┐               │
│  │  1. Parse candidate["instructions"]          │               │
│  │     (JSON string → actual instructions)      │               │
│  └──────────────────────────────────────────────┘               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────┐               │
│  │  2. with agent.override(instructions=...):   │               │
│  │       - Creates context with new instructions│               │
│  │       - Uses ContextVars (thread-safe)       │               │
│  └──────────────────────────────────────────────┘               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────┐                │
│  │  3. dataset.evaluate(task)                  │                │
│  │       - Runs cases in parallel              │                │
│  │       - Applies evaluators                  │                │
│  │       - Captures traces                     │                │
│  └─────────────────────────────────────────────┘                │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────┐                │
│  │  4. Extract scores from report              │                │
│  │     - Uses score_key (e.g., "accuracy")     │                │
│  │     - Builds trajectories for reflection    │                │
│  └─────────────────────────────────────────────┘                │
│         │                                                       │
│         ▼                                                       │
│  Return EvaluationBatch(outputs, scores, trajectories)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Using `Agent.override()` for instruction injection**: We could just create the agent from scratch each time, but this is a little cleaner. The override context manager temporarily replaces the agent's instructions using context variables, making it thread-safe and avoiding any side effects.

```python
with self.agent.override(instructions=instructions):
    report = await temp_dataset.evaluate(
        self.task,
        max_concurrency=self.max_concurrency,
    )
```

**JSON serialization for candidates**: GEPA expects candidates as `dict[str, str]`. We serialize instructions to JSON and deserialize when evaluating. We don't need to do this now, but this approach allows for future extension to structured prompt components. (More to come on that soon!)

**Reusing dataset evaluators**: Instead of implementing scoring logic in the adapter, we reuse the evaluators from the dataset. This keeps scoring logic in one place and ensures the adapter is generic.

### The Proposer Agent

When GEPA needs to generate improved prompts, our adapter uses a pydantic-ai agent:

```python
self._proposer_agent = Agent(
    self.proposer_model,
    output_type=str,
    instructions="""You are an expert prompt engineer. Your task is to improve
system prompts for AI agents based on evaluation feedback.

You will receive:
1. The current instructions being used
2. Examples of inputs, outputs, and feedback from evaluation

Analyze what went wrong and propose improved instructions that will:
- Increase accuracy on the task
- Handle edge cases better
- Be clear and specific

Return ONLY the improved instructions text, nothing else.""",
)
```

The proposer sees the reflective dataset—a structured view of what went wrong:

```python
# Example reflective dataset entry
{
    "case_name": "noisy_email_thread",
    "inputs": {"text": "Re: Meeting Follow-up..."},
    "expected_output": {"name": "Robert Chen", ...},
    "actual_output": {"name": "Alex", ...},  # Wrong!
    "score": 0.6,
    "scores": {"accuracy": 0.6, "fields_correct": 3, "fields_total": 5}
}
```

This gives the proposer specific, actionable feedback to work with.

---

## Running the Optimization

### Setting Up the Environment

Create a `.env` file with your API key:

```bash
OPENAI_API_KEY=your-api-key-here
```

### Running Evaluation

First, let's see how the initial instructions perform:

```bash
uv run python -m prompt_optimization.run_optimization eval
```

Output:
```
Running evaluation with instructions:
Extract contact information from the provided text....
------------------------------------------------------------
Evaluation Results:
============================================================
  simple_email_signature: accuracy=1.00
  business_card_format: accuracy=1.00
  inline_contact: accuracy=1.00
  international_format: accuracy=0.80
  noisy_email_thread: accuracy=0.60
  partial_info: accuracy=1.00
  informal_intro: accuracy=0.75
  complex_signature: accuracy=0.80
------------------------------------------------------------
Average accuracy: 86.88%
```

The simple instructions work well on easy cases but struggle with complex ones.

### Running Optimization

Now let's run GEPA to improve the prompt:

```bash
uv run python -m prompt_optimization.run_optimization optimize --max-calls 50
```

The algorithm will:
1. Evaluate the initial prompt
2. Sample mini-batches and identify failures
3. Generate improved prompts based on feedback
4. Accept improvements that increase scores
5. Repeat until convergence or budget exhaustion

After optimization, you might see output like:

```
============================================================
Optimization Complete!
============================================================

Best validation score: 96.88%

Optimized Instructions:
Extract contact information from the provided text with high precision.

Guidelines for extraction:
1. NAME: Look for full names (first and last). Exclude titles (Dr., Mr., etc.)
   and credentials (Ph.D., Jr., III) from the extracted name.
2. EMAIL: Extract any valid email address in the format user@domain.tld.
3. PHONE: Extract phone numbers in any format, including international codes,
   parentheses, dashes, and vanity numbers.
4. COMPANY: Identify organization names, typically found near job titles or
   following words like "at" or "from".
5. TITLE: Extract job titles or roles, usually appearing near names or before
   company names.

Important considerations:
- When multiple contacts appear, focus on the PRIMARY contact being introduced
  or highlighted in the text.
- If information is genuinely missing, leave that field as null rather than
  guessing.
- Preserve the original formatting of phone numbers.
```

The optimized prompt is more specific, handles edge cases explicitly, and achieves higher accuracy.

---

## Best Practices and Tips

### 1. Design Your Dataset Carefully

The quality of optimization depends entirely on your test cases:

- **Cover the full difficulty spectrum**: Include easy cases (to avoid regression) and hard cases (to drive improvement)
- **Include adversarial examples**: What inputs might confuse the model?
- **Balance the dataset**: Don't let easy cases dominate—the algorithm will optimize for the average

### 2. Choose the Right Score Metric

Your `score_key` determines what GEPA optimizes for:

- **Use continuous scores** (0.0-1.0) rather than binary pass/fail
- **Ensure scores are comparable** across cases
- **Consider weighted scoring** if some fields matter more than others

### 3. Set Reasonable Budgets

- **`max_metric_calls`**: How many total evaluations to run
- **Start small** (20-50 calls) to verify the setup works
- **Scale up** (100-200 calls) for production optimization

### 4. Monitor with Logfire

Enable observability to understand what's happening:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

You'll be able to see:
- Each LLM call's inputs and outputs
- Token usage and latency
- The full evaluation trace

### 5. Use Separate Train/Validation Sets

For serious use, split your dataset:

```python
result = optimize(
    seed_candidate=seed_candidate,
    trainset=train_cases,  # Used for optimization
    valset=val_cases,      # Used for final evaluation
    adapter=adapter,
)
```

This will help prevent overfitting to specific cases.

---

## Challenges and Limitations

### Evaluation Quality is Everything

GEPA can only optimize what you can measure. If your evaluator has blind spots, the optimized prompt will exploit them. Invest significant effort in:
- Comprehensive test cases
- Robust evaluation logic
- Human review of edge cases

### Cost Considerations

Each optimization iteration involves:
- Multiple LLM calls for evaluation (one per test case)
- LLM calls for the proposer
- Validation runs

For 50 iterations with 8 test cases, that's 400+ LLM calls. Be thoughtful about whether your task really requires an expensive model—if it does, be prepared for the costs involved and monitor spending closely.

### This Example Only Optimizes a Single Module

Traditional genetic algorithms benefit from "crossover"—mixing genetic material from two parents. With a single prompt, there's nothing to cross over. GEPA still works because:
- The proposer LLM provides intelligent mutations
- The acceptance test prevents regression
- The reflective dataset provides rich feedback

But you won't get the benefits of crossover with just one module. The adapter pattern we've built here can be extended to optimize multiple components—for example, you could optimize both instructions *and* model selection (using `Agent.override(instructions=..., model=...)`) to let GEPA explore combinations of prompts and models. For multi-agent systems, you could optimize prompts across several agents simultaneously, which is where GEPA's genetic crossover really shines.

---

## Conclusion and Next Steps

We've built a complete prompt optimization pipeline that:

1. **Defines a structured task** with pydantic-ai
2. **Creates a rich evaluation dataset** with pydantic-evals
3. **Integrates with GEPA** through a custom adapter
4. **Automatically improves prompts** based on evaluation feedback

The key insights:

- **`Agent.override()`** enables clean prompt injection without modifying agent definitions
- **pydantic-evals** provides parallel evaluation, rich metrics, and observability
- **GEPA's reflective mutations** are smarter than random search
- **The adapter pattern** keeps components decoupled and reusable

---

## Resources

- [pydantic-ai Documentation](https://ai.pydantic.dev/)
- [pydantic-evals Documentation](https://ai.pydantic.dev/evals/)
- [GEPA GitHub Repository](https://github.com/gepa-ai/gepa)
- [Logfire](https://logfire.pydantic.dev/)
