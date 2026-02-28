"""
Baseline Reasoning Preambles

Defines the extra reasoning instructions that are **prepended** to the
prompt builders' output for each baseline type (CoT, ReAct, ToT).

These preambles are injected into the *existing* pipeline prompts via
monkey-patching (see patch_prompts.py), so no changes to the pipeline
code itself are required.

Naive baseline uses no preamble — it just runs the pipeline as-is with
--no_improvement and --no-sf.
"""

# ---------------------------------------------------------------------------
# Chain-of-Thought preamble
# ---------------------------------------------------------------------------
COT_PROPOSER_PREAMBLE = """
## Reasoning Mode: Chain-of-Thought

Before selecting your XAI tools and strategy, think through the problem
step-by-step:

1. **Understand the question**: What exactly is being asked about this
   model's prediction?
2. **Consider the modality**: What kinds of evidence are most relevant
   for this data type?
3. **Plan your analysis**: Which XAI tools will give you the most direct
   evidence to answer this question?
4. **Anticipate the answer format**: What specific information do you need
   to produce a complete, well-supported response?

Think step by step, then provide your strategy.

"""

COT_ACTOR_PREAMBLE = """
## Reasoning Mode: Chain-of-Thought

Before generating your explanation, reason through the evidence
step-by-step:

1. **Understand the question**: What exactly is being asked?
2. **Analyze the XAI tool results**: What do the attribution maps,
   importance scores, and statistics tell us?
3. **Identify the key features/regions**: Based on the tool outputs,
   which specific parts of the input are most relevant?
4. **Synthesize your answer**: Combine the evidence into a clear,
   specific, and well-supported explanation.

Think step by step, then provide your final answer as valid JSON.

"""

# ---------------------------------------------------------------------------
# ReAct preamble (applied to proposer + actor within the 3-agent system)
# ---------------------------------------------------------------------------
REACT_PROPOSER_PREAMBLE = """
## Reasoning Mode: ReAct (Reasoning + Acting)

Use a ReAct-style approach to plan your XAI tool strategy:

**Thought**: Consider what evidence you need to answer this question.
What aspects of the model's decision are most relevant?

**Action**: Select the tools that will provide the most informative
evidence for this specific question type and modality.

**Observation**: Anticipate what each tool will reveal, and ensure
your strategy covers all aspects needed for a complete answer.

Think about what you'd observe from each tool, then decide your
strategy accordingly.

"""

REACT_ACTOR_PREAMBLE = """
## Reasoning Mode: ReAct (Reasoning + Acting)

Interleave your reasoning with the evidence to build your explanation:

For each piece of evidence from the XAI tools:
  **Thought**: What does this result tell us about the model's decision?
  **Observation**: Note the key values, features, or regions highlighted.
  **Reflection**: How does this connect to answering the question?

After processing all evidence:
  **Final Thought**: Synthesize all observations into a coherent answer.

Then provide your final answer as valid JSON.

"""

# ---------------------------------------------------------------------------
# Tree-of-Thought preamble
# ---------------------------------------------------------------------------
TOT_PROPOSER_PREAMBLE = """
## Reasoning Mode: Tree of Thought

Consider multiple possible analysis strategies before selecting one:

**Branch 1**: What if you focus on attribution-based tools (LIME, SHAP)?
  - What evidence would this provide?
  - How well does this address the question?

**Branch 2**: What if you focus on gradient-based tools (GradCAM, IG)?
  - What complementary evidence would this add?
  - Does this approach capture different aspects?

**Branch 3**: What if you combine both approaches?
  - What's the most comprehensive strategy?
  - Is there redundancy or complementary value?

Evaluate which branch provides the strongest evidence for this specific
question, then select the best strategy.

"""

TOT_ACTOR_PREAMBLE = """
## Reasoning Mode: Tree of Thought

Consider multiple possible interpretations of the evidence before
settling on your answer:

**Interpretation 1**: What does the evidence suggest if you weight
  the highest-attribution features most heavily?

**Interpretation 2**: What alternative explanation could account for
  the same tool results?

**Interpretation 3**: Which interpretation is most consistent with
  ALL the evidence, not just the strongest signal?

Evaluate each interpretation against the evidence quality, specificity,
and completeness. Select the strongest interpretation and provide
your final answer as valid JSON.

"""

# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------
PROPOSER_PREAMBLES = {
    "naive": "",  # No preamble for naive
    "cot": COT_PROPOSER_PREAMBLE,
    "react": REACT_PROPOSER_PREAMBLE,
    "tot": TOT_PROPOSER_PREAMBLE,
}

ACTOR_PREAMBLES = {
    "naive": "",
    "cot": COT_ACTOR_PREAMBLE,
    "react": REACT_ACTOR_PREAMBLE,
    "tot": TOT_ACTOR_PREAMBLE,
}
