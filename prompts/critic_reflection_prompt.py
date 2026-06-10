"""
Critic Reflection Prompt for Strategy Faithfulness Evaluation

Generates two separate reflection JSONs:
1. Proposer Reflection: Feedback on tool selection strategy
2. Actor Reflection: Feedback on explanation generation

This prompt is designed to be generic and work across all question types
and modalities (vision, text, tabular).
"""


def build_critic_reflection_prompt(
    question: dict,
    strategy: dict,
    explanation: dict,
    faithfulness_result: dict,
    tool_importance_details: str,
    threshold: float = 0.1
) -> str:
    """
    Build the critic reflection prompt.
    
    Args:
        question: Original question dict
        strategy: Strategy from Proposer
        explanation: Explanation from Actor (results dict)
        faithfulness_result: Faithfulness evaluation result
        tool_importance_details: Formatted string of tool importance scores
        threshold: Faithfulness threshold
        
    Returns:
        Formatted prompt string
    """
    # Extract strategy info
    selected_tools = strategy.get('selected_tools', [])
    tool_names = [t.get('tool_name', 'unknown') for t in selected_tools]
    strategy_reasoning = strategy.get('reasoning', 'No reasoning provided')
    strategy_type = strategy.get('strategy_type', 'tools')
    
    # Extract explanation info
    explanation_text = explanation.get('explanation', 
                       explanation.get('output', {}).get('explanation', 'No explanation'))
    agent_output = explanation.get('output', {})
    explanation_confidence = explanation.get('confidence', 0.0)
    
    # Extract faithfulness info
    faith_score = faithfulness_result.get('score', 0.0)
    faith_passed = faithfulness_result.get('passed', False)
    metric_name = faithfulness_result.get('metric_name', 'Unknown')
    metric_formula = faithfulness_result.get('metric_formula', '')
    p_original = faithfulness_result.get('p_original', 0.0)
    p_modified = faithfulness_result.get('p_modified', 0.0)
    
    # Format values safely
    faith_score = faith_score if faith_score is not None else 0.0
    p_original = p_original if p_original is not None else 0.0
    p_modified = p_modified if p_modified is not None else 0.0

    tool_importance_section = (
        f"\n## Tool Importance Analysis (Strategy Faithfulness)\n{tool_importance_details}\n"
        if tool_importance_details and tool_importance_details != "No tool importance scores available."
        else ""
    )

    prompt = f"""You are a critical analyst evaluating an XAI agent's strategy and explanation.

## Original Question
{question.get('question', question.get('example', 'Unknown question'))}

## Question Context
- Question Type: Q{question.get('q_type', 'unknown')}
- Modality: {question.get('modality', 'unknown')}

## Strategy Used
- Strategy Type: {strategy_type}
- Selected Tools: {tool_names}
- Strategy Reasoning: {strategy_reasoning}

## Generated Explanation
- Output: {agent_output}
- Explanation: {explanation_text}
- Confidence: {explanation_confidence}

## Explanation Faithfulness Evaluation
- Score: {faith_score:.4f} (threshold: {threshold})
- Status: {"PASSED" if faith_passed else "FAILED"}
- Metric: {metric_name} ({metric_formula})
- P_original: {p_original:.4f}, P_modified: {p_modified:.4f}
{tool_importance_section}## Your Task
Analyze the strategy and results, then provide TWO separate JSON outputs:

1. **Proposer Reflection**: Feedback for the Proposer Agent about tool selection strategy
2. **Actor Reflection**: Feedback for the Actor Agent about explanation generation

Think about:
- What went wrong (if faithfulness failed)
- Which tools were most/least helpful based on importance scores
- How the strategy could be improved
- How the explanation could be more accurate

**IMPORTANT**: Output exactly TWO JSON objects, clearly labeled.

=== PROPOSER_REFLECTION ===
{{
    "analysis": "Overall analysis of the strategy and tool selection",
    "tool_analysis": {{
        "tool_name": {{
            "effectiveness": "high/medium/low",
            "importance_score": 0.0,
            "recommendation": "keep/remove/replace_with_X"
        }}
    }},
    "tools_to_keep": ["tool1", "tool2"],
    "tools_to_remove": ["tool3"],
    "tools_to_add": ["tool4"],
    "strategy_suggestions": [
        "Specific suggestion for improving tool selection"
    ],
    "confidence": 0.0
}}

=== ACTOR_REFLECTION ===
{{
    "analysis": "Overall analysis of the explanation quality",
    "identified_issues": ["issue1", "issue2"],
    "region_feedback": "Feedback on the identified region/feature accuracy",
    "explanation_feedback": "Feedback on explanation clarity and correctness",
    "improvement_suggestions": [
        "Specific suggestion for improving the explanation"
    ],
    "confidence": 0.0
}}
"""
    return prompt


def parse_dual_reflection(response: str) -> tuple:
    """
    Parse the VLM response to extract both reflections.
    
    Args:
        response: Raw VLM response string
        
    Returns:
        Tuple of (proposer_reflection_str, actor_reflection_str)
        Returns the raw JSON strings to be passed to agents
    """
    import re
    import json
    
    proposer_reflection = None
    actor_reflection = None
    
    # Try to find sections by markers
    proposer_match = re.search(
        r'===\s*PROPOSER_REFLECTION\s*===\s*(\{[\s\S]*?\})\s*(?====|$)',
        response,
        re.IGNORECASE
    )
    actor_match = re.search(
        r'===\s*ACTOR_REFLECTION\s*===\s*(\{[\s\S]*?\})\s*(?====|$)',
        response,
        re.IGNORECASE
    )
    
    if proposer_match:
        proposer_reflection = proposer_match.group(1).strip()
    
    if actor_match:
        actor_reflection = actor_match.group(1).strip()
    
    # Fallback: try to find two JSON objects
    if not proposer_reflection or not actor_reflection:
        json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
        
        if len(json_objects) >= 2:
            if not proposer_reflection:
                proposer_reflection = json_objects[0]
            if not actor_reflection:
                actor_reflection = json_objects[1] if len(json_objects) > 1 else json_objects[0]
        elif len(json_objects) == 1:
            # Use same reflection for both if only one found
            proposer_reflection = proposer_reflection or json_objects[0]
            actor_reflection = actor_reflection or json_objects[0]
    
    # Provide defaults if parsing failed
    if not proposer_reflection:
        proposer_reflection = json.dumps({
            "analysis": "Unable to parse proposer reflection",
            "tool_analysis": {},
            "tools_to_keep": [],
            "tools_to_remove": [],
            "tools_to_add": [],
            "strategy_suggestions": [],
            "confidence": 0.0
        })
    
    if not actor_reflection:
        actor_reflection = json.dumps({
            "analysis": "Unable to parse actor reflection",
            "identified_issues": [],
            "region_feedback": "",
            "explanation_feedback": "",
            "improvement_suggestions": [],
            "confidence": 0.0
        })
    
    return proposer_reflection, actor_reflection


def format_tool_importance_details(tool_importance_scores: dict) -> str:
    """
    Format tool importance scores for the prompt.
    
    Args:
        tool_importance_scores: Dict of {tool_name: importance_score}
        
    Returns:
        Formatted string for prompt
    """
    if not tool_importance_scores:
        return "No tool importance scores available."
    
    lines = []
    sorted_tools = sorted(
        tool_importance_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for tool_name, score in sorted_tools:
        if score > 0.2:
            level = "high contribution"
        elif score > 0.1:
            level = "medium contribution"
        else:
            level = "low contribution"
        
        lines.append(f"- {tool_name}: importance={score:.4f} ({level})")
    
    return "\n".join(lines)
