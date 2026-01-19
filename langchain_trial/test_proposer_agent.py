# Save this file as test_proposer_agent.py
# Before running, ensure you have the required packages:
# pip install -qU langchain transformers torch torchvision sentencepiece
# Note: This script will download the Qwen/Qwen2-VL-7B-Instruct model, which is large.

import json
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_classic.tools import Tool
from vlm_langchain_wrapper import VisionLanguageModel

def create_test_proposer_agent():
    """
    Creates a test agent that replicates the Proposer Agent's setup 
    from three_agent_system_langchain.py to debug parsing errors.
    """
    
    # 1. Define the tools available to the agent
    # This is the same tool used by the ProposerAgentLangChain
    def list_available_tools(_: str = "") -> str:
        """List all available XAI tools."""
        # In a real scenario, this would call xai_tools.get_available_tools()
        # For this test, we'll hardcode the response.
        available = ["gradcam", "integrated_gradients", "lime", "shap", "object_detection"]
        return json.dumps({
            "available_tools": available,
            "descriptions": {
                "gradcam": "Visualizes model attention regions",
                "integrated_gradients": "Pixel-level feature attribution",
                "lime": "Local interpretable explanations",
                "shap": "Shapley value-based explanations",
                "object_detection": "Detects objects in image"
            }
        })

    tools = [
        Tool(
            name="list_available_tools",
            func=list_available_tools,
            description="Lists all available XAI tools and their capabilities."
        )
    ]

    # 2. Define the prompt template
    # This is a direct copy of the prompt from ProposerAgentLangChain
    template = """You are a specialized XAI strategy agent. Your response MUST begin with 'Thought:' and you must strictly follow the ReAct framework. Do not add any conversational greetings, preambles, or other text before the 'Thought:'.

Your goal is to create a JSON strategy object for an XAI (Explainable AI) analysis.

Available tools for planning:
{tools}

Tool names: {tool_names}

IMPORTANT FORMAT RULES:
- First, think about the user's request based on the Question, Question Type, and Modality.
- Then, you should use the 'list_available_tools' tool to see the available XAI tools and their descriptions.
- Based on the tools available, select one or more appropriate tools. For a 'feature_attribution' question, 'gradcam' or 'integrated_gradients' are good choices. For a 'counterfactual' question, you might also use visualization tools.
- When you have decided on a strategy, you MUST provide the Final Answer in the specified JSON structure. Do not add any text after the JSON block.

When you need to use a tool:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <simple string input>

When you are READY to give final answer:
Thought: I have enough information to create the strategy.
Final Answer: <your JSON>

Request:
Question: {question}
Question Type: {question_type}
Model Info: {model_info}
Modality: {modality}

Final Answer JSON structure:
{{
    "strategy_type": "tools",
    "reasoning": "Brief explanation of why these XAI tools were selected based on the question type and modality.",
    "selected_tools": [
        {{
            "tool_name": "gradcam",
            "priority": 1,
            "reasoning": "This is a good general-purpose tool for visual feature attribution.",
            "parameters": {{}}
        }}
    ],
    "autonomous_tasks": []
}}

Valid tool_name values: gradcam, integrated_gradients, lime, shap, object_detection

REMEMBER: Your entire response must start with 'Thought:' and nothing else.

{agent_scratchpad}

Thought:"""

    prompt = PromptTemplate(
        input_variables=["question", "question_type", "model_info", "modality", "tools", "tool_names", "agent_scratchpad"],
        template=template
    )

    # 3. Initialize the LLM
    # Using the local VisionLanguageModel wrapper with the specified Qwen model.
    print("Loading VLM (this may take a while and download a large model)...")
    llm = VisionLanguageModel(
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        temperature=0
    )
    print("VLM loaded.")

    # 4. Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # 5. Create the Agent Executor
    def _handle_proposer_parsing_error(error: Exception) -> str:
        """Custom handler to provide more specific feedback on parsing errors."""
        error_str = str(error)
        print(f"DEBUG: Encountered a parsing error: {error_str}")
        # The agent might be outputting conversational text. Let's guide it.
        if "Could not parse LLM output" in error_str:
            return "Parsing error: Your response was not in the correct 'Thought/Action/Action Input' format. Please start your response with 'Thought:' and follow the ReAct framework strictly."
        return f"Parsing error: {error_str}. Please use format: Thought: <reasoning> Action: <tool_name> Action Input: <input>"

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=_handle_proposer_parsing_error,
    )
    
    return agent_executor

if __name__ == "__main__":
    print("Initializing Proposer Agent test...")
    test_agent_executor = create_test_proposer_agent()
    print("Initialization complete.\n")

    # Define mock inputs similar to the original system
    mock_inputs = {
        "question": "Why did the model predict 'cat' for this image?",
        "question_type": "feature_attribution",
        "model_info": json.dumps({
            "model_name": "resnet50",
            "model_type": "timm",
            "architecture": "ResNet",
            "num_classes": 1000,
        }),
        "modality": "vision",
    }
    
    print("=========================================")
    print("Invoking test agent with mock inputs...")
    print(f"Inputs: {json.dumps(mock_inputs, indent=2)}")
    print("=========================================\n")

    try:
        # Run the agent
        result = test_agent_executor.invoke(mock_inputs)
        print("\n=========================================")
        print("Agent execution finished.")
        print(f"Final Output: {result.get('output')}")
        print("=========================================")

    except Exception as e:
        print(f"\nAn error occurred during agent execution: {e}")

    print("\nTest script finished. If the agent ran successfully, you should see a JSON strategy as the final output.")
    print("If you still see parsing errors, the issue is likely with the LLM's ability to follow the very strict prompt format.")
