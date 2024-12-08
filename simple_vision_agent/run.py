#!/usr/bin/env python
from dotenv import load_dotenv
import json
from litellm import completion
from naptha_sdk.schemas import AgentDeployment, AgentRunInput, LLMConfig
import os
from pathlib import Path
from simple_vision_agent.schemas import InputSchema, SystemPromptSchema
from simple_vision_agent.utils import get_logger
from typing import List, Dict, Union

load_dotenv()

logger = get_logger(__name__)

# Get current directory (simple_vision_agent directory)
CURRENT_DIR = Path(__file__).parent

def load_configs():
    """Load both agent deployments and LLM configs from the correct location."""
    config_dir = CURRENT_DIR / "configs"
    
    # Load agent deployments
    with open(config_dir / "agent_deployments.json", "r") as f:
        agent_deployments_data = json.load(f)
    
    # Load LLM configs
    with open(config_dir / "llm_configs.json", "r") as f:
        llm_configs_data = json.load(f)
    
    # Match LLM configs with agent deployments
    llm_configs = {config["config_name"]: LLMConfig(**config) for config in llm_configs_data}
    
    for deployment in agent_deployments_data:
        llm_config_name = deployment["agent_config"]["llm_config"]["config_name"]
        deployment["agent_config"]["llm_config"] = llm_configs[llm_config_name]
    
    return [AgentDeployment(**deployment) for deployment in agent_deployments_data]

class SimpleVisionAgent:
    def __init__(self, agent_deployment: AgentDeployment):
        self.agent_deployment = agent_deployment
        self.system_prompt = SystemPromptSchema(
            role=agent_deployment.agent_config.system_prompt["role"], 
            persona=None
        )

    def format_messages(self, question: str, images: List[str]) -> List[Dict]:
        """Format messages for OpenAI's vision API."""
        content = [
            {"type": "text", "text": question}
        ]
        
        # Add up to 2 images
        for image_url in images[:2]:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        return [
            {"role": "system", "content": self.system_prompt.role},
            {"role": "user", "content": content}
        ]

    def run(self, inputs: Dict[str, Union[str, List[str]]]) -> str:
        """Run the vision agent with the given inputs."""
        logger.info("Running vision analysis")
        
        # Extract question and images from inputs
        question = inputs.get("question", "What can you tell me about these images?")
        images = inputs.get("images", [])
        if isinstance(images, str):
            images = [images]  # Convert single image to list
        
        # Format messages for the API call
        messages = self.format_messages(question, images)
        logger.info(f"Sending request with messages: {messages}")
        
        # Make the API call
        try:
            response = completion(
                model=self.agent_deployment.agent_config.llm_config.model,
                messages=messages,
                api_base=self.agent_deployment.agent_config.llm_config.api_base,
                max_tokens=self.agent_deployment.agent_config.llm_config.max_tokens,
                temperature=self.agent_deployment.agent_config.llm_config.temperature,
            )
            logger.info(f"Got response: {response}")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
            raise

def run(agent_run: AgentRunInput) -> str:
    """Run the agent with the given input."""
    logger.info(f"Running with inputs {agent_run.inputs.tool_input_data}")
    
    simple_vision_agent = SimpleVisionAgent(agent_run.agent_deployment)
    return simple_vision_agent.run(agent_run.inputs.tool_input_data)

if __name__ == "__main__":
    from naptha_sdk.client.naptha import Naptha

    naptha = Naptha()

    # Example usage with multiple images
    # input_params = InputSchema(
    #     tool_name="vision",
    #     tool_input_data={
    #         "images": [
    #             "https://docs.naptha.ai/assets/images/multi-node-flow-16da22dde6a48a22fabc86ed40d1bbd6.png",
    #             "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    #         ],
    #         "question": "What can you tell me about these images? What are the differences between them?"
    #     }
    # )
    # Example usage with a single image
    input_params = InputSchema(
        tool_name="vision",
        tool_input_data={
            "images": "https://docs.naptha.ai/assets/images/multi-node-flow-16da22dde6a48a22fabc86ed40d1bbd6.png",
            "question": "What can you tell me about this image?"
        }
    )

    # Load configs using our custom function
    agent_deployments = load_configs()

    agent_run = AgentRunInput(
        inputs=input_params,
        agent_deployment=agent_deployments[0],
        consumer_id=naptha.user.id,
    )

    response = run(agent_run)
    logger.info("=== Vision Analysis Result ===")
    logger.info(response)
    logger.info("=============================")