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

    def vision(self, inputs: InputSchema) -> str:
        """Run the vision agent with the given inputs."""
        logger.info("Running vision analysis")
        
        if isinstance(inputs.tool_input_data, dict):
            # Extract question and images from input
            question = inputs.tool_input_data.get("question", "What can you tell me about these images?")
            images = inputs.tool_input_data.get("images", [])
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
        else:
            raise ValueError("Vision input must be a dictionary containing 'images' and optional 'question' keys")

def run(agent_run: AgentRunInput, *args, **kwargs):
    """Run the agent with the given input."""
    logger.info(f"Running with inputs {agent_run.inputs.tool_input_data}")
    
    simple_vision_agent = SimpleVisionAgent(agent_run.agent_deployment)
    method = getattr(simple_vision_agent, agent_run.inputs.tool_name, None)
    return method(agent_run.inputs)

if __name__ == "__main__":
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import load_agent_deployments

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
    agent_deployments = load_agent_deployments(
        "simple_vision_agent/configs/agent_deployments.json", 
        load_persona_data=False, 
        load_persona_schema=False
    )

    agent_run = AgentRunInput(
        inputs=input_params,
        agent_deployment=agent_deployments[0],
        consumer_id=naptha.user.id,
    )

    response = run(agent_run)
    logger.info("=== Vision Analysis Result ===")
    logger.info(response)
    logger.info("=============================")