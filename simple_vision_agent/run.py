#!/usr/bin/env python
from dotenv import load_dotenv
from naptha_sdk.schemas import AgentRunInput, AgentDeployment
import os
from simple_vision_agent.schemas import InputSchema, SystemPromptSchema
import logging
from typing import List, Dict, Union
from naptha_sdk.inference import InferenceClient
from naptha_sdk.user import sign_consumer_id
import json
import requests

load_dotenv()

logger = logging.getLogger(__name__)

class SimpleVisionAgent:
    def __init__(self, deployment: AgentDeployment):
        self.deployment = deployment
        self.system_prompt = SystemPromptSchema(
            role=deployment.config.system_prompt["role"], 
            persona=None
        )
        
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = "https://api.openai.com/v1/chat/completions"
        
        if self.api_key is None:
            raise ValueError("OpenAI API key not set")

    def vision(self, inputs: InputSchema) -> str:
        """Run the vision agent with the given inputs."""
        logger.info(f"Running vision analysis for image: {inputs.tool_input_data}")
        
        data = {
            "model": self.deployment.config.llm_config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": inputs.tool_input_data
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.deployment.config.llm_config.max_tokens
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(self.api_base, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise ValueError(f"Failed to generate response: {str(e)}")

def run(module_run: Dict, *args, **kwargs) -> str:
    """Run the agent for image analysis."""
    module_run = AgentRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)
    
    simple_vision_agent = SimpleVisionAgent(module_run.deployment)
    method = getattr(simple_vision_agent, module_run.inputs.tool_name, None)
    
    if not method:
        raise ValueError(f"Method {module_run.inputs.tool_name} not found")
    
    return method(module_run.inputs)

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    
    naptha = Naptha()
    
    deployment = asyncio.run(setup_module_deployment(
        "agent", 
        "simple_vision_agent/configs/deployment.json",
        node_url=os.getenv("NODE_URL")
    ))
    
    input_params = {
        "tool_name": "vision",
        "tool_input_data": "https://docs.naptha.ai/assets/images/multi-node-flow-16da22dde6a48a22fabc86ed40d1bbd6.png"
    }
    
    module_run = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }
    
    print(run(module_run))