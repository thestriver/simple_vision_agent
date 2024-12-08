#!/usr/bin/env python
from dotenv import load_dotenv
from naptha_sdk.client.naptha import Naptha
from naptha_sdk.configs import load_agent_deployments
from naptha_sdk.schemas import AgentRunInput
from simple_vision_agent.schemas import InputSchema
from simple_vision_agent.run import run

load_dotenv()

def test_vision():
    naptha = Naptha()

    # Example usage with an image URL
    input_params = InputSchema(
        tool_name="vision",
        tool_input_data={
            "image": "https://docs.naptha.ai/assets/images/multi-node-flow-16da22dde6a48a22fabc86ed40d1bbd6.png",
            "question": "What can you tell me about this image?"
        }
    )

    # Load Configs
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
    print("Response: ", response)

if __name__ == "__main__":
    test_vision() 