# Simple Vision Agent

A Naptha agent module for analyzing images using OpenAI's GPT-4 with vision capabilities.

## Prerequisites

1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install the Naptha SDK:
```bash
pip install naptha-sdk
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PRIVATE_KEY`: Your Naptha private key
- `NODE_URL`: Naptha node URL (default: http://node.naptha.ai:7001)

## Installation

```bash
git clone https://github.com/thestriver/simple_vision_agent
cd simple_vision_agent
poetry install
```

## Testing Locally

1. Test the agent directly:
```bash
poetry run python simple_vision_agent/run.py
```

This will run the default test case using the image URL specified in the run.py file.

## Deployment and Usage

1. Publish the agent to Naptha Hub:
```bash
naptha publish -r https://github.com/thestriver/simple_vision_agent
```

2. Run the agent on Naptha:
```bash
naptha run agent:simple_vision_agent -p "tool_name=vision tool_input_data=https://example.com/image.jpg"
```

Example:
```bash
naptha run agent:simple_vision_agent -p "tool_name=vision tool_input_data=https://docs.naptha.ai/assets/images/multi-node-flow-16da22dde6a48a22fabc86ed40d1bbd6.png"
```

## Configuration

The agent can be configured through the following files:



### Agent Deployment Configuration: `configs/deployment.json`


## Troubleshooting

1. If you get authentication errors:
   - Verify your PRIVATE_KEY is set correctly in .env
   - Ensure you're logged in to Naptha: `naptha login`

2. If the agent can't analyze images:
   - Check your OPENAI_API_KEY is valid and has access to vision models
   - Verify the image URL is accessible
   - Check the response in the logs for any error messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

