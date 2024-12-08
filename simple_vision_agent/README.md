# Simple Vision Module

A Naptha module for analyzing images using GPT-4 Vision. This module allows you to send an image URL along with an optional question to get AI-powered analysis and descriptions of images.

## Usage

You can use this module through the Naptha SDK:

```bash
naptha run module:simple_vision_agent -p "tool_name='vision' tool_input_data={'image': 'https://example.com/image.jpg', 'question': 'What is in this image?'}"
```

### Input Format

The module expects input in the following format:
- `tool_name`: Should be "vision"
- `tool_input_data`: A dictionary containing:
  - `image`: URL of the image to analyze
  - `question`: (Optional) Specific question about the image. If not provided, will default to general image description.

### Requirements

- OpenAI API key with GPT-4 Vision access
- Python 3.8+
- Naptha SDK

### Environment Variables

Make sure to set the following in your .env file:
```
OPENAI_API_KEY=your_openai_api_key
NODE_URL=http://localhost:7001
```

## Development

To run locally for development:

```bash
poetry install
poetry run python simple_vision_agent/run.py
``` 