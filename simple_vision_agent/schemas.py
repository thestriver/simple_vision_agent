from pydantic import BaseModel
from typing import Dict, Optional, Union

class InputSchema(BaseModel):
    tool_name: str
    tool_input_data: str

class SystemPromptSchema(BaseModel):
    """Schema for system prompts."""
    role: str = "You are a helpful AI assistant that can understand and discuss images."
    persona: Optional[Dict] = None