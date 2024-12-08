from pydantic import BaseModel
from typing import Dict, Optional, Union, List

class InputSchema(BaseModel):
    tool_name: str
    tool_input_data: Dict[str, Union[str, List[str]]]  # Can contain question and list of image URLs

class SystemPromptSchema(BaseModel):
    """Schema for system prompts."""
    role: str = "You are a helpful AI assistant that can understand and discuss images."
    persona: Optional[Union[Dict, BaseModel]] = None