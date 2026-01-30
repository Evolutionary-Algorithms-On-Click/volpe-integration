from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, RootModel

class NotebookCellMetadata(BaseModel):
    additionalProp1: Dict[str, Any] = Field(default_factory=dict)

class NotebookCell(BaseModel):
    cell_type: str
    cell_name: Optional[str] = None
    source: str
    execution_count: Optional[int] = None
    metadata: NotebookCellMetadata = Field(default_factory=NotebookCellMetadata)

class NotebookMetadata(BaseModel):
    additionalProp1: Dict[str, Any] = Field(default_factory=dict)

class Notebook(BaseModel):
    cells: List[NotebookCell] = Field(..., min_length=12, max_length=30)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    requirements: str = Field(default="", description="Newline-separated requirements for the code")

class Preferences(RootModel[Dict[str, Any]]):
    pass

class OptimizationRequest(BaseModel):
    user_id: str
    notebook_id: str
    notebook: Notebook
    preferences: Optional[Preferences] = None
    requirements: Optional[str] = None

class WrapperMainSchema(BaseModel):
    name: str
    version: str
    description: str | None = None

class JobMetadata(BaseModel):
    problemID: str | None = None  
    memory: int = Field(default=1, description="Memory in GB")
    targetInstances: int = Field(default=8, description="Number of worker instances")