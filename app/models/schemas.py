import pydantic

# TODO: Add fields for wrapper API
class WrapperMainSchema(pydantic.BaseModel):
    name: str
    version: str
    description: str | None = None
