"""Base contract behavior shared by stage-0 schemas."""

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
