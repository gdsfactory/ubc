"""Test validator."""


from typing import Literal
from pydantic import BaseModel, ValidationError, validator


class SequenceStepModel(BaseModel):
    """Model."""

    polarization: Literal['TE', 'TM']


    @validator("polarization")
    def check_polariztion(cls, v, values, **kwargs):
        """Validator."""
        if polarization not in ['TE', 'TM']:
            raise ValueError("must be greater than a")
        return v

