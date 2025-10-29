from typing import Optional
from pydantic import BaseModel, Field, conint, confloat


class ProcessQuery(BaseModel):
    category_id: conint(ge=1, le=13) = Field(..., description="DeepFashion2 category id (1..13)")
    true_size: str = Field(..., description="Baseline apparel size label, e.g., XS,S,M,L,XL,XXL")
    true_waist: Optional[confloat(gt=0)] = Field(None, description="True waist in cm/inch according to 'unit'")
    unit: str = Field("cm", description="cm or inch")


class ProcessResponse(BaseModel):
    measurement_vis: str
    size_scale: str
    error: Optional[str] = None


