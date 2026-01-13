from typing import Optional
from pydantic import BaseModel

class Metadata(BaseModel):
    Code_Name: str
    Time_Series: str
    Start_Date: str
    End_Date: str
    Name: str
    Var_Name: str
    Units: str
    Var_Scale: Optional[float] = None
    Var_Offset: float
    Orig_Units: str
    Bands: int
    Processing: str
    Projection: str
    File_Path: str
    Comment: str
    MetadataLink: str