from typing import Literal, Annotated, TypedDict, List
from datetime import datetime, date

MarketTypeAnnotation = Literal["premarket", "regular", "afterhrs"]
TargetColTypeHint = Literal["Close", "Open", "High", "Low"]

class ProbaReturnType(TypedDict):
    probability: float
    case_date: List[date]
    
