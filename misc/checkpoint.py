from typing import TypedDict, Dict, List, Any, Optional


class CheckPoint(TypedDict):
    net: Dict[str, Any]
    opts: Optional[List[Dict[str, Any]]]
