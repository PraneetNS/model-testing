import json
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List, Dict, Any

class Item(BaseModel):
    name: str
    time: datetime

class Result(BaseModel):
    items: List[Item]
    meta: Dict[str, Any]

# Simulate the data
res = Result(
    items=[Item(name="test", time=datetime.utcnow())],
    meta={"timestamp": datetime.utcnow()}
)

print("--- Original ---")
try:
    print(json.dumps(res.dict()))
except Exception as e:
    print(f"res.dict() fails: {e}")

print("\n--- JSONABLE ENCODER ---")
data = jsonable_encoder(res)
print(f"Type of timestamp: {type(data['meta']['timestamp'])}")
print(f"Value: {data['meta']['timestamp']}")
try:
    print(f"Success: {json.dumps(data)}")
except Exception as e:
    print(f"Fails: {e}")
