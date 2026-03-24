import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8000/api/v1/ws/stream?model_id=test"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            await websocket.send(json.dumps({"prediction": 0.5, "confidence": 0.9}))
            resp = await websocket.recv()
            print(f"Received: {resp}")
    except Exception as e:
        print(f"Failed: {e}")

asyncio.run(test_ws())
