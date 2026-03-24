import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8090/api/v1/ws/stream?model_id=test"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            await websocket.send(json.dumps({"prediction": 0.8}))
            print("Message sent")
            response = await websocket.recv()
            print(f"Received: {response}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
