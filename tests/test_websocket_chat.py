import asyncio
import websockets
import json

async def test_websocket_chat():
    uri = "ws://localhost:8000/ws/chat"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to Avatar Chat WebSocket")
            
            # Receive welcome message
            welcome = await websocket.recv()
            print(f"📩 Welcome: {json.loads(welcome)['message']}")
            
            # Test prompts
            test_prompts = [
                "Tell me a short joke",
                "Tell me a 3-word joke", 
                "Why don't scientists trust atoms?"
            ]
            
            for prompt in test_prompts:
                print(f"\n📤 Sending: {prompt}")
                await websocket.send(prompt)
                
                # Receive thinking status
                thinking = await websocket.recv()
                thinking_data = json.loads(thinking)
                print(f"   {thinking_data['message']}")
                
                # Receive response
                response = await websocket.recv()
                response_data = json.loads(response)
                
                print(f"📥 Provider: {response_data.get('provider', 'unknown')}")
                print(f"📥 Response: {response_data['response']}")
                print(f"📥 Timestamp: {response_data['timestamp']}")
                
                await asyncio.sleep(2)  # Wait between requests
            
            print("\n✅ Test complete")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket_chat())