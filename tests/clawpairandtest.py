import asyncio
import json
import websockets
import uuid
import hmac
import hashlib
import time
import subprocess
import sys

TOKEN = "<your token>" #i.e. "Jim*********2026"
WS_URL = "ws://127.0.0.1:18789"

async def connect_with_device_id(device_id):
    """Connect using a device ID"""
    print(f"\n🔌 Connecting with device ID: {device_id[:8]}...")
    
    headers = {"Origin": "http://127.0.0.1:18789"}
    
    try:
        async with websockets.connect(
            WS_URL,
            additional_headers=headers,
            subprotocols=["openclaw-v1"]
        ) as ws:
            print("✅ WebSocket connected")
            
            # First connect
            first_msg = {
                "type": "req",
                "id": str(uuid.uuid4()),
                "method": "connect",
                "params": {
                    "minProtocol": 3,
                    "maxProtocol": 3,
                    "client": {
                        "id": "cli",
                        "version": "2026.3.28",
                        "platform": "python",
                        "mode": "node"
                    },
                    "role": "node",
                    "scopes": ["node.read", "node.write"]
                }
            }
            
            print("📤 Sending connect...")
            await ws.send(json.dumps(first_msg))
            
            # Get challenge
            response = await ws.recv()
            data = json.loads(response)
            print(f"📥 Received: {data.get('event')}")
            
            if data.get("event") == "connect.challenge":
                nonce = data["payload"]["nonce"]
                print(f"🎲 Nonce: {nonce[:16]}...")
                
                # Create signature
                timestamp = int(time.time() * 1000)
                message = f"{nonce}{device_id}{timestamp}".encode()
                signature = hmac.new(
                    TOKEN.encode(),
                    message,
                    hashlib.sha256
                ).hexdigest()
                
                # Second connect with device auth
                second_msg = {
                    "type": "req",
                    "id": str(uuid.uuid4()),
                    "method": "connect",
                    "params": {
                        "minProtocol": 3,
                        "maxProtocol": 3,
                        "client": {
                            "id": "cli",
                            "version": "2026.3.28",
                            "platform": "python",
                            "mode": "node"
                        },
                        "role": "node",
                        "scopes": ["node.read", "node.write"],
                        "auth": {"token": TOKEN},
                        "device": {
                            "id": device_id,
                            "nonce": nonce,
                            "signature": signature,
                            "signedAt": timestamp
                        }
                    }
                }
                
                print("📤 Sending device auth...")
                await ws.send(json.dumps(second_msg))
                
                # Get auth result
                result = await ws.recv()
                auth_result = json.loads(result)
                print(f"📥 Auth result: {json.dumps(auth_result, indent=2)}")
                
                if auth_result.get("ok"):
                    print("\n✅✅✅ CONNECTION SUCCESSFUL! ✅✅✅")
                    
                    # Send a chat
                    chat_id = str(uuid.uuid4())
                    chat_msg = {
                        "type": "req",
                        "id": chat_id,
                        "method": "chat.send",
                        "params": {
                            "content": "Tell me a 3-word joke.",
                            "role": "user"
                        }
                    }
                    
                    print("\n📤 Sending chat...")
                    await ws.send(json.dumps(chat_msg))
                    
                    print("📨 Response: ", end="", flush=True)
                    while True:
                        msg = await ws.recv()
                        msg_data = json.loads(msg)
                        
                        if msg_data.get("event") == "chat.delta":
                            print(msg_data["payload"].get("content", ""), end="", flush=True)
                        elif msg_data.get("event") == "chat.done":
                            print("\n")
                            break
                        elif msg_data.get("type") == "res" and not msg_data.get("ok"):
                            print(f"\n❌ Error: {msg_data.get('error')}")
                            break
                    
                    return True
                else:
                    print(f"❌ Auth failed: {auth_result.get('error')}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

def list_devices():
    """List existing devices via CLI"""
    print("\n" + "="*60)
    print("Checking existing devices...")
    print("="*60)
    
    try:
        result = subprocess.run(
            ["openclaw", "devices", "list"],
            capture_output=True,
            text=True,
            timeout=5,
            shell=True
        )
        
        if result.returncode == 0:
            print(result.stdout)
            
            # Try to extract device IDs
            import re
            device_ids = re.findall(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', result.stdout, re.IGNORECASE)
            if device_ids:
                print(f"\n📱 Found device IDs:")
                for did in device_ids:
                    print(f"  - {did}")
                return device_ids
        else:
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    return []

def pair_new_device():
    """Pair a new device via CLI"""
    print("\n" + "="*60)
    print("Pairing new device...")
    print("="*60)
    
    try:
        result = subprocess.run(
            ["openclaw", "devices", "pair", "--name", "python-client"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=True
        )
        
        if result.returncode == 0:
            print("✅ Device paired!")
            print(result.stdout)
            
            # Try to extract device ID
            import re
            match = re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', result.stdout, re.IGNORECASE)
            if match:
                device_id = match.group()
                print(f"\n📱 New device ID: {device_id}")
                return device_id
        else:
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    return None

def main():
    print("="*60)
    print("OpenClaw Gateway - Node Device Connection")
    print("="*60)
    
    # First, list existing devices
    devices = list_devices()
    
    if devices:
        print(f"\nFound {len(devices)} existing device(s)")
        print("\nChoose a device ID to connect:")
        for i, did in enumerate(devices):
            print(f"  {i+1}. {did}")
        
        choice = input(f"\nEnter number (1-{len(devices)}) or press Enter to pair new device: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(devices):
            device_id = devices[int(choice)-1]
            asyncio.run(connect_with_device_id(device_id))
            return
    
    # No devices selected, try to pair new one
    print("\nNo device selected. Let's pair a new device.")
    print("\n" + "="*60)
    print("PAIRING INSTRUCTIONS")
    print("="*60)
    print("""
You need to pair a new device. Choose one method:

METHOD 1 - CLI (Recommended):
  In a NEW terminal, run:
    openclaw devices pair --name python-client
  
  Then copy the device ID that appears and paste it here.

METHOD 2 - Web UI:
  1. Open http://127.0.0.1:18789
  2. Go to Settings → Devices
  3. Click "Pair New Device"
  4. Name it "python-client"
  5. Copy the device ID

After you have the device ID, paste it below.
    """)
    
    device_id = input("\nPaste device ID (or press Enter to try CLI pairing): ").strip()
    
    if not device_id:
        # Try CLI pairing
        device_id = pair_new_device()
    
    if device_id:
        asyncio.run(connect_with_device_id(device_id))
    else:
        print("\n❌ No device ID provided. Cannot connect.")
        print("\nTo get a device ID, run in another terminal:")
        print("  openclaw devices pair --name python-client")

if __name__ == "__main__":
    main()