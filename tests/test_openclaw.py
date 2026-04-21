# claw_hello_world_proxy.py
import asyncio
from openclaw_sdk import OpenClawClient

async def main():
    # Connect to the proxy instead of the gateway
    client = await OpenClawClient.connect(
        gateway_ws_url="ws://127.0.0.1:18790",
        api_key="JimAvatar2026"
    )
    async with client:
        print("✅ Connected!")
        agent = client.get_agent("main")
        result = await agent.execute("Tell me a 3-word joke")
        print(f"Response: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())