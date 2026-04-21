import asyncio
from openclaw_sdk import OpenClawClient

async def main():
    async with await OpenClawClient.connect(
        gateway_ws_url="ws://127.0.0.1:18790",
        api_key="your gateway apikey"
    ) as client:
        agent = client.get_agent("main")
        
        prompts = [
            "Tell me a short joke",
            "Tell me a 3-word joke",
            "Why don't scientists trust atoms?"
        ]
        
        for prompt in prompts:
            print(f"\n📤 {prompt}")
            result = await agent.execute(prompt)
            print(f"📥 {result.content}\n")

if __name__ == "__main__":
    asyncio.run(main())