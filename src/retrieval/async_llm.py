import asyncio
import aiohttp
from time import sleep


async def create_completion(session,prompt,model="gpt-4o-mini"):
    API_KEY = ""
    BASE_URL = "https://us.vveai.com/v1/"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    retry = 0
    while retry < 5:
        try:
            async with session.post(url=f"{BASE_URL}chat/completions",
                json={
                    "model": model,
                    "max_tokens":512,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # print(result['choices'][0]['message']['content'])
                    # print('-----------------------')
                    return result['choices'][0]['message']['content']
                else:
                    print(f"Error: {response.status}")
        except Exception as e:
            retry += 1
            sleep(1)
            print(f"Error: {e}", flush=True)

async def run_async(prompts,model="gpt-4o-mini"):
    async with aiohttp.ClientSession() as session:
        tasks = [create_completion(session,prompt,model) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
    return responses

if __name__ == "__main__":
    prompts = ['who are you?','what you like?']
    responses = asyncio.run(run_async(prompts,model="gpt-4o-mini"))
    print('--------')
    print(responses)
