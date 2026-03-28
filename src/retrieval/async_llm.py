import asyncio
from time import sleep

import aiohttp

from src.common.project_runtime import resolve_api_key, resolve_base_url, resolve_enable_thinking


def _normalize_base_url(base_url):
    base = (base_url or "").strip()
    if not base:
        base = "https://api.openai.com/v1"
    if not base.endswith("/"):
        base = base + "/"
    return base


async def create_completion(
    session,
    prompt,
    model="gpt-4o-mini",
    api_key="",
    base_url="",
    max_tokens=512,
    temperature=0.0,
    enable_thinking=False,
    extra_body=None,
):
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError("API key is required for async LLM completion.")

    base_url = _normalize_base_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "messages": [{"role": "user", "content": prompt}],
    }
    if isinstance(extra_body, dict) and extra_body:
        payload.update(extra_body)
    if enable_thinking:
        payload["enable_thinking"] = True

    retry = 0
    while retry < 5:
        try:
            async with session.post(
                url=f"{base_url}chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                body = await response.text()
                print(f"Error: {response.status} {body}")
        except Exception as e:
            retry += 1
            sleep(1)
            print(f"Error: {e}", flush=True)

    raise RuntimeError("Async completion failed after retries.")


async def run_async(
    prompts,
    model="gpt-4o-mini",
    api_key="",
    base_url="",
    max_tokens=512,
    temperature=0.0,
    enable_thinking=None,
    extra_body=None,
):
    resolved_api_key = resolve_api_key(explicit_key=(api_key or "").strip(), env_name="OPENAI_API_KEY")
    resolved_base_url = resolve_base_url(explicit_base=(base_url or "").strip(), env_name="OPENAI_BASE_URL")
    resolved_enable_thinking = resolve_enable_thinking(explicit_enable=enable_thinking)

    if not resolved_api_key:
        raise ValueError(
            "Missing API key. Provide api_key arg, or set OPENAI_API_KEY / DASHSCOPE_API_KEY, "
            "or configure project_settings.py."
        )

    async with aiohttp.ClientSession() as session:
        tasks = [
            create_completion(
                session,
                prompt,
                model=model,
                api_key=resolved_api_key,
                base_url=resolved_base_url,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=resolved_enable_thinking,
                extra_body=extra_body,
            )
            for prompt in prompts
        ]
        responses = await asyncio.gather(*tasks)
    return responses


if __name__ == "__main__":
    prompts = ["who are you?", "what you like?"]
    responses = asyncio.run(run_async(prompts, model="gpt-4o-mini"))
    print("--------")
    print(responses)
