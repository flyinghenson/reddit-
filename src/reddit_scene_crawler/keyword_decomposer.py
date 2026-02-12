import json
from typing import List

from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL


SYSTEM_PROMPT = """You are an expert at decomposing a Chinese scenario description into highly specific, contextual English Reddit search keywords.

Rules:
1) All keywords MUST be in English (translate if needed)
2) Generate many SPECIFIC phrases (2-6 words) rather than generic single words
3) Prefer including WHO + WHEN + WHERE + WHAT qualifiers (user group, time, context, symptom/problem)
4) Avoid overly broad terms like "productivity", "health", "energy"
5) Output JSON: {"keywords":[{"keyword":"...","dimension":"..."}]}
"""


async def decompose_keywords(scene_description: str, *, keyword_count: int = 30) -> List[str]:
    """
    Use OpenAI to generate English Reddit search keywords.
    If you don't want to rely on OpenAI, run the script with --keywords.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'openai'. Install requirements, or run with --keywords to skip LLM."
        ) from e

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty. Set env var or run with --keywords.")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    resp = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Generate {keyword_count} keywords.\n\n"
                    f"Chinese scenario:\n{scene_description}"
                ),
            },
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content
    data = json.loads(content)
    items = data.get("keywords", []) if isinstance(data, dict) else data

    out: List[str] = []
    for it in items:
        if isinstance(it, str):
            k = it.strip()
        else:
            k = str((it or {}).get("keyword", "")).strip()
        if k:
            out.append(k)

    # de-dup preserve order
    out = list(dict.fromkeys(out))
    return out

