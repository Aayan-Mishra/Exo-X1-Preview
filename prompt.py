#!/usr/bin/env python3
"""
Robust prompt generator for ExoT2I.

Responsibilities:
- Request batches of prompts from the Groq chat completions API.
- Enforce strict JSON output from the model:
  {"prompts": ["<paragraph 1> Style: <style>. Mood: <mood>", ... ]}
- Retry a small number of times on malformed outputs and surface helpful debug info.
- Save prompts to `prompt.json` (a list of strings) for downstream consumers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, List

try:
    from tqdm import tqdm
except Exception:
    # Minimal fallback if tqdm isn't available. Provide a simple object that supports update() and close()
    # and can be used as an iterable when called as tqdm(iterable) with no kwargs.
    class _TqdmFallback:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get("total")
            self.n = kwargs.get("initial", 0)

        def update(self, n=1):
            self.n += n

        def close(self):
            pass

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

    def tqdm(iterable=None, **kwargs):
        # If called like tqdm(range(...)) with no kwargs, return the iterable to allow normal iteration.
        if iterable is not None and not kwargs:
            return iterable
        return _TqdmFallback(iterable, **kwargs)


try:
    from groq import Groq
except Exception:
    Groq = None  # We'll handle absence gracefully below

# Configuration
PROMPT_FILE = "prompt.json"
TARGET_COUNT = int(os.environ.get("TARGET_COUNT", "1000"))
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
GROQ_API_KEY = "YOUR_GROQ_API_KEY"  # Encouraged to set in env instead of hardcoding
MAX_RETRIES = 3
SLEEP_BETWEEN_REQUESTS = 0.7  # seconds

CATEGORIES = [
    "Hyper-realistic Portraits",
    "Futuristic Cyberpunk Cityscapes",
    "Ethereal Fantasy Landscapes",
    "Minimalist Product Design",
    "Macro Nature Photography",
    "Surrealist Abstract Art",
    "Brutalist & Gilded Architecture",
    "Deep Space Nebula & Galaxies",
    "Cinematic Noir Street Photography",
    "Ancient Mythological Scenes",
    "Post-Apocalyptic Overgrown Ruins",
    "High-Fashion Editorial Shoots",
    "Vibrant Pop Art Illustrations",
    "Intricate Steampunk Machinery",
    "Serene Japanese Zen Gardens",
]

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("prompt_generator")


def load_prompts() -> List[str]:
    """Load existing prompts; normalize into a list of strings."""
    if not os.path.exists(PROMPT_FILE):
        return []
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(
            "Could not read or parse %s: %s. Starting fresh.", PROMPT_FILE, e
        )
        return []

    # Normalize many potential shapes into a list of strings
    if isinstance(data, list):
        return [_stringify_item(item) for item in data]
    if (
        isinstance(data, dict)
        and "prompts" in data
        and isinstance(data["prompts"], list)
    ):
        return [_stringify_item(item) for item in data["prompts"]]
    # Fallback: try to stringify whatever we got
    try:
        return [str(data)]
    except Exception:
        return []


def save_prompts(prompts: List[str]) -> None:
    """Persist the prompts as a JSON list (list of strings)."""
    with open(PROMPT_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=4, ensure_ascii=False)


def _stringify_item(item: Any) -> str:
    """Turn a prompt item into a single descriptive string.

    Accepts either a string or an object like {"title":..., "description":..., "style":..., "mood":...}.
    """
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        # Prefer a 'description' if present, otherwise try to assemble from other fields
        parts: List[str] = []
        if item.get("title"):
            parts.append(item["title"].rstrip(".") + ".")
        if item.get("description"):
            parts.append(item["description"].rstrip(".") + ".")
        # Append style/mood annotations if present
        if item.get("style"):
            parts.append(f"Style: {item['style']}.")
        if item.get("mood"):
            parts.append(f"Mood: {item['mood']}")
        text = " ".join(parts).strip()
        return text
    # Fallback
    return str(item).strip()


def _build_system_and_user_messages(category: str) -> List[dict]:
    """Return system and user messages that strongly enforce the JSON schema."""
    # Keep this single-line-ish JSON example short and valid; it helps the model understand exact format.
    example_json = '{"prompts": ["<paragraph 1> Style: <style>. Mood: <mood>", "...", "...", "...", "..."]}'

    system_msg = {
        "role": "system",
        "content": (
            "You are a master novelist and professional prompt engineer for high-end image diffusion models. "
            "Write flowing, descriptive prose following: Subject -> Setting -> Details -> Lighting -> Atmosphere. "
            "Mastery of lighting is required (source, quality, direction, temperature). "
            "IMPORTANT OUTPUT REQUIREMENT: Return EXACTLY ONE valid JSON object and NOTHING ELSE. "
            "The JSON object MUST have a single key, 'prompts', whose value is a list of 5 STRINGS. "
            "Each string must be a single paragraph (about 50-120 words) that ends with 'Style: <style>.' and 'Mood: <mood>' (e.g. 'Style: Neo-Realism. Mood: Contemplative'). "
            "Do not include any keys other than 'prompts', do not wrap your result in markdown or code fences, and do not add commentary. "
            f"Example: {example_json}"
        ),
    }

    user_msg = {
        "role": "user",
        "content": (
            f"Generate 5 high-end, novelist-style visual prompts in the category: {category}. "
            "Write each prompt as a single flowing paragraph (50-120 words). "
            "Describe lighting like a professional photographer. "
            "Append 'Style: <style>.' and 'Mood: <mood>' at the end of each paragraph. "
            "Return only the JSON object described above and nothing else."
        ),
    }

    return [system_msg, user_msg]


def _extract_failed_generation_from_exception(e: Exception) -> str | None:
    """Try to pull the server-side 'failed_generation' snippet out of the exception if available."""
    # Use getattr to avoid static-analysis errors about 'response' not existing on a generic Exception
    resp = getattr(e, "response", None)
    if not resp:
        return None
    try:
        body = resp.json()
        return body.get("error", {}).get("failed_generation")
    except Exception:
        # Not able to parse; return a string representation if possible
        try:
            return str(resp)
        except Exception:
            return None


def generate_prompts() -> None:
    if not GROQ_API_KEY:
        logger.error(
            "GROQ_API_KEY not set in environment. Please set it and try again."
        )
        return

    if Groq is None:
        logger.error("groq library not available. Install it with: pip install groq")
        return

    client = Groq(api_key=GROQ_API_KEY)
    prompts = load_prompts()
    current_count = len(prompts)
    logger.info("Current prompt count: %d / %d", current_count, TARGET_COUNT)

    if current_count >= TARGET_COUNT:
        logger.info("Already reached target prompt count.")
        return

    pbar = tqdm(total=TARGET_COUNT, initial=current_count)

    try:
        while len(prompts) < TARGET_COUNT:
            category = CATEGORIES[len(prompts) % len(CATEGORIES)]
            messages = _build_system_and_user_messages(category)

            raw_new_prompts: List[str] = []
            attempt = 0
            while attempt < MAX_RETRIES and not raw_new_prompts:
                attempt += 1
                try:
                    logger.info(
                        "Requesting prompts (category='%s', attempt=%d)...",
                        category,
                        attempt,
                    )
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                except Exception as e:
                    # Attempt to surface diagnostics returned by the API (if present)
                    failed_snippet = _extract_failed_generation_from_exception(e)
                    logger.warning("API call failed on attempt %d: %s", attempt, e)
                    if failed_snippet:
                        logger.debug("Failed generation snippet:\n%s", failed_snippet)
                    if attempt < MAX_RETRIES:
                        logger.info("Retrying with clarified instruction...")
                        # Add a follow-up nudge to be stricter
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response could not be parsed as valid JSON. "
                                    "Please return EXACTLY one JSON object and nothing else: "
                                    '{"prompts": ["<paragraph 1> Style: <style>. Mood: <mood>", "...", "...", "...", "..."]}'
                                ),
                            }
                        )
                        time.sleep(1)
                        continue
                    else:
                        raise

                # If we get here the API returned successfully (server-side validated it as json_object)
                # The client library usually surfaces the JSON object in response.choices[0].message.content,
                # but to be robust, try to parse the content
                try:
                    content = response.choices[0].message.content
                except Exception:
                    # defensive fallback
                    content = None

                if not content:
                    logger.warning(
                        "No content found in the model response on attempt %d.", attempt
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(1)
                        continue
                    else:
                        raise RuntimeError("Model returned an empty response")

                # Parse the content as JSON; it's already validated by the server, but be defensive
                try:
                    parsed = json.loads(content)
                except Exception as e:
                    logger.warning(
                        "Failed to parse JSON from model output on attempt %d: %s",
                        attempt,
                        e,
                    )
                    logger.debug("Raw output:\n%s", content)
                    if attempt < MAX_RETRIES:
                        # nudge the model to stick to the exact JSON form
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "The JSON you returned could not be parsed. "
                                    'RETURN ONLY: {"prompts": ["<paragraph 1> Style: <style>. Mood: <mood>", ..., "<paragraph 5> Style: <style>. Mood: <mood>"]}'
                                ),
                            }
                        )
                        time.sleep(1)
                        continue
                    else:
                        raise

                # Get the list out of parsed data
                candidate_list = []
                if isinstance(parsed, dict) and isinstance(parsed.get("prompts"), list):
                    candidate_list = parsed["prompts"]
                elif isinstance(parsed, list):
                    candidate_list = parsed
                else:
                    # Try to find any list in the dict
                    if isinstance(parsed, dict):
                        for v in parsed.values():
                            if isinstance(v, list):
                                candidate_list = v
                                break

                if not candidate_list:
                    logger.warning(
                        "No list of prompts found in parsed JSON on attempt %d: %s",
                        attempt,
                        parsed,
                    )
                    if attempt < MAX_RETRIES:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "I could not find a list of prompts in your JSON. "
                                    'Please return exactly: {"prompts": [<5 strings>]} and nothing else.'
                                ),
                            }
                        )
                        time.sleep(1)
                        continue
                    else:
                        raise RuntimeError(
                            "Model returned JSON without a 'prompts' list"
                        )

                # Normalize into strings
                for item in candidate_list:
                    s = _stringify_item(item)
                    if s:
                        raw_new_prompts.append(s)

                # Basic sanity check: we expected 5 prompts
                if len(raw_new_prompts) != 5:
                    logger.warning(
                        "Expected 5 prompts, model returned %d items. Attempt %d.",
                        len(raw_new_prompts),
                        attempt,
                    )
                    if attempt < MAX_RETRIES:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"You returned {len(raw_new_prompts)} items but I need exactly 5 strings in 'prompts'. "
                                    "Please return exactly 5 strings in the array and nothing else."
                                ),
                            }
                        )
                        raw_new_prompts = []
                        time.sleep(1)
                        continue
                    else:
                        # We'll accept what's available at the last attempt, but still proceed
                        logger.info(
                            "Proceeding with the %d prompts returned on the last attempt.",
                            len(raw_new_prompts),
                        )

            # If we exit attempts loop without prompts, break out to avoid infinite loop
            if not raw_new_prompts:
                logger.error(
                    "Failed to generate a valid batch of prompts after %d attempts. Aborting.",
                    MAX_RETRIES,
                )
                break

            # Append unique new prompts
            for p in raw_new_prompts:
                if p not in prompts and len(prompts) < TARGET_COUNT:
                    prompts.append(p)
                    pbar.update(1)

            # Save intermediate state frequently
            save_prompts(prompts)

            # Respect rate limits
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        pbar.close()
        save_prompts(prompts)
        logger.info("Generation finished. Final prompt count: %d", len(prompts))

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Saving progress...")
        save_prompts(prompts)
    except Exception as exc:
        logger.exception("Error while generating prompts: %s", exc)
        save_prompts(prompts)


if __name__ == "__main__":
    generate_prompts()
