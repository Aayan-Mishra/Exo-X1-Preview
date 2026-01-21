#!/usr/bin/env python3
"""
ExoT2I/prompt_rpm30.py

Prompt generator that enforces a requests-per-minute (RPM) limit (default 30 RPM).
Generates batches of high-quality, novelist-style visual prompts via the Groq chat
API and saves them to a prompt JSON file (`prompt.json` by default) as a list of strings.

Usage:
    python ExoT2I/prompt_rpm30.py [--target N] [--batch B] [--rpm R] [--out path] [--dry-run] [--mock] [--debug]

Important:
- Set your GROQ_API_KEY in the environment before running:
    export GROQ_API_KEY="your_api_key_here"
- The script will try to ensure at least 60/RPM seconds between requests.
- Default RPM is 30 (2 seconds between requests).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Attempt to import the Groq client (best effort)
try:
    from groq import Groq  # type: ignore
except Exception:
    Groq = None  # We'll prompt the user to install it when necessary

# Default configuration values
DEFAULT_PROMPT_FILE = "prompt.json"
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TARGET = 1000
DEFAULT_BATCH = 5
DEFAULT_RPM = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_BATCH_SLEEP = 0.5  # Additional pause between batches (will be > min interval)

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


def load_prompts(path: str) -> List[str]:
    """Load existing prompts (normalize to a list of strings)."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, list):
        return [str(x).strip() for x in data]
    if (
        isinstance(data, dict)
        and "prompts" in data
        and isinstance(data["prompts"], list)
    ):
        return [str(x).strip() for x in data["prompts"]]
    # Fallback: stringify whatever we have
    return [str(data).strip()]


def save_prompts_list(path: str, prompts: List[str]) -> None:
    """Save the prompts as a JSON list (list of strings) to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)


def _stringify_item(item: Any) -> str:
    """Normalize an item (str or dict) into one descriptive string."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        # Prefer 'description' if available, else assemble from fields
        parts = []
        if item.get("title"):
            parts.append(str(item["title"]).rstrip(".") + ".")
        if item.get("description"):
            parts.append(str(item["description"]).rstrip(".") + ".")
        if item.get("style"):
            parts.append(f"Style: {item['style']}.")
        if item.get("mood"):
            parts.append(f"Mood: {item['mood']}")
        return " ".join(parts).strip()
    return str(item)


def extract_json_snippet(text: str) -> Optional[Any]:
    """
    Try to extract JSON from a text blob.
    First try json.loads on the full text; if that fails, try to find the first {...} substring.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


def build_messages(category: str, batch_size: int) -> List[Dict[str, Any]]:
    """
    Build strongly-worded system and user messages that force JSON output:
    A single JSON object with a 'prompts' key whose value is a list of batch_size strings.
    Each string: 50-120 words, flowing paragraph (Subject->Setting->Details->Lighting->Atmosphere),
    ending with 'Style: <style>.' and 'Mood: <mood>'
    """
    example = (
        '{"prompts": ["<paragraph 1> Style: <style>. Mood: <mood>", "...", "..."]}'
    )
    system = {
        "role": "system",
        "content": (
            "You are a master novelist and professional prompt engineer for high-end image diffusion models. "
            "Write flowing, narrative prose (not keyword lists). Use the structure Subject -> Setting -> Details -> Lighting -> Atmosphere. "
            "Your output MUST be EXACTLY one valid JSON object and NOTHING ELSE. The object MUST be: "
            f"{example} "
            f"The value of 'prompts' should be a list of {batch_size} STRINGS. Each string must be 50-120 words and must end with 'Style: <style>.' and 'Mood: <mood>'. "
            "Do not include any other keys or any extraneous text, explanation, or markdown. Use double quotes for strings."
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"Generate {batch_size} high-end, novelist-style visual prompts in the category: {category}. "
            "Write each prompt as a flowing paragraph (50-120 words). "
            "Describe lighting like a professional photographer (source, quality, direction, temperature). "
            "Append 'Style: <style>.' and 'Mood: <mood>' at the end of each paragraph. Return only the JSON object described in the system message and nothing else."
        ),
    }
    return [system, user]


def call_model_with_retries(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    min_interval: float,
    last_request_time_ref: Dict[str, float],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[Any]:
    """
    Call the chat completions API, respecting min_interval between calls (RPM enforcement).
    Returns parsed JSON (dict or list) on success, or None on failure.
    last_request_time_ref is a dict holding a mutable 'last' timestamp (so it can be updated by caller).
    """
    # We will attempt up to max_retries. On failure, we append a clarifying user message and retry.
    for attempt in range(1, max_retries + 1):
        # RPM: enforce minimum interval
        elapsed = time.time() - last_request_time_ref.get("last", 0.0)
        if elapsed < min_interval:
            wait = min_interval - elapsed
            logging.info(
                "Sleeping %.2fs to respect RPM (%d RPM)", wait, int(60.0 / min_interval)
            )
            time.sleep(wait)

        try:
            logging.debug("Sending request to model (attempt %d)...", attempt)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            last_request_time_ref["last"] = time.time()
        except Exception as exc:
            # Try to extract 'failed_generation' if present in the error payload
            failed_snippet = None
            try:
                resp = getattr(exc, "response", None)
                if resp is not None:
                    payload = resp.json()
                    failed_snippet = payload.get("error", {}).get("failed_generation")
            except Exception:
                failed_snippet = None

            logging.warning("Model request failed on attempt %d: %s", attempt, exc)
            if failed_snippet:
                logging.debug("failed_generation snippet:\n%s", failed_snippet)

            if attempt < max_retries:
                # Clarify and retry
                clarification = {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. Return EXACTLY one JSON object and NOTHING ELSE: "
                        f'{{"prompts": ["<paragraph 1> Style: <style>. Mood: <mood>", "...", "..."]}}'
                    ),
                }
                messages = messages + [clarification]
                time.sleep(1.0)
                continue
            else:
                return None

        # If we got here, the API returned something that passed server-side json_object validation.
        # Attempt to read the content
        try:
            content = response.choices[0].message.content
        except Exception:
            # Fallback: some clients might put JSON directly on response.content
            try:
                content = response.content
            except Exception:
                content = None

        if content is None:
            logging.warning(
                "No content found in model response on attempt %d.", attempt
            )
            if attempt < max_retries:
                time.sleep(1.0)
                continue
            else:
                return None

        # If content is not already a dict/list, try to parse
        if not isinstance(content, (dict, list)):
            parsed = extract_json_snippet(str(content))
        else:
            parsed = content

        if parsed is None:
            logging.warning(
                "Could not parse JSON from model output on attempt %d.", attempt
            )
            logging.debug("Raw model output:\n%s", content)
            if attempt < max_retries:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "The JSON you returned could not be parsed. RETURN ONLY: "
                            '{"prompts": ["<paragraph 1> Style: <style>. Mood: <mood>", ..., "<paragraph 5> Style: <style>. Mood: <mood>"]}'
                        ),
                    }
                ]
                time.sleep(1.0)
                continue
            else:
                return None

        # Parsed successfully
        return parsed

    return None


def normalize_to_string_list(parsed: Any) -> List[str]:
    """Given parsed JSON (dict or list), extract list of strings."""
    out: List[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            out.append(_stringify_item(item))
        return out
    if isinstance(parsed, dict):
        # Prefer 'prompts' key if present
        if "prompts" in parsed and isinstance(parsed["prompts"], list):
            for item in parsed["prompts"]:
                out.append(_stringify_item(item))
            return out
        # Otherwise, find the first list value
        for v in parsed.values():
            if isinstance(v, list):
                for item in v:
                    out.append(_stringify_item(item))
                return out
    return out


def generate_prompts_main(
    *,
    prompt_file: str,
    model_name: str,
    target_count: int,
    batch_size: int,
    rpm: int,
    max_retries: int,
    dry_run: bool,
    mock: bool,
) -> None:
    """Main generation loop."""
    if mock:
        logging.info("Running in MOCK mode: no network calls will be performed.")
    if not mock and Groq is None:
        logging.error("groq library not available. Install with: pip install groq")
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not mock and not api_key:
        logging.error("GROQ_API_KEY not set in environment. Please set it and retry.")
        sys.exit(1)

    client = None if mock else Groq(api_key=api_key)  # type: ignore
    prompts = load_prompts(prompt_file)
    logging.info("Loaded %d existing prompts from %s", len(prompts), prompt_file)

    min_interval = 60.0 / max(1, rpm)
    logging.info("RPM set to %d (min interval %.2fs)", rpm, min_interval)

    last_request_time_ref = {"last": 0.0}

    try:
        while len(prompts) < target_count:
            category = CATEGORIES[len(prompts) % len(CATEGORIES)]
            logging.info(
                "Generating next batch in category: %s (have %d/%d)",
                category,
                len(prompts),
                target_count,
            )
            messages = build_messages(category, batch_size)

            if mock:
                # Create mock prompts for quick testing
                new_prompts = []
                for i in range(batch_size):
                    new_prompts.append(
                        f"MOCK: A sample {category.lower()} prompt number {len(prompts) + i + 1}. Style: MockStyle. Mood: MockMood"
                    )
                parsed = {"prompts": new_prompts}
            else:
                parsed = call_model_with_retries(
                    client=client,
                    model=model_name,
                    messages=messages,
                    min_interval=min_interval,
                    last_request_time_ref=last_request_time_ref,
                    max_retries=max_retries,
                )
                if parsed is None:
                    logging.error(
                        "Failed to get valid JSON from the model after retries. Aborting generation."
                    )
                    break

            # Normalize to a list of strings
            batch_list = normalize_to_string_list(parsed)
            if not batch_list:
                logging.warning(
                    "Model returned JSON but no prompt list could be extracted. Skipping."
                )
                # small backoff and continue
                time.sleep(min_interval)
                continue

            # Append unique prompts and save progressively
            appended = 0
            for p in batch_list:
                if p not in prompts and len(prompts) < target_count:
                    prompts.append(p)
                    appended += 1

            logging.info(
                "Appended %d new prompts (total now %d).", appended, len(prompts)
            )

            if not dry_run:
                save_prompts_list(prompt_file, prompts)
                logging.debug("Saved prompts to %s", prompt_file)

            # Sleep between batches to be gentle and respect rate limits
            time.sleep(max(DEFAULT_BATCH_SLEEP, min_interval))

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Saving progress (if any).")
        if not dry_run:
            save_prompts_list(prompt_file, prompts)

    logging.info("Finished generation. Final prompt count: %d", len(prompts))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prompt generator with RPM enforcement (default 30 RPM)."
    )
    p.add_argument(
        "--target",
        "-t",
        type=int,
        default=int(os.environ.get("TARGET_COUNT", DEFAULT_TARGET)),
    )
    p.add_argument(
        "--batch",
        "-b",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", DEFAULT_BATCH)),
    )
    p.add_argument(
        "--rpm",
        type=int,
        default=int(os.environ.get("RPM", DEFAULT_RPM)),
        help="Requests per minute",
    )
    p.add_argument(
        "--out",
        "-o",
        default=os.environ.get("PROMPT_FILE", DEFAULT_PROMPT_FILE),
        help="Output JSON file",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME", DEFAULT_MODEL),
        help="Model name to call",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("MAX_RETRIES", DEFAULT_MAX_RETRIES)),
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Do not write file; just show progress"
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Mock mode (no API calls; generates dummy prompts)",
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    # Basic validation
    if args.batch <= 0 or args.target <= 0:
        logging.error("Batch and target must be positive integers.")
        sys.exit(1)

    if args.batch > 25:
        logging.warning(
            "Large batch sizes may be slow or hit API limits; consider smaller batches."
        )

    generate_prompts_main(
        prompt_file=args.out,
        model_name=args.model,
        target_count=args.target,
        batch_size=args.batch,
        rpm=args.rpm,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
