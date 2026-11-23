"""
Extracts C++ source from a raw model response (spec §6's 'no Markdown
fences' instruction, defended against here in case the model disobeys it).
Kept dependency-free (no torch/transformers) so orchestrator.py and tests
can use it without loading the ML stack.
"""

import re


# in src/code_extraction.py, add this at the top of extract_code()
import re

def extract_code(raw_response: str) -> str:
    # Strip DeepSeek-R1 chain-of-thought reasoning block before extracting code.
    # The <think>...</think> block appears before the actual response.
    #text = raw_response.strip()
    text = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
    # ... rest of the existing function unchanged

    fenced = re.match(r"^```(?:cpp|c\+\+|c)?\s*\n(.*?)\n?```$", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    if "```" in text:
        parts = text.split("```")
        candidates = [p for i, p in enumerate(parts) if i % 2 == 1]
        if candidates:
            longest = max(candidates, key=len)
            first_line, _, rest = longest.partition("\n")
            if first_line.strip().lower() in ("cpp", "c++", "c"):
                return rest.strip()
            return longest.strip()

    return text
