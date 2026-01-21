import re
from typing import List

def parse_reasoning(full_text: str, patterns: List[str]):
    # If we have no reasoning, directly output everything
    extracted_reasoning = None
    final_answer = full_text

    for start_tag, end_tag in patterns:
        # --- CASE A: No Reasoning ---
        if start_tag not in full_text and end_tag not in full_text:
            continue
        # Prepare regex safe patterns
        esc_start = re.escape(start_tag)
        esc_end = re.escape(end_tag)
        # --- CASE B: Standard Case (Start ... End) ---
        if start_tag in full_text and end_tag in full_text:
            pattern_full = re.compile(f"{esc_start}(.*?){esc_end}", re.DOTALL)
            match = pattern_full.search(full_text)

            if match:
                extracted_reasoning = match.group(1)
                # Remove reasoning + tags from answer
                final_answer = pattern_full.sub("", full_text, count=1).strip()
        # --- CASE C: No Start tag in generated text. ---
        elif end_tag in full_text and start_tag not in full_text:
            parts = full_text.split(end_tag, 1)

            extracted_reasoning = parts[0].strip()
            if len(parts) > 1:
                final_answer = parts[1].strip()
            else:
                final_answer = ""

        # --- CASE D: Cut-off Case (Max Tokens Hit), No End Tag but a start tag ---
        elif start_tag in full_text and end_tag not in full_text:
            parts = full_text.split(start_tag, 1)
            extracted_reasoning = parts[1].strip()
            final_answer = ""  # Reasoning wasn't finished, so no answer exist
        
    return final_answer, extracted_reasoning