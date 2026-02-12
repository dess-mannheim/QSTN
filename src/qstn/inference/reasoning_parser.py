import re
from typing import List

def parse_reasoning(full_text: str, patterns: List[str]):
    # If we have no reasoning, directly output everything
    extracted_reasoning = None
    final_answer = full_text

    for start_tag, end_tag in patterns:
        # --- Check if start tag and end tag in full_text --- #
        has_start = (start_tag is not None and start_tag != "" and start_tag in full_text)
        has_end = (end_tag is not None and end_tag != "" and end_tag in full_text)

        # --- CASE A: No Reasoning ---
        if not has_start and not has_end:
            continue
    
        # --- CASE B: Standard Case (Start ... End) ---
        if has_start and has_end:
            # Prepare regex safe patterns
            esc_start = re.escape(start_tag)
            esc_end = re.escape(end_tag)
            pattern_full = re.compile(f"{esc_start}(.*?){esc_end}", re.DOTALL)
            match = pattern_full.search(full_text)

            if match:
                extracted_reasoning = match.group(1)
                # Remove reasoning + tags from answer
                final_answer = pattern_full.sub("", full_text, count=1).strip()
        # --- CASE C: No Start tag in generated text. ---
        elif has_end and not has_start:
            parts = full_text.split(end_tag, 1)

            extracted_reasoning = parts[0].strip()
            # If output stops exactly at reasoning -> empty final answer
            if len(parts) > 1:
                final_answer = parts[1].strip()
            else:
                final_answer = ""

        # --- CASE D: Cut-off Case (Max Tokens Hit), No End Tag but a start tag ---
        # Some models do not have a start token (e.g. Qwen-3-VL). 
        # In this case we cannot decide if it is reasoning or actual output
        elif has_start and not has_end:
            parts = full_text.split(start_tag, 1)
            extracted_reasoning = parts[1].strip()
            final_answer = ""  # Reasoning wasn't finished, so no answer exist
        
    return final_answer, extracted_reasoning