import random
import string

from qstn.inference.survey_inference import *

def key_typos(text: str, probability: float = 0.1) -> str:
    """
    Randomly replaces characters with random alphabet letters to simulate typos.
    Args:
        text (str): The input text to perturb.
        probability (float): The probability of replacing each character.
    Returns:
        str: The text with random character replacements based on the given probability.
    """
    if not text: 
        return text
    
    # Get all possible letters (a-z and A-Z)
    alphabet = string.ascii_letters 
    
    text_list = list(text)
    
    for i, char in enumerate(text_list):
        # We check char.isalpha() so we don't replace spaces or punctuation
        if char.isalpha() and random.random() < probability:
            text_list[i] = random.choice(alphabet)
            
    return "".join(text_list)


def keyboard_typos(text: str, probability: float = 0.1) -> str:
        """
        Introduces typos based on keyboard proximity.
        Args:
            text (str): The input text to perturb.
            probability (float): The probability of introducing a typo for each character.
        Returns:
            str: The text with keyboard-based typos introduced based on the given probability.
        """
        keyboard_neighbors = {
            'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfcx', 'e': 'wsdr', 
            'f': 'rtgdvc', 'g': 'tyfhbv', 'h': 'yugjbn', 'i': 'ujko', 'j': 'uikhnm',
            'k': 'ijolm', 'l': 'opk', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
            'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedz', 
            't': 'rfgy', 'u': 'yhji', 'v': 'cfgb', 
            'w': 'qase', 'x': 'zsdc', 
            'y': 'tugh', 
            'z': 'asx'
        }
        
        if not text: return text
        
        text_list = list(text)
        for i in range(len(text_list)):
            char = text_list[i].lower()
            if char in keyboard_neighbors and random.random() < probability:
                neighbors = keyboard_neighbors[char]
                typo_char = random.choice(neighbors)
                # Preserve original case
                if text_list[i].isupper():
                    typo_char = typo_char.upper()
                text_list[i] = typo_char
        return "".join(text_list)    

def letter_swaps(text: str, probability: float = 0.1) -> str:
        """
        Randomly swaps adjacent letters in the text.
        Args:
            text (str): The input text to perturb.
            probability (float): The probability of swapping each adjacent letter pair.
        Returns:
            str: The text with adjacent letters swapped based on the given probability.
        """
        if not text: return text
        
        text_list = list(text)
        i = 0
        while i < len(text_list) - 1:
            if random.random() < probability:
                text_list[i], text_list[i+1] = text_list[i+1], text_list[i]
                i += 2  # Skip next character to avoid double swapping
            else:
                i += 1
        return "".join(text_list)



def make_synonyms(all_prompts: List[str], model: str, instruction: str) -> str:
    """
    Uses a language model to replace words with their synonyms.
    Args:
        all_prompts (List[str]): The input prompts as a list to perturb.
        model (str): The language model to use for generating synonyms as a vllm LLM object.
        instruction (str): The instruction prompt for the model.
    Returns:
        List[str]: The prompts with words replaced by their synonyms as a list of strings.
    """
    system_msg = "You are a helpful assistant that replaces words with their synonyms while preserving the original meaning."        
    
    all_segments_to_perturb = []
    prompt_maps = []

    for prompt in all_prompts:
        parts = re.split(r'(\{.*?\})', prompt)
        structure = [] # Stores (is_placeholder, content)
        for part in parts:
            is_placeholder = part.startswith("{") and part.endswith("}")
            if not is_placeholder and part.strip():
                structure.append((False, len(all_segments_to_perturb)))
                all_segments_to_perturb.append(part)
            else:
                structure.append((True, part))
        prompt_maps.append(structure)
    
    flat_results, _, _ = batch_generation(
        model=model,
        system_messages=[system_msg] * len(all_segments_to_perturb),
        prompts=[instruction + text for text in all_segments_to_perturb],
        response_generation_method=[ResponseGenerationMethod()] * len(all_segments_to_perturb),
        max_tokens=1024
    )

    final_prompts = []
    for structure in prompt_maps:
        reconstructed = []
        for is_placeholder, content in structure:
            if is_placeholder:
                reconstructed.append(content)
            else:
                reconstructed.append(flat_results[content])
        final_prompts.append("".join(reconstructed))

    return final_prompts
     

def make_paraphrase(all_prompts: List[str], model: str, instruction: str) -> str:
    """
    Uses a language model to paraphrase the input text.
    Args:
        all_prompts (List[str]): The input prompts as a list to perturb.
        model (str): The language model to use for paraphrasing as a vllm LLM object.
        instruction (str): The instruction prompt for the model.
    Returns:
        List[str]: The paraphrased text as a list of strings.
    """
    system_msg = "You are a helpful assistant that paraphrases text while preserving the original meaning."
    
    all_segments_to_perturb = []
    prompt_maps = []

    for prompt in all_prompts:
        parts = re.split(r'(\{.*?\})', prompt)
        structure = [] # Stores (is_placeholder, content)
        for part in parts:
            is_placeholder = part.startswith("{") and part.endswith("}")
            if not is_placeholder and part.strip(): 
                structure.append((False, len(all_segments_to_perturb)))
                all_segments_to_perturb.append(part)
            else:
                structure.append((True, part))
        prompt_maps.append(structure)
    
    flat_results, _, _ = batch_generation(
        model=model,
        system_messages=[system_msg] * len(all_segments_to_perturb),
        prompts=[instruction + text for text in all_segments_to_perturb],
        response_generation_method=[ResponseGenerationMethod()] * len(all_segments_to_perturb),
        max_tokens=1024
    )

    final_prompts = []
    for structure in prompt_maps:
        reconstructed = []
        for is_placeholder, content in structure:
            if is_placeholder:
                reconstructed.append(content)
            else:
                reconstructed.append(flat_results[content])
        final_prompts.append("".join(reconstructed))

    return final_prompts



def apply_safe_perturbation(prompts: list, perturbation_func, **kwargs):
        """
        Splits list of prompts by curly brace placeholders (e.g., {PROMPT_OPTIONS}).
        Applies the perturbation_func ONLY to the prompts segments, protecting the keys.
        
        Args:
            prompts (List[str]): The input prompts containing placeholders.
            perturbation_func (function): The function to apply to non-placeholder text.
            **kwargs: Additional keyword arguments to pass to the perturbation function (e.g., probability).
        Returns:
            List[str]: The prompts with perturbations applied safely.
        """
        import re
        if not prompts:
            return prompts
        
        if perturbation_func in [make_synonyms, make_paraphrase]:
            print("Using batch perturbation function:", perturbation_func)
            if perturbation_func == make_synonyms:
                final_prompts = make_synonyms(
                    all_prompts=prompts,
                    model=kwargs.get("model"),
                    instruction=kwargs.get("instruction")
                )
            elif perturbation_func == make_paraphrase:
                final_prompts = make_paraphrase(
                    all_prompts=prompts,
                    model=kwargs.get("model"),
                    instruction=kwargs.get("instruction")
                )
            return final_prompts   
        else:
            perturbed_prompts = []
            for prompt in prompts:

                parts = re.split(r'(\{.*?\})', prompt)
                
                processed_parts = []
                for part in parts:
                    # Check if this part is a placeholder
                    if part.startswith("{") and part.endswith("}"):
                        # Append exactly as is
                        processed_parts.append(part)
                    else:
                        # Apply the typo function
                        processed_parts.append(perturbation_func(part, **kwargs))
                    
                perturbed_prompts.append("".join(processed_parts))

            return perturbed_prompts