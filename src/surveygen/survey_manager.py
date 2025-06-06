from typing import List, Dict, Optional, Union, Any, overload, Tuple, NamedTuple, Self, Literal
from dataclasses import dataclass, replace
from string import ascii_lowercase, ascii_uppercase

from .utilities.prompt_creation import PromptCreation
from .utilities.survey_objects import AnswerOptions, SurveyItem, QuestionLLMResponseTuple
from .utilities import constants

from .inference.survey_inference import batch_generation, batch_turn_by_turn_generation
from .inference.dynamic_pydantic import generate_pydantic_model

from vllm import LLM
from vllm.sampling_params import GuidedDecodingParams

import pandas as pd

import random

import copy

import tqdm

class SurveyOptionGenerator:
    #TODO This probably should be its own file instead. My Java programming took over :D
    """
    A class responsible for preparing optional answers for the LLM.
    """

    LIKERT_5: List[str] = ["disagree strongly", "disagree a little", "neither agree nor disagree", "agree a little", "agree strongly"]
    LIKERT_NO_MIDDLE: List[str] = ["disagree strongly", "disagree a little", "agree a little", "agree strongly"]

    LIKERT_IMPORTANCE_FROM_TO: List[str] = ["Not at all important", "Very Important"]
    LIKERT_JUSTIFIABLE_FROM_TO: List[str] = ["Never justifiable", "Always justifiable"]
    _IDX_TYPES = Literal["char_low", "char_upper", "integer"]

    @staticmethod
    def generate_likert_options(n: int, descriptions: Optional[List[str]], # update naming of description
                                only_from_to_scale:bool = False, 
                                random_order:bool = False, 
                                reversed_order:bool = False,
                                even_order:bool = False,
                                start_idx: int = 1,
                                idx_type: _IDX_TYPES="integer") -> AnswerOptions:
        
        if only_from_to_scale:
            assert len(descriptions) == 2, "If from to scale, provide exactly two descriptions"
            assert idx_type == "integer", "Index type must be integer, not lower/uppercase characters."
        else:
            if descriptions:
                assert len(descriptions) == n, "Description list must be the same length as options"
        if even_order:
                assert n % 2 != 0, "There must be a odd number of options!"
                middle_index = n // 2
                descriptions = descriptions[:middle_index] + descriptions[middle_index+1:]
                n = n-1
        if random_order:
            assert len(descriptions) >= 2, "There must be at least two answer options to reorder randomly."
            random.shuffle(descriptions) # no assignment needed because shuffles already inplace
        if reversed_order:
            assert len(answer_option) >= 2, "There must be at least two answer options to reverse options."
            answer_options = answer_options[::-1]
             

        answer_options = []
        if idx_type == "integer":
            for i in range(n):
                option_number = i + start_idx #rename answer options
                answer_option = f"{option_number}"
                if only_from_to_scale:
                    if i == 0:
                        answer_option = f"{option_number}: {descriptions[0]}"
                    elif i == (n-1):
                        answer_option = f"{option_number}: {descriptions[1]}"
                elif descriptions:
                    answer_option = f"{option_number}: {descriptions[i]}"
                answer_options.append(answer_option)
        else:
            #TODO @Jens add these to constants.py
            if idx_type == "char_low":
                for i in range(n):
                    answer_option = f"{ascii_lowercase[i]}: {descriptions[i]}"
                    answer_options.append(answer_option)
            elif idx_type == "char_upper":
                for i in range(n):
                    answer_option = f"{ascii_uppercase[i]}: {descriptions[i]}"
                    answer_options.append(answer_option)


        survey_option = AnswerOptions(answer_options, from_to_scale=only_from_to_scale)
        #print(survey_option)
        return survey_option
    
    # @staticmethod
    # def generate_interval_options(
    #         answer_options: List[str],
    #         only_from_to_scale:bool = False, 
    #         random_order:bool = False, 
    #         reversed_order:bool = False,
    #         even_order:bool = False,

    # ):
        
    
    @staticmethod
    def generate_generic_options(
                descriptions: Dict,
                only_from_to_scale:bool = False, 
                random_order:bool = False, 
                reversed_order:bool = False,
                even_order:bool = False,
                to_lowercase:bool = False,
                to_uppercase:bool = False,
                to_integer:bool = False,
                ):
            
            n = len(descriptions.values())
            answer_codes = descriptions.keys()
            answer_texts = descriptions.values()
            #answer_options = descriptions

            if to_lowercase:
                if all(isinstance(item, int) for item in answer_codes):
                    new_codes = []
                    for i in answer_codes:
                        code = ascii_lowercase[i-1]
                        new_codes.append(code)
                    answer_codes = new_codes 
                else:
                    answer_codes = [s.lower() for s in answer_codes]
            if to_uppercase:
                if all(isinstance(item, int) for item in answer_codes):
                    new_codes = []
                    for i in answer_codes:
                        code = ascii_uppercase[i-1]
                        new_codes.append(code)
                    answer_codes = new_codes 
                else:
                    answer_codes = [s.upper() for s in answer_codes]
            if to_integer:
                answer_codes = range(1,len(answer_codes)+1)
            
            answer_options = dict(zip(answer_codes, answer_texts))

            if only_from_to_scale:
                assert all(isinstance(item, int) for item in answer_codes), "To use from-to scale you must have integer answer codes."
                    
            if random_order:
                assert n >= 2, "There must be at least two answer options to reorder randomly."
                temp = list(answer_texts)
                random.shuffle(temp)
                # reassigning to keys
                answer_options = dict(zip(answer_codes, temp))
            if reversed_order:
                assert n >= 2, "There must be at least two answer options to reverse options."
                reversed_values = list(answer_texts)[::-1]
                answer_options = dict(zip(answer_codes, reversed_values))
            if even_order:
                assert n % 2 != 0, "There must be a odd number of options!"
                middle_index = n // 2
                # Get the key of the item to be removed
                key_to_remove = list(answer_codes)[middle_index]
                # Create a new dictionary, excluding the item with key_to_remove
                # This uses a dictionary comprehension.
                answer_options = {key: value for key, value in answer_options.items() if key != key_to_remove}
                if all(isinstance(element, int) for element in list(answer_codes)):
                    first_part = list(answer_codes)[:middle_index]
                    last_part = list(answer_codes)[middle_index+1:]
                    last_part = [x - 1 for x in last_part]            
                    answer_options = dict(zip(first_part + last_part, answer_options.values()))
                elif set(list(answer_codes)).issubset(list(ascii_lowercase)):
                    first_part = list(answer_codes)[:middle_index]
                    #print("First part:", first_part)
                    last_part = list(answer_codes)[middle_index+1:]
                    #print("Last part:", last_part)
                    last_parts = []
                    for i in range(middle_index+1,len(list(answer_codes))):
                        part = list(ascii_lowercase)[i-1]
                        last_parts.append(part) 
                        #print("Last parts:", last_parts)           
                    answer_options = dict(zip(first_part + last_parts, answer_options.values()))
                elif set(list(answer_codes)).issubset(list(ascii_uppercase)):
                    first_part = list(answer_codes)[:middle_index]
                    #print("First part:", first_part)
                    last_part = list(answer_codes)[middle_index+1:]
                    #print("Last part:", last_part)
                    last_parts = []
                    for i in range(middle_index+1,len(list(answer_codes))):
                        part = list(ascii_uppercase)[i-1]
                        last_parts.append(part) 
                        #print("Last parts:", last_parts)           
                    answer_options = dict(zip(first_part + last_parts, answer_options.values()))

            answer_options = [f"{key}: {val}" for key, val in answer_options.items()]        
            print(answer_options)   

            survey_option = AnswerOptions(answer_options, from_to_scale=only_from_to_scale)
            #print(survey_option)
            return survey_option

@dataclass
class InferenceOptions:
    system_prompt: str
    task_instruction: str
    question_prompts: Dict[int, str]
    guided_decodings: Optional[Dict[int, GuidedDecodingParams]]
    full_guided_decoding: Optional[GuidedDecodingParams]
    json_structure: Optional[List[str]]
    full_json_structure: Optional[List[str]]
    order: List[int]

    def create_single_question(self, question_id: int) -> str:
        return f"""{self.task_instruction} 
{self.question_prompts[question_id]}""".strip()
    
    def create_all_questions(self) -> str:
        default_prompt = f"{self.task_instruction}"
        all_questions_prompt = ""
        for question_prompt in self.question_prompts.values():
            all_questions_prompt = f"{all_questions_prompt}\n{question_prompt}"
        if len(default_prompt) > 0:
            all_prompt = f"{default_prompt.strip()}\n{all_questions_prompt.strip()}"
        else:
            all_prompt = all_questions_prompt.strip()
        return all_prompt

    def json_system_prompt(self, json_options:List[str]) -> str:
        creator = PromptCreation()
        creator.set_ouput_format_json(json_attributes=json_options, json_explanation=None)
        json_appendix = creator.get_output_prompt()

        system_prompt = f"""{self.system_prompt}
{json_appendix}"""
        return system_prompt


DEFAULT_SYSTEM_PROMPT: str = "You will be given questions and possible answer options for each. Please reason about each question before answering."
DEFAULT_TASK_INSTRUCTION: str = ""

DEFAULT_JSON_STRUCTURE: List[str] = ["reasoning", "answer"]

DEFAULT_SURVEY_ID: str = "Survey"

class LLMSurvey:
    """
    A class responsible for preparing and conducting surveys on LLMs.
    """
    DEFAULT_SYSTEM_PROMPT: str = "You will be given questions and possible answer options for each. Please reason about each question before answering."
    DEFAULT_TASK_INSTRUCTION: str = ""

    DEFAULT_JSON_STRUCTURE: List[str] = ["reasoning", "answer"]

    def __init__(self, survey_path: str, survey_name:str = DEFAULT_SURVEY_ID,  system_prompt: str = DEFAULT_SYSTEM_PROMPT, task_instruction: str = DEFAULT_TASK_INSTRUCTION, verbose=False, seed:int = 42):
        random.seed(seed)
        self.load_survey(survey_path=survey_path)
        self.verbose: bool = verbose

        self.survey_name: str = survey_name

        self.system_prompt: str = system_prompt 
        self.task_instruction: str = task_instruction

        self._global_options: AnswerOptions = None

    def duplicate(self):
        return copy.deepcopy(self)

    def get_survey_questions(self) -> str:
        return self._questions

    def load_survey(self, survey_path: str) -> Self:
        """
        Loads a prepared survey in csv format from a path.

        Currently csv files need to have the structure:
        question_id, survey_question
        1, question1

        :param survey_path: Path to the survey to load.
        :return: List of Survey Questions
        """
        survey_questions: List[SurveyItem] = []

        df = pd.read_csv(survey_path)

        for _ , row in df.iterrows():
            survey_item_id = row[constants.SURVEY_ITEM_ID]
            survey_question_content = row[constants.QUESTION_CONTENT]

            generated_survey_question = SurveyItem(item_id=survey_item_id, question_content=survey_question_content)
            survey_questions.append(generated_survey_question)

        self._questions = survey_questions
        return self
    

    #TODO Item order could be given by ids
    @overload
    def prepare_survey(self, question_stem: Optional[str] = "Do you see yourself as someone who...", 
                       answer_options: Optional[AnswerOptions] = None, global_options:bool = False,
                       prefilled_responses: Optional[Dict[int, str]] = None, randomized_item_order:bool = False) -> Self: ...

    @overload
    def prepare_survey(self, question_stem: Optional[List[str]] = ["Do you see yourself as someone who..."], 
                       answer_options: Optional[Dict[int, AnswerOptions]] = None, global_options:bool = False,
                       prefilled_responses: Optional[Dict[int, str]] = None, randomized_item_order:bool = False) -> Self: ...
    
    def prepare_survey(self, question_stem: Optional[Union[str, List[str]]] = "Do you see yourself as someone who...", 
                       answer_options: Optional[Union[AnswerOptions, Dict[int, AnswerOptions]]] = None, global_options:bool = False,
                       prefilled_responses: Optional[Dict[int, str]] = None, randomized_item_order:bool = False) -> Self:
        """
        Prepares a survey with additional prompts for each question, answer options and prefilled answers.

        :param prompt: Either one prompt for each question, or a list of different questions. Needs to have the same amount of prompts as the survey questions.
        :param options: Either the same Survey Options for all questions, or a dictionary linking the question id to the desired survey options.
        :para, prefilled_answers Linking survey question id to a prefilled answer.
        :return: List of updated Survey Questions
        """
        survey_questions: List[SurveyItem] = self._questions

        prompt_list = isinstance(question_stem, list)
        if prompt_list:
            assert len(question_stem) == len(survey_questions), "If a list of question stems is given, length of prompt and survey questions have to be the same" 
        
        options_dict = False

        if isinstance(answer_options, AnswerOptions):
            options_dict = False
            if global_options:
                self._global_options = answer_options
        elif isinstance(answer_options, Dict):
            options_dict = True

        updated_questions: List[SurveyItem] = []

        if not prefilled_responses:
            prefilled_responses = {}
            #for survey_question in survey_questions:
                #prefilled_answers[survey_question.question_id] = None

        if not prompt_list and not options_dict:            
            updated_questions = []
            for i in range(len(survey_questions)):
                new_survey_question = replace(survey_questions[i], question_stem = question_stem, answer_options = answer_options if not self._global_options else None, 
                                              prefilled_response=prefilled_responses.get(survey_questions[i].item_id)) 
                updated_questions.append(new_survey_question)

        elif not prompt_list and options_dict:
            for i in range(len(survey_questions)):
                new_survey_question = replace(survey_questions[i], question_stem = question_stem, answer_options = answer_options.get(survey_questions[i].item_id), 
                                              prefilled_response=prefilled_responses.get(survey_questions[i].item_id)) 
                updated_questions.append(new_survey_question)

        elif prompt_list and not options_dict:
            for i in range(len(survey_questions)):
                new_survey_question = replace(survey_questions[i], question_stem = question_stem[i], answer_options = answer_options if not self._global_options else None, 
                                              prefilled_response=prefilled_responses.get(survey_questions[i].item_id)) 
                updated_questions.append(new_survey_question)
        elif prompt_list and options_dict:
            for i in range(len(survey_questions)):
                new_survey_question = replace(survey_questions[i], question_stem = question_stem[i], answer_options = answer_options.get(survey_questions[i].item_id), 
                                              prefilled_response=prefilled_responses.get(survey_questions[i].item_id)) 
                updated_questions.append(new_survey_question)
        
        if randomized_item_order:
            random.shuffle(updated_questions)

        self._questions = updated_questions
        return self

    def generate_question_prompt(self, survey_question: SurveyItem) -> str:
        """
        Returns the string of how a survey question would be prompted to the model.

        :param survey_question: Survey question to prompt.
        :return: Prompt that will be given to the model for this question.
        """
        if constants.QUESTION_CONTENT_PLACEHOLDER in survey_question.question_stem:
            question_prompt = survey_question.question_stem.format(**{constants.QUESTION_CONTENT_PLACEHOLDER: survey_question.question_content})
        else:
            question_prompt = f"""{survey_question.question_stem} {survey_question.question_content}"""

        if survey_question.answer_options:
            options_prompt = survey_question.answer_options.create_options_str()
            question_prompt = f"""{question_prompt} 
{options_prompt}"""

        return question_prompt
    
    def _generate_inference_options(self, json_structured_output:bool=False, json_structure: List[str] = DEFAULT_JSON_STRUCTURE):
        survey_questions = self._questions

        default_prompt = f"""{self.task_instruction}"""

        if self._global_options:
            options_prompt = self._global_options.create_options_str()
            if len(default_prompt) > 0:
                default_prompt = f"""{default_prompt} 
{options_prompt}"""
            else:
                default_prompt = options_prompt

        question_prompts = {}

        guided_decoding_params = None
        extended_json_structure: List[str] = None
        json_list: List[str] = None

        order = []

        if json_structured_output:
            guided_decoding_params = {}
            extended_json_structure = []
            json_list = json_structure

        full_guided_decoding_params = None

        constraints: Dict[str, List[str]] =  {}

        for i, survey_question in enumerate(survey_questions):
            question_prompt = self.generate_question_prompt(survey_question=survey_question)
            question_prompts[survey_question.item_id] = question_prompt

            order.append(survey_question.item_id)
            
            guided_decoding = None
            if json_structured_output:

                for element in json_structure:
                    extended_json_structure.append(f"{element}{i+1}")
                    if element == json_structure[-1]:
                        if survey_question.answer_options:
                            constraints[f"{element}{i+1}"] = survey_question.answer_options.answer_text

                single_constraints = {}
                if survey_question.answer_options:
                    single_constraints = {json_structure[-1]: survey_question.answer_options.answer_text}
    
                pydantic_model = generate_pydantic_model(fields=json_structure, constraints=single_constraints)
                json_schema = pydantic_model.model_json_schema()
                guided_decoding = GuidedDecodingParams(json=json_schema)
                guided_decoding_params[survey_question.item_id] = guided_decoding
        
        if json_structured_output:
            pydantic_model = generate_pydantic_model(fields=extended_json_structure, constraints=constraints)
            full_json_schema = pydantic_model.model_json_schema()
            full_guided_decoding_params = GuidedDecodingParams(json=full_json_schema)

        return InferenceOptions(self.system_prompt, default_prompt, question_prompts, 
                                guided_decoding_params, full_guided_decoding_params, 
                                json_list, extended_json_structure,
                                order)


@dataclass
class SurveyResult():
    survey: LLMSurvey
    results: Dict[int, QuestionLLMResponseTuple]

    def to_dataframe(self) -> pd.DataFrame:
        answers = []
        for item_id, question_llm_response_tuple in self.results.items():
            answers.append((item_id, *question_llm_response_tuple))
        return pd.DataFrame(answers, columns=[constants.SURVEY_ITEM_ID, *question_llm_response_tuple._fields])


def conduct_survey_question_by_question(model: LLM, surveys: List[LLMSurvey], json_structured_output:bool=False, json_structure: List[str] = DEFAULT_JSON_STRUCTURE, print_conversation:bool=False, print_progress: bool = True, seed:int = 42, **generation_kwargs: Any) -> List[SurveyResult]:
    """
    Conducts the survey with each question in a new context.

    :param model: LLM instance of vllm.
    :param system_prompt: The system prompt of the model.
    :param task_instruction: The task instructio the model will be prompted with.
    :param json_structured_output: If json_structured output should be used.
    :param json_structure: The structure the final ouput should have.
    :param batch_size: How many inferences should run in parallel.
    :param print_conversation: If True, the whole conversation will be printed.
    :param generation_kwargs: All keywords needed for SamplingParams.
    :return: Generated text by the LLM in double list format
    """
    inference_options: List[InferenceOptions] = []

    max_survey_length: int = 0
    
    question_llm_response_pairs: List[Dict[int, QuestionLLMResponseTuple]] = []
    
    for survey in surveys:
        inference_option = survey._generate_inference_options(json_structured_output, json_structure)
        inference_options.append(inference_option)
        survey_length = len(inference_option.order)
        if survey_length > max_survey_length:
            max_survey_length = survey_length

        question_llm_response_pairs.append({})

    survey_results: List[SurveyResult] = []

    for i in (tqdm.tqdm(range(max_survey_length)) if print_progress else range(max_survey_length)):
        current_batch = [inference_option for inference_option in inference_options if len(inference_option.order) > i]

        if json_structured_output:
            system_messages = [inference.json_system_prompt(json_options=json_structure) for inference in current_batch]
        else:
            system_messages = [inference.system_prompt for inference in current_batch]
        prompts = [inference.create_single_question(inference.order[i]) for inference in current_batch]
        guided_decoding_params = [inference.guided_decodings[inference.order[i]] for inference in current_batch if inference.guided_decodings]

        output = batch_generation(model=model, 
                                    system_messages = system_messages, 
                                    prompts = prompts, 
                                    guided_decoding_params = guided_decoding_params, 
                                    print_conversation=print_conversation, 
                                    print_progress=print_progress,
                                    seed=seed,
                                    **generation_kwargs)
        
        for survey_id, prompt, answer, item in zip(range(len(current_batch)), prompts, output, current_batch):
            question_llm_response_pairs[survey_id].update({item.order[i]: QuestionLLMResponseTuple(prompt, answer)})
        
    for i, survey in enumerate(surveys):
        survey_results.append(SurveyResult(survey, question_llm_response_pairs[i]))

    return survey_results

def conduct_whole_survey_one_prompt(model: LLM, surveys: List[LLMSurvey], json_structured_output:bool=False, json_structure: List[str] = DEFAULT_JSON_STRUCTURE, print_conversation:bool=False, print_progress: bool = True, seed:int = 42, **generation_kwargs: Any) -> List[SurveyResult]:
    """
    Conducts the entire survey in one single LLM prompt.

    :param model: LLM instance of vllm.
    :param system_prompt: The system prompt of the model.
    :param task_instruction: The task instructio the model will be prompted with.
    :param json_structured_output: If json_structured output should be used.
    :param json_structure: The structure the final ouput should have.
    :param batch_size: How many inferences should run in parallel.
    :param print_conversation: If True, the whole conversation will be printed.
    :param generation_kwargs: All keywords needed for SamplingParams.
    :return: Generated text by the LLM in double list format
    """
    inference_options: List[InferenceOptions] = []

    #We always conduct the survey in one prompt
    max_survey_length: int = 1
    
    question_llm_response_pairs: List[Dict[int, QuestionLLMResponseTuple]] = []
    
    for survey in surveys:
        inference_option = survey._generate_inference_options(json_structured_output, json_structure)
        inference_options.append(inference_option)

        question_llm_response_pairs.append({})

    survey_results: List[SurveyResult] = []

    for i in (tqdm.tqdm(range(max_survey_length)) if print_progress else range(max_survey_length)):
        current_batch = [inference_option for inference_option in inference_options if len(inference_option.order) > i]

        if json_structured_output:
            system_messages = [inference.json_system_prompt(inference.full_json_structure) for inference in current_batch]
        else:
            system_messages = [inference.system_prompt for inference in current_batch]
        prompts = [inference.create_all_questions() for inference in current_batch]


        guided_decoding_params = [inference.full_guided_decoding for inference in current_batch if inference.full_guided_decoding]

        output = batch_generation(model=model, 
                                    system_messages = system_messages, 
                                    prompts = prompts, 
                                    guided_decoding_params = guided_decoding_params, 
                                    print_conversation=print_conversation, 
                                    print_progress=print_progress,
                                    seed=seed,
                                    **generation_kwargs)
        
        for survey_id, prompt, answer in zip(range(len(current_batch)), prompts, output):
            question_llm_response_pairs[survey_id].update({-1: QuestionLLMResponseTuple(prompt, answer)})
        
    for i, survey in enumerate(surveys):
        survey_results.append(SurveyResult(survey, question_llm_response_pairs[i]))

    return survey_results

def conduct_survey_in_context(model:LLM, surveys: List[LLMSurvey], json_structured_output:bool=False, json_structure: List[str] = DEFAULT_JSON_STRUCTURE, print_conversation:bool=False, print_progress: bool = True, seed:int = 42, **generation_kwargs: Any) -> List[SurveyResult]:
    """
    Conducts the entire survey multiple prompts but within the same context window.

    :param model: LLM instance of vllm.
    :param system_prompt: The system prompt of the model.
    :param task_instruction: The task instructio the model will be prompted with.
    :param json_structured_output: If json_structured output should be used.
    :param json_structure: The structure the final ouput should have.
    :param batch_size: How many inferences should run in parallel.
    :param print_conversation: If True, the whole conversation will be printed.
    :param generation_kwargs: All keywords needed for SamplingParams.
    :return: Generated text by the LLM in double list format
    """

    inference_options: List[InferenceOptions] = []

    max_survey_length: int = 0
    
    question_llm_response: List[Dict[int, QuestionLLMResponseTuple]] = []
    
    for survey in surveys:
        inference_option = survey._generate_inference_options(json_structured_output, json_structure)
        inference_options.append(inference_option)
        survey_length = len(inference_option.order)
        if survey_length > max_survey_length:
            max_survey_length = survey_length

        question_llm_response.append({})

    survey_results: List[SurveyResult] = []

    all_prompts: List[List[str]] = []
    assistant_messages: List[List[str]] = []

    for i in range(len(surveys)):
        assistant_messages.append([])
        all_prompts.append([])

    for i in (tqdm.tqdm(range(max_survey_length)) if print_progress else range(max_survey_length)):
        current_batch = [inference_option for inference_option in inference_options if len(inference_option.order) > i]
        current_surveys = [surv for surv in surveys if len(surv._questions) > i]

        prompts = [inference.create_single_question(inference.order[i]) for inference in current_batch]
        for c in range(len(current_surveys)):
            all_prompts[c].append(prompts[c])

        current_assistant_messages: List[str] = []

        missing_indeces = []

        for index, surv in enumerate(current_surveys):
            prefilled_answer = surv._questions[i].prefilled_response
            if prefilled_answer:                
                current_assistant_messages.append(prefilled_answer)
                missing_indeces.append(index)

        current_batch = [item for a, item in enumerate(current_batch) if a not in missing_indeces]

        if len(current_batch) == 0:
            for c in range(len(current_surveys)):
                assistant_messages[c].append(current_assistant_messages[c])
            for survey_id, prompt, llm_response, item in zip(range(len(current_surveys)), prompts, current_assistant_messages, current_surveys):
                question_llm_response[survey_id].update({item._questions[i].item_id: QuestionLLMResponseTuple(prompt, llm_response)})
            continue    
        if json_structured_output:
            system_messages = [inference.json_system_prompt(json_options=json_structure) for inference in current_batch]
        else:
            system_messages = [inference.system_prompt for inference in current_batch]

        guided_decoding_params = [inference.guided_decodings[inference.order[i]] for inference in current_batch if inference.guided_decodings]

        output = batch_turn_by_turn_generation(model=model, 
                                    system_messages = system_messages, 
                                    prompts = all_prompts,
                                    assistant_messages= assistant_messages,
                                    guided_decoding_params = guided_decoding_params, 
                                    print_conversation=print_conversation, 
                                    print_progress=print_progress,
                                    seed=seed,
                                    **generation_kwargs)
        
        for num, index in enumerate(missing_indeces):
            output.insert(index, current_assistant_messages[num])
        for survey_id, prompt, llm_response, item in zip(range(len(current_surveys)), prompts, output, current_surveys):
            question_llm_response[survey_id].update({item._questions[i].item_id: QuestionLLMResponseTuple(prompt, llm_response)})
        assistant_messages.append(output)    
        
    for i, survey in enumerate(surveys):
        survey_results.append(SurveyResult(survey, question_llm_response[i]))

    return survey_results


if __name__ == "__main__":
    pass
    