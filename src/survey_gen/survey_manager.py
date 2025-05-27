from typing import List, Dict, Optional, Union, Any, overload, Tuple, NamedTuple, Self
from dataclasses import dataclass, replace

from .utilities.prompt_creation import PromptCreation
from .utilities.survey_classes.survey_objects import SurveyOptions, SurveyQuestion, QuestionAnswerTuple

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
    LIKTERT_JUSTIFIABLE_FROM_TO: List[str] = ["Never justifiable", "Always justifiable"]

    @staticmethod
    def generate_likert_options(n: int, descriptions: Optional[List[str]], 
                                only_from_to_scale:bool = False, 
                                random_order:bool = False, 
                                reversed_order:bool = False,
                                even_order:bool = False,
                                start_idx: int = 1) -> SurveyOptions:
        
        if only_from_to_scale:
            assert len(descriptions) == 2, "If from to scale, provide exactly two descriptions"
        else:
            if descriptions:
                assert len(descriptions) == n, "Description list must be the same length as options"

        answer_options = []
        for i in range(n):
            option_number = i + start_idx
            answer_option = f"{option_number}"
            if only_from_to_scale:
                if i == 0:
                    answer_option = f"{option_number}: {descriptions[0]}"
                elif i == (n-1):
                    answer_option = f"{option_number}: {descriptions[1]}"
            elif descriptions:
                answer_option = f"{option_number}: {descriptions[i]}"
            answer_options.append(answer_option)

        if random_order:
            assert len(answer_option) >= 2, "There must be at least two answer options to reorder randomly."
            random.shuffle(answer_options) # no asignment needed because shuffles already inplace
        if reversed_order:
            assert len(answer_option) >= 2, "There must be at least two answer options to reverse options."
            answer_options = answer_options[::-1]
        if even_order:
            assert len(answer_option) % 2 != 0, "There must be a odd number of options!"
            middle_index = len(answer_option) // 2
            answer_option.pop(middle_index)


        survey_option = SurveyOptions(answer_options, from_to_scale=only_from_to_scale)
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
        return f"{self.task_instruction} {self.question_prompts[question_id]}".strip()
    
    def create_all_questions(self) -> str:
        default_prompt = f"{self.task_instruction}"
        all_questions_prompt = ""
        for question_prompt in self.question_prompts.values():
            all_questions_prompt = f"{all_questions_prompt}\n{question_prompt}"
        
        all_prompt = default_prompt.strip() + all_questions_prompt.strip()
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

class LLMSurvey:
    """
    A class responsible for preparing and conducting surveys on LLMs.
    """
    DEFAULT_SYSTEM_PROMPT: str = "You will be given questions and possible answer options for each. Please reason about each question before answering."
    DEFAULT_TASK_INSTRUCTION: str = ""

    DEFAULT_JSON_STRUCTURE: List[str] = ["reasoning", "answer"]

    def __init__(self, survey_path: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, task_instruction: str = DEFAULT_TASK_INSTRUCTION, verbose=False):
        #random.seed(seed)
        self.load_survey(survey_path=survey_path)
        self.verbose = verbose

        self.system_prompt: str = system_prompt 
        self.task_instruction: str = task_instruction

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
        survey_questions: List[SurveyQuestion] = []

        df = pd.read_csv(survey_path)

        for _ , row in df.iterrows():
            question_id = row["question_id"]
            survey_question = row["survey_question"]

            generated_survey_question = SurveyQuestion(question_id=question_id, survey_question=survey_question)
            survey_questions.append(generated_survey_question)

        self._questions = survey_questions
        return self
    
    @overload
    def prepare_survey(self, prompt: Optional[str] = "Do you see yourself as someone who...", 
                       options: Optional[SurveyOptions] = None, 
                       prefilled_answers: Optional[Dict[int, str]] = None) -> Self: ...

    @overload
    def prepare_survey(self, prompt: Optional[List[str]] = ["Do you see yourself as someone who..."], 
                       options: Optional[Dict[int, SurveyOptions]] = None, 
                       prefilled_answers: Optional[Dict[int, str]] = None) -> Self: ...
    
    def prepare_survey(self, prompt: Optional[Union[str, List[str]]] = "Do you see yourself as someone who...", 
                       options: Optional[Union[SurveyOptions, Dict[int, SurveyOptions]]] = None, 
                       prefilled_answers: Optional[Dict[int, str]] = None) -> Self:
        """
        Prepares a survey with additional prompts for each question, answer options and prefilled answers.

        :param prompt: Either one prompt for each question, or a list of different questions. Needs to have the same amount of prompts as the survey questions.
        :param options: Either the same Survey Options for all questions, or a dictionary linking the question id to the desired survey options.
        :para, prefilled_answers Linking survey question id to a prefilled answer.
        :return: List of updated Survey Questions
        """
        survey_questions = self._questions

        prompt_list = isinstance(prompt, list)
        if prompt_list:
            assert len(prompt) == len(survey_questions), "If a list of prompts is given, length of prompt and survey questions have to be the same" 
        
        options_dict = False

        if isinstance(options, SurveyOptions):
            options_dict = False
        elif isinstance(options, Dict):
            options_dict = True

        updated_questions = []

        if not prefilled_answers:
            prefilled_answers = {}
            #for survey_question in survey_questions:
                #prefilled_answers[survey_question.question_id] = None

        if not prompt_list and not options_dict:            
            updated_questions = []
            for i in range(len(survey_questions)):
                new_survey_question = replace(survey_questions[i], prompt = prompt, options = options, 
                                              prefilled_answer=prefilled_answers.get(survey_questions[i].question_id)) 
                updated_questions.append(new_survey_question)

        elif not prompt_list and options_dict:
            for i in range(len(survey_questions)):
                new_survey_question = replace(survey_questions[i], prompt = prompt, options = options.get(survey_questions[i].question_id), 
                                              prefilled_answer=prefilled_answers.get(survey_questions[i].question_id)) 
                updated_questions.append(new_survey_question)

        elif prompt_list and not options_dict:
            for i in range(len(survey_questions)):
                new_survey_question = replace(survey_questions[i], prompt = prompt[i], options = options, 
                                              prefilled_answer=prefilled_answers.get(survey_questions[i].question_id)) 
                updated_questions.append(new_survey_question)
        elif prompt_list and options_dict:
            for i in range(len(survey_questions)):
                new_survey_question = replace(survey_questions[i], prompt = prompt[i], options = options.get(survey_questions[i].question_id), 
                                              prefilled_answer=prefilled_answers.get(survey_questions[i].question_id)) 
                updated_questions.append(new_survey_question)
                    
        self._questions = updated_questions
        return self

    def generate_question_prompt(self, survey_question: SurveyQuestion) -> str:
        """
        Returns the string of how a survey question would be prompted to the model.

        :param survey_question: Survey question to prompt.
        :return: Prompt that will be given to the model for this question.
        """
        if survey_question.options:
            options_prompt = survey_question.options.create_options_str()
        else:
            options_prompt = ""

        survey_prompt = f"""{survey_question.prompt} {survey_question.survey_question}
{options_prompt}"""
        return survey_prompt
    
    def _generate_inference_options(self, json_structured_output:bool=False, json_structure: List[str] = DEFAULT_JSON_STRUCTURE):
        survey_questions = self._questions

        default_prompt = f"""{self.task_instruction}"""

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
            question_prompts[survey_question.question_id] = question_prompt

            order.append(survey_question.question_id)
            
            guided_decoding = None
            if json_structured_output:

                for element in json_structure:
                    extended_json_structure.append(f"{element}{i+1}")
                    if element == json_structure[-1]:
                        if survey_question.options:
                            constraints[f"{element}{i+1}"] = survey_question.options.option_descriptions

                single_constraints = {}
                if survey_question.options:
                    single_constraints = {json_structure[-1]: survey_question.options.option_descriptions}
    
                pydantic_model = generate_pydantic_model(fields=json_structure, constraints=single_constraints)
                json_schema = pydantic_model.model_json_schema()
                guided_decoding = GuidedDecodingParams(json=json_schema)
                guided_decoding_params[survey_question.question_id] = guided_decoding
        
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
    results: Dict[int, QuestionAnswerTuple]


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
    
    question_answer_pairs: List[Dict[int, QuestionAnswerTuple]] = []
    
    for survey in surveys:
        inference_option = survey._generate_inference_options(json_structured_output, json_structure)
        inference_options.append(inference_option)
        survey_length = len(inference_option.order)
        if survey_length > max_survey_length:
            max_survey_length = survey_length

        question_answer_pairs.append({})

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
                                    verbose=print_conversation, 
                                    seed=seed,
                                    **generation_kwargs)
        
        for survey_id, prompt, answer, item in zip(range(len(current_batch)), prompts, output, current_batch):
            question_answer_pairs[survey_id].update({item.order[i]: QuestionAnswerTuple(prompt, answer)})
        
    for i, survey in enumerate(surveys):
        survey_results.append(SurveyResult(survey, question_answer_pairs[i]))

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

    max_survey_length: int = 1
    
    question_answer_pairs: List[Dict[int, QuestionAnswerTuple]] = []
    
    for survey in surveys:
        inference_option = survey._generate_inference_options(json_structured_output, json_structure)
        inference_options.append(inference_option)

        question_answer_pairs.append({})

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
                                    verbose=print_conversation, 
                                    seed=seed,
                                    **generation_kwargs)
        
        for survey_id, prompt, answer in zip(range(len(current_batch)), prompts, output):
            question_answer_pairs[survey_id].update({-1: QuestionAnswerTuple(prompt, answer)})
        
    for i, survey in enumerate(surveys):
        survey_results.append(SurveyResult(survey, question_answer_pairs[i]))

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
    
    question_answer_pairs: List[Dict[int, QuestionAnswerTuple]] = []
    
    for survey in surveys:
        inference_option = survey._generate_inference_options(json_structured_output, json_structure)
        inference_options.append(inference_option)
        survey_length = len(inference_option.order)
        if survey_length > max_survey_length:
            max_survey_length = survey_length

        question_answer_pairs.append({})

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
            prefilled_answer = surv._questions[i].prefilled_answer
            if prefilled_answer:                
                current_assistant_messages.append(prefilled_answer)
                missing_indeces.append(index)

        current_batch = [item for a, item in enumerate(current_batch) if a not in missing_indeces]

        if len(current_batch) == 0:
            for c in range(len(current_surveys)):
                assistant_messages[c].append(current_assistant_messages[c])
            for survey_id, prompt, answer, item in zip(range(len(current_surveys)), prompts, current_assistant_messages, current_surveys):
                question_answer_pairs[survey_id].update({item._questions[i].question_id: QuestionAnswerTuple(prompt, answer)})
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
                                    verbose=print_conversation, 
                                    seed=seed,
                                    **generation_kwargs)
        
        for num, index in enumerate(missing_indeces):
            output.insert(index, current_assistant_messages[num])
        for survey_id, prompt, answer, item in zip(range(len(current_surveys)), prompts, output, current_surveys):
            question_answer_pairs[survey_id].update({item._questions[i].question_id: QuestionAnswerTuple(prompt, answer)})
        assistant_messages.append(output)    
        
    for i, survey in enumerate(surveys):
        survey_results.append(SurveyResult(survey, question_answer_pairs[i]))

    return survey_results


if __name__ == "__main__":
    pass
    