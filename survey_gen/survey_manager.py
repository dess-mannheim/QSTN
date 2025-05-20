from typing import List, Dict, Optional, Union, Any, overload
from dataclasses import dataclass, replace

from utilities.prompt_creation import PromptCreation
from utilities.survey_classes.survey_objects import SurveyOptions, SurveyQuestion

from parser.llm_answer_parser import LLMAnswerParser

from inference.survey_inference import batch_generation, batch_turn_by_turn_generation
from inference.dynamic_pydantic import generate_pydantic_model

from vllm import LLM
from vllm.sampling_params import GuidedDecodingParams

import pandas as pd

import random

class SurveyOptionGenerator:
    """
    A class responsible for preparing optional answers for the LLM.
    """

    LIKERT_5: List[str] = ["disagree strongly", "disagree a little", "neither agree nor disagree", "agree a little", "agree strongly"]
    LIKERT_NO_MIDDLE: List[str] = ["disagree strongly", "disagree a little", "agree a little", "agree strongly"]

    LIKERT_IMPORTANCE_FROM_TO: List[str] = ["Not at all important", "Very Important"]
    LIKTERT_JUSTIFIABLE_FROM_TO: List[str] = ["Never justifiable", "Always justifiable"]

    @staticmethod
    def generate_likert_options(n: int, descriptions: Optional[List[str]], only_from_to_scale:bool = False, start_idx: int = 1) -> SurveyOptions:
        
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

        survey_option = SurveyOptions(answer_options, from_to_scale=only_from_to_scale)
        #print(survey_option)
        return survey_option

class LLMSurvey:
    """
    A class responsible for preparing and conducting surveys on LLMs.
    """
    DEFAULT_SYSTEM_PROMPT: str = "You will be given questions and possible answer options for each. Please reason about each question before answering."
    DEFAULT_TASK_INSTRUCTION: str = ""

    DEFAULT_JSON_STRUCTURE: List[str] = ["reasoning", "answer"]

    def __init__(self, survey_path: str, verbose=False, seed:int = 42):
        random.seed(42)
        self._survey: List[SurveyQuestion] = self.load_survey(survey_path=survey_path)
        self.verbose = verbose

    def get_survey_questions(self) -> str:
        return self._survey

    def load_survey(self, survey_path: str) -> List[SurveyQuestion]:
        """
        Loads a prepared survey by ID.

        :param survey_path: Path to the survey to load.
        :param survey_id: How the survey should be accessed for further calls.
        :return: List of Survey Questions
        """
        survey_questions: List[SurveyQuestion] = []

        df = pd.read_csv(survey_path)

        for _ , row in df.iterrows():
            question_id = row["question_id"]
            survey_question = row["survey_question"]

            generated_survey_question = SurveyQuestion(question_id=question_id, survey_question=survey_question)
            survey_questions.append(generated_survey_question)

        self._survey = survey_questions
        return survey_questions
    
    @overload
    def prepare_survey(self, prompt: Optional[str] = "Do you see yourself as someone who...", 
                       options: Optional[SurveyOptions] = None, 
                       prefilled_answers: Optional[Dict[int, str]] = None) -> List[SurveyQuestion]: ...

    @overload
    def prepare_survey(self, prompt: Optional[List[str]] = ["Do you see yourself as someone who..."], 
                       options: Optional[List[SurveyOptions]] = None, 
                       prefilled_answers: Optional[Dict[int, str]] = None) -> List[SurveyQuestion]: ...
    
    def prepare_survey(self, prompt: Optional[Union[str, List[str]]] = "Do you see yourself as someone who...", 
                       options: Optional[Union[SurveyOptions, List[SurveyOptions]]] = None, 
                       prefilled_answers: Optional[Dict[int, str]] = None) -> List[SurveyQuestion]:
        survey_questions = self._survey

        prompt_list = isinstance(prompt, list)
        if prompt_list:
            assert len(prompt) == len(survey_questions), "If a list of prompts is given, length of prompt and survey questions have to be the same" 
        
        options_double_list = False

        if isinstance(options, SurveyOptions):
            options_double_list = False
        elif isinstance(options, list):
            options_double_list = True
    
        if options_double_list:
            assert len(options) == len(survey_questions), "If a list of options is given, length of options and survey questions have to be the same" 

        updated_questions = []

        if not prefilled_answers:
            prefilled_answers = {}
            #for survey_question in survey_questions:
                #prefilled_answers[survey_question.question_id] = None

        if not prompt_list and not options_double_list:            
            updated_questions = []
            for i in range(len(survey_questions)):
                new_survey_question = replace(survey_questions[i], prompt = prompt, options = options, 
                                              prefilled_answer=prefilled_answers.get(survey_questions[i].question_id)) 
                updated_questions.append(new_survey_question)

        elif not prompt_list and options_double_list:
            for i in range(len(survey_questions)):
                new_survey_question = replace(survey_questions[i], prompt = prompt, options = options[i], 
                                              prefilled_answer=prefilled_answers.get(survey_questions[i].question_id)) 
                updated_questions.append(new_survey_question)

        elif prompt_list and not options_double_list:
            for i in range(len(survey_questions)):
                new_survey_question = replace(survey_questions[i], prompt = prompt[i], options = options, 
                                              prefilled_answer=prefilled_answers.get(survey_questions[i].question_id)) 
                updated_questions.append(new_survey_question)
        elif prompt_list and options_double_list:
            for i in range(len(survey_questions)):
                new_survey_question = replace(survey_questions[i], prompt = prompt[i], options = options[i], 
                                              prefilled_answer=prefilled_answers.get(survey_questions[i].question_id)) 
                updated_questions.append(new_survey_question)
                    
        self._survey = updated_questions
        return updated_questions

    def generate_survey_prompt(self, survey_question: SurveyQuestion) -> str:
        options_prompt = survey_question.options.create_options_str()

        survey_prompt = f"""{survey_question.prompt} {survey_question.survey_question}
{options_prompt}"""
        return survey_prompt

    def conduct_survey_in_context(self, model:LLM, system_prompt:str=DEFAULT_SYSTEM_PROMPT, task_instruction: str = DEFAULT_TASK_INSTRUCTION, json_structured_output:bool=False, json_structure: List[str] = DEFAULT_JSON_STRUCTURE, batch_size:int=1, print_conversation:bool=False, **generation_kwargs: Any) -> List[str]:
        """
        Conducts the entire survey in a single LLM context window.

        :param survey_id: Identifier of the survey to conduct.
        :return: Mapping of question IDs to answers.
        """
        survey_questions = self._survey
        
        default_prompt = f"""{task_instruction}"""

        #model = BasicLLama(model_id)

        answers = []

        assistant_messages: List[List[str]] = [[]]
        while len(assistant_messages) <= batch_size:
            assistant_messages.append([])
        
        prompts: List[List[str]] = [[]]
        while len(prompts) <= batch_size:
            prompts.append([])
        for survey_question in survey_questions:
            survey_prompt = self.generate_survey_prompt(survey_question=survey_question)
            full_prompt = default_prompt + " " + survey_prompt
            full_prompt = full_prompt.strip()
            for i in range(batch_size):
                prompts[i].append(full_prompt)
            
            if survey_question.prefilled_answer:
                batch_prefilled_answers = []

                for i in range(batch_size):
                    assistant_messages[i].append(survey_question.prefilled_answer)
                    batch_prefilled_answers.append(survey_question.prefilled_answer)
                answers.append(batch_prefilled_answers)
                continue

            if json_structured_output:
                creator = PromptCreation()
                creator.set_ouput_format_json(json_attributes=json_structure, json_explanation=None)
                json_appendix = creator.get_output_prompt()

                json_system_prompt = f"""{system_prompt}
{json_appendix}"""
                
                #print(json_system_prompt)
                constraints = {"answer": survey_question.options.option_descriptions}
                pydantic_model = generate_pydantic_model(fields=json_structure, constraints=constraints)
                json_schema = pydantic_model.model_json_schema()
                guided_decoding = GuidedDecodingParams(json=json_schema)
                #print(json_system_prompt)
                output = batch_turn_by_turn_generation(model, system_messages=[json_system_prompt]*batch_size, prompts=prompts, assistant_messages=assistant_messages, guided_decoding_params=[guided_decoding]*batch_size, verbose=print_conversation ,**generation_kwargs)
            else:
                output = batch_turn_by_turn_generation(model, system_messages=[system_prompt]*batch_size, prompts=prompts, assistant_messages=assistant_messages, verbose=print_conversation, **generation_kwargs)
            for i in range(len(output)):
                assistant_messages[i].append(output[i])

            answers.append(output)

        #model.shutdown()

        return answers

    def conduct_survey_question_by_question(self, model: LLM, system_prompt:str=DEFAULT_SYSTEM_PROMPT, task_instruction: str = DEFAULT_TASK_INSTRUCTION, json_structured_output:bool=False, json_structure: List[str] = DEFAULT_JSON_STRUCTURE, batch_size:int=1, print_conversation:bool=False, **generation_kwargs: Any) -> List[List[str]]:
        """
        Conducts the survey one question at a time, each with a fresh LLM context.

        :param survey_id: Identifier of the survey to conduct.
        :return: Mapping of question IDs to answers.
        """
        survey_questions = self._survey
        
        #print(survey_questions)

        default_prompt = f"""{task_instruction}"""

        #model = BasicLLama(model_id)

        answers = []

        for survey_question in survey_questions:
            #print(survey_question)
            survey_prompt = self.generate_survey_prompt(survey_question=survey_question)
            full_prompt = default_prompt + " " + survey_prompt
            full_prompt = full_prompt.strip()
            #if print_prompts:
                #print(full_prompt, flush=True)
            
            if json_structured_output:
                creator = PromptCreation()
                creator.set_ouput_format_json(json_attributes=json_structure, json_explanation=None)
                json_appendix = creator.get_output_prompt()

                json_system_prompt = f"""{system_prompt}
{json_appendix}"""
                
                constraints = {"answer": survey_question.options.option_descriptions}
                pydantic_model = generate_pydantic_model(fields=json_structure, constraints=constraints)
                json_schema = pydantic_model.model_json_schema()
                guided_decoding = GuidedDecodingParams(json=json_schema)
                output = batch_generation(model=model, system_messages=[json_system_prompt]*batch_size, prompts=[full_prompt]*batch_size, guided_decoding_params=[guided_decoding]*batch_size, verbose=print_conversation, **generation_kwargs)
            else:
                output = batch_generation(model=model, system_messages=[system_prompt]*batch_size, prompts=[full_prompt]*batch_size, verbose=print_conversation, **generation_kwargs)
            answers.append(output)
        
        #model.shutdown()
        return answers

    def conduct_whole_survey_one_prompt(self, model: LLM, system_prompt:str=DEFAULT_SYSTEM_PROMPT, task_instruction: str=DEFAULT_TASK_INSTRUCTION, json_structured_output:bool=False, json_structure: List[str] = DEFAULT_JSON_STRUCTURE, batch_size:int=1, print_conversation:bool=False, **generation_kwargs: Any) -> List[str]:
        """
        Conducts a survey in one prompt.

        :param survey_id: Identifier of the survey to conduct.
        :return: Updated mapping of all question IDs to answers.
        """
        survey_questions = self._survey
        
        default_prompt = f"""{task_instruction}"""

        all_questions_prompt = ""
        question_number = 1
        extended_json_structure: List[str] = []
        constraints: Dict[str, List[str]] =  {}
        for i in range(len(survey_questions)):
            survey_prompt = self.generate_survey_prompt(survey_question=survey_questions[i])
            
            if json_structured_output:
                for element in json_structure:
                    extended_json_structure.append(f"{element}{i}")
                    if element == "answer":
                        constraints[f"{element}{i}"] = survey_questions[i].options.option_descriptions

            all_questions_prompt = all_questions_prompt + "\n" + survey_prompt
            #print(full_prompt, flush=True)
            question_number +=1

        full_prompt = default_prompt + all_questions_prompt
    	
        #print(full_prompt)

        if json_structured_output:
            creator = PromptCreation()

            creator.set_ouput_format_json(json_attributes=extended_json_structure, json_explanation=None)
            json_appendix = creator.get_output_prompt()

            json_system_prompt = f"""{system_prompt}
{json_appendix}"""
            
            #print(json_system_prompt)

            pydantic_model = generate_pydantic_model(fields=extended_json_structure, constraints=constraints)
            json_schema = pydantic_model.model_json_schema()
            guided_decoding = GuidedDecodingParams(json=json_schema)
            output = batch_generation(model=model, system_messages=[json_system_prompt]*batch_size, prompts=[full_prompt]*batch_size, guided_decoding_params=[guided_decoding]*batch_size, verbose=print_conversation,  **generation_kwargs)
        else:
            output = batch_generation(model=model, system_messages=[system_prompt]*batch_size, prompts=[full_prompt]*batch_size, verbose=print_conversation, **generation_kwargs)

        return output

if __name__ == "__main__":
    pass
    