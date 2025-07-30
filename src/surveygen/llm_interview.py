from typing import (
    List,
    Dict,
    Optional,
    Union,
    overload,
    Self,
)

from dataclasses import replace

from .utilities.survey_objects import AnswerOptions

from .utilities.survey_objects import AnswerOptions, InterviewItem, InferenceOptions

from .utilities import constants
from .utilities.constants import InterviewType

import pandas as pd

import random

import copy


from transformers import AutoTokenizer


class LLMInterview:
    """
    A class responsible for preparing and conducting surveys on LLMs.
    """

    DEFAULT_INTERVIEW_ID: str = "Interview"

    DEFAULT_SYSTEM_PROMPT: str = (
        "You will be given questions and possible answer options for each. Please reason about each question before answering."
    )
    DEFAULT_TASK_INSTRUCTION: str = ""

    DEFAULT_JSON_STRUCTURE: List[str] = ["reasoning", "answer"]

    def __init__(
        self,
        interview_path: str,
        interview_name: str = DEFAULT_INTERVIEW_ID,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        interview_instruction: str = DEFAULT_TASK_INSTRUCTION,
        verbose=False,
        seed: int = 42,
    ):
        random.seed(seed)
        self.load_interview_format(interview_path=interview_path)
        self.verbose: bool = verbose

        self.interview_name: str = interview_name

        self.system_prompt: str = system_prompt
        self.interview_instruction: str = interview_instruction

        self._global_options: AnswerOptions = None

    def duplicate(self):
        return copy.deepcopy(self)

    def get_prompt_structure(self) -> str:
        parts = [
            "SYSTEM PROMPT:",
            self.system_prompt,
            "INTERVIEW INSTRUCTIONS:",
            self.interview_instruction,
        ]

        if self._global_options:
            parts.append(self._global_options.create_options_str())

        parts.append("FIRST QUESTION:")
        parts.append(self.generate_question_prompt(self._questions[0]))
        
        return "\n".join(parts)

    def get_prompt_for_interview_type(self, interview_type: InterviewType = InterviewType.QUESTION):
        parts = [self.system_prompt, self.interview_instruction]

        if self._global_options:
            parts.append(self._global_options.create_options_str())

        if interview_type == InterviewType.QUESTION:
            parts.append(self.generate_question_prompt(self._questions[0]))
            
        elif interview_type in (InterviewType.ONE_PROMPT, InterviewType.CONTEXT):
            # Use extend to add all question strings from the generator
            parts.extend(
                self.generate_question_prompt(question) for question in self._questions
            )

        # Join all the collected parts with a newline
        whole_prompt = "\n".join(parts)

        return whole_prompt

    def calculate_input_token_estimate(
        self, model_id: str, interview_type: InterviewType = InterviewType.QUESTION
    ) -> int:
        """
        Calculates the input token estimate for different survey types. Remember that the model needs to
        have enough context length to also fit the output tokens. For SurveyType.CONTEXT the total input token estimate
        is just a very rough estimation. It depends on how many tokens the model produces in each response.

        :param model_id: The huggingface model id of the model you want to use.
        :param survey_type: The survey type you will be running.
        :return: Estimated number of input tokens needed for the survey
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        whole_prompt = self.get_prompt_for_interview_type(interview_type=interview_type)
        tokens = tokenizer.encode(whole_prompt)

        return (
            len(tokens) if interview_type != InterviewType.CONTEXT else len(tokens) * 3
        )

    def get_survey_questions(self) -> str:
        return self._questions

    def load_interview_format(self, interview_path: str) -> Self:
        """
        Loads a prepared survey in csv format from a path.

        Currently csv files need to have the structure:
        question_id, survey_question
        1, question1

        :param survey_path: Path to the survey to load.
        :return: List of Survey Questions
        """
        interview_questions: List[InterviewItem] = []

        df = pd.read_csv(interview_path)

        for _, row in df.iterrows():
            interview_item_id = row[constants.INTERVIEW_ITEM_ID]
            # if constants.QUESTION in df.columns:
            #     question = row[constants.QUESTION]
            if constants.QUESTION_CONTENT in df.columns:
                interview_question_content = row[constants.QUESTION_CONTENT]
            else:
                interview_question_content = None

            if constants.QUESTION_STEM in df.columns:
                interview_question_stem = row[constants.QUESTION_STEM]
            else:
                interview_question_stem = None

            generated_interview_question = InterviewItem(
                item_id=interview_item_id,
                question_content=interview_question_content,
                question_stem=interview_question_stem,
            )
            interview_questions.append(generated_interview_question)

        self._questions = interview_questions
        return self

    # TODO Item order could be given by ids
    @overload
    def prepare_interview(
        self,
        question_stem: Optional[str] = None,
        answer_options: Optional[AnswerOptions] = None,
        global_options: bool = False,
        prefilled_responses: Optional[Dict[int, str]] = None,
        randomized_item_order: bool = False,
    ) -> Self: ...

    @overload
    def prepare_interview(
        self,
        question_stem: Optional[List[str]] = None,
        answer_options: Optional[Dict[int, AnswerOptions]] = None,
        global_options: bool = False,
        prefilled_responses: Optional[Dict[int, str]] = None,
        randomized_item_order: bool = False,
    ) -> Self: ...

    def prepare_interview(
        self,
        question_stem: Optional[Union[str, List[str]]] = None,
        answer_options: Optional[Union[AnswerOptions, Dict[int, AnswerOptions]]] = None,
        global_options: bool = False,
        prefilled_responses: Optional[Dict[int, str]] = None,
        randomized_item_order: bool = False,
    ) -> Self:
        """
        Prepares a survey with additional prompts for each question, answer options and prefilled answers.

        :param prompt: Either one prompt for each question, or a list of different questions. Needs to have the same amount of prompts as the survey questions.
        :param options: Either the same Survey Options for all questions, or a dictionary linking the question id to the desired survey options.
        :para, prefilled_answers Linking survey question id to a prefilled answer.
        :return: List of updated Survey Questions
        """
        interview_questions: List[InterviewItem] = self._questions

        prompt_list = isinstance(question_stem, list)
        if prompt_list:
            assert len(question_stem) == len(
                interview_questions
            ), "If a list of question stems is given, length of prompt and survey questions have to be the same"

        options_dict = False

        if isinstance(answer_options, AnswerOptions):
            options_dict = False
            if global_options:
                self._global_options = answer_options
        elif isinstance(answer_options, Dict):
            options_dict = True

        updated_questions: List[InterviewItem] = []

        if not prefilled_responses:
            prefilled_responses = {}
            # for survey_question in survey_questions:
            # prefilled_answers[survey_question.question_id] = None

        if not prompt_list and not options_dict:
            updated_questions = []
            for i in range(len(interview_questions)):
                new_interview_question = replace(
                    interview_questions[i],
                    question_stem=(
                        question_stem
                        if question_stem
                        else interview_questions[i].question_stem
                    ),
                    answer_options=answer_options if not self._global_options else None,
                    prefilled_response=prefilled_responses.get(
                        interview_questions[i].item_id
                    ),
                )
                updated_questions.append(new_interview_question)

        elif not prompt_list and options_dict:
            for i in range(len(interview_questions)):
                new_interview_question = replace(
                    interview_questions[i],
                    question_stem=(
                        question_stem
                        if question_stem
                        else interview_questions[i].question_stem
                    ),
                    answer_options=answer_options.get(interview_questions[i].item_id),
                    prefilled_response=prefilled_responses.get(
                        interview_questions[i].item_id
                    ),
                )
                updated_questions.append(new_interview_question)

        elif prompt_list and not options_dict:
            for i in range(len(interview_questions)):
                new_interview_question = replace(
                    interview_questions[i],
                    question_stem=(
                        question_stem[i]
                        if question_stem
                        else interview_questions[i].question_stem
                    ),
                    answer_options=answer_options if not self._global_options else None,
                    prefilled_response=prefilled_responses.get(
                        interview_questions[i].item_id
                    ),
                )
                updated_questions.append(new_interview_question)
        elif prompt_list and options_dict:
            for i in range(len(interview_questions)):
                new_interview_question = replace(
                    interview_questions[i],
                    question_stem=(
                        question_stem[i]
                        if question_stem
                        else interview_questions[i].question_stem
                    ),
                    answer_options=answer_options.get(interview_questions[i].item_id),
                    prefilled_response=prefilled_responses.get(
                        interview_questions[i].item_id
                    ),
                )
                updated_questions.append(new_interview_question)

        if randomized_item_order:
            random.shuffle(updated_questions)

        self._questions = updated_questions
        return self

    def generate_question_prompt(self, interview_question: InterviewItem) -> str:
        """
        Returns the string of how a survey question would be prompted to the model.

        :param survey_question: Survey question to prompt.
        :return: Prompt that will be given to the model for this question.
        """
        if constants.QUESTION_CONTENT_PLACEHOLDER in interview_question.question_stem:
            question_prompt = interview_question.question_stem.format(
                **{
                    constants.QUESTION_CONTENT_PLACEHOLDER: interview_question.question_content
                }
            )
        else:
            question_prompt = f"""{interview_question.question_stem} {interview_question.question_content}"""

        if interview_question.answer_options:
            options_prompt = interview_question.answer_options.create_options_str()
            question_prompt = f"""{question_prompt} 
{options_prompt}"""

        return question_prompt

    def _generate_inference_options(
        self,
        # json_structured_output: bool = False,
        # json_structure: List[str] = DEFAULT_JSON_STRUCTURE,
        # json_force_answer: bool = False,
    ):
        interview_questions = self._questions

        default_prompt = f"""{self.interview_instruction}"""

        if self._global_options:
            options_prompt = self._global_options.create_options_str()
            if len(default_prompt) > 0:
                default_prompt = f"""{default_prompt} 
{options_prompt}"""
            else:
                default_prompt = options_prompt

        question_prompts = {}

        # guided_decoding_params = None
        # extended_json_structure: List[str] = None
        # json_list: List[str] = None

        order = []

        # if json_structured_output:
        #     guided_decoding_params = {}
        #     extended_json_structure = []
        #     json_list = json_structure

        # full_guided_decoding_params = None

        # constraints: Dict[str, List[str]] = {}

        answer_options = []
        for i, interview_question in enumerate(interview_questions):
            question_prompt = self.generate_question_prompt(
                interview_question=interview_question
            )
            question_prompts[interview_question.item_id] = question_prompt
            answer_options.append(interview_question.answer_options)
            order.append(interview_question.item_id)

            # guided_decoding = None
            # if json_structured_output:

            #     for element in json_structure:
            #         extended_json_structure.append(f"{element}{i+1}")
            #         if element == json_structure[-1]:
            #             if survey_question.answer_options:
            #                 constraints[f"{element}{i+1}"] = (
            #                     survey_question.answer_options.answer_text
            #                 )
            #             elif self._global_options:
            #                 constraints[f"{element}{i+1}"] = (
            #                     self._global_options.answer_text
            #                 )

            #     single_constraints = {}
            #     if survey_question.answer_options:
            #         single_constraints = {
            #             json_structure[-1]: survey_question.answer_options.answer_text
            #         }
            #     elif self._global_options:
            #         single_constraints = {
            #             json_structure[-1]: self._global_options.answer_text
            #         }
            #     pydantic_model = generate_pydantic_model(
            #         fields=json_structure, constraints=single_constraints
            #     )
            #     json_schema = pydantic_model.model_json_schema()
            #     guided_decoding = GuidedDecodingParams(json=json_schema)
            #     guided_decoding_params[survey_question.item_id] = guided_decoding

        # if json_structured_output:
        #     pydantic_model = generate_pydantic_model(
        #         fields=extended_json_structure,
        #         constraints=constraints if json_force_answer else None,
        #     )
        #     full_json_schema = pydantic_model.model_json_schema()
        #     full_guided_decoding_params = GuidedDecodingParams(json=full_json_schema)

        return InferenceOptions(
            system_prompt=self.system_prompt,
            task_instruction=default_prompt,
            question_prompts=question_prompts,
            # guided_decoding_params,
            # full_guided_decoding_params,
            # json_list,
            # extended_json_structure,
            answer_options=answer_options,
            order=order,
        )
