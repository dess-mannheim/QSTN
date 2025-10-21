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
    Main class for setting up and managing an LLM-based interview or survey.

    This class handles loading questions, preparing prompts, managing answer options,
    and generating prompt structures for different interview types.

    Usage example:
    --------------
    ```python
    interview = LLMInterview(interview_path="questions.csv")
    interview.prepare_interview(question_stem="Do you think {QUESTION_CONTENT_PLACEHOLDER} is good?", answer_options=AnswerOptions(...))
    prompt = interview.get_prompt_structure()
    print(prompt)
    ```
    """

    DEFAULT_INTERVIEW_ID: str = "Interview"

    DEFAULT_SYSTEM_PROMPT: str = (
        "You will be given questions and possible answer options for each. Please reason about each question before answering."
    )
    DEFAULT_TASK_INSTRUCTION: str = ""

    DEFAULT_JSON_STRUCTURE: List[str] = ["reasoning", "answer"]

    def __init__(
        self,
        interview_path: Optional[str] = None,
        interview_dataframe: Optional[pd.DataFrame] = None,
        interview_name: str = DEFAULT_INTERVIEW_ID,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        interview_instruction: str = DEFAULT_TASK_INSTRUCTION,
        verbose=False,
        seed: int = 42,
    ):
        """
        Initialize an LLMInterview instance. Either a path to a csv file or a pandas dataframe has to be provided

        Args:
            interview_path (str): Path to the CSV file containing the interview structure and questions.
            interview_dataframe (pd.Dataframe): A pandas dataframe interview structure and questions.
            interview_name (str): Name/ID for the interview.
            system_prompt (str): System prompt to prepend to all questions.
            interview_instruction (str): Instructions that will be given to the model before asking the questions.
            verbose (bool): If True, enables verbose output.
            seed (int): Random seed for reproducibility.
        """
        random.seed(seed)

        if interview_path == None and interview_dataframe is None:
            raise ValueError("Either a path or a dataframe have to be provided")

        if interview_dataframe is None:
            self.load_interview_format(interview_path=interview_path)
        else:
            self.load_interview_format(interview_dataframe=interview_dataframe)
        
        self.verbose: bool = verbose

        self.interview_name: str = interview_name

        self.system_prompt: str = system_prompt
        self.interview_instruction: str = interview_instruction

        self._global_options: AnswerOptions = None

    def duplicate(self):
        """
        Create a deep copy of the current interview instance.

        Returns:
            LLMInterview: A deep copy of the current object.
        """
        return copy.deepcopy(self)

    def get_prompt_structure(self) -> str:
        """
        Generate a prompt structure for the first question, including system prompt and instructions.

        Returns:
            str: The full prompt as a string.
        """
        parts = [
            "SYSTEM PROMPT:",
            self.system_prompt,
            "INTERVIEW INSTRUCTIONS:",
            self.interview_instruction,
        ]

        if self._global_options:
            _options_str = self._global_options.create_options_str()
            if _options_str is not None:
                parts.append(_options_str)

        parts.append("FIRST QUESTION:")
        parts.append(self.generate_question_prompt(self._questions[0]))

        return "\n".join(parts)

    def get_prompt_for_interview_type(
        self, interview_type: InterviewType = InterviewType.QUESTION
    ):
        """
        Generate the full prompt for a given interview type.

        Args:
            interview_type (InterviewType): The type of interview prompt to generate.

        Returns:
            str: The constructed prompt for the interview type.
        """
        parts = [self.system_prompt, self.interview_instruction]

        if self._global_options:
            _options_str = self._global_options.create_options_str()
            if _options_str is not None:
                parts.append(_options_str)

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
        Estimate the number of input tokens for the prompt, given a model and interview type.
        Remember that the model also has to have enough context length to fit its own response 
        in case of CONTEXT and ONE_PROMPT type.

        Args:
            model_id (str): Huggingface model id.
            interview_type (InterviewType): Type of interview prompt.

        Returns:
            int: Estimated number of input tokens.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        whole_prompt = self.get_prompt_for_interview_type(interview_type=interview_type)
        tokens = tokenizer.encode(whole_prompt)

        return (
            len(tokens) if interview_type != InterviewType.CONTEXT else len(tokens) * 3
        )

    def get_survey_questions(self) -> str:
        """
        Get the list of loaded interview questions.

        Returns:
            List[InterviewItem]: The loaded questions.
        """
        return self._questions

    def load_interview_format(self, interview_path: Optional[str] = None, interview_dataframe: Optional[pd.DataFrame] = None) -> Self:
        """
        Load interview questions from a CSV file.

        The CSV should have columns: interview_item_id, question_content
        Optionally it can also have question_stem.

        Args:
            interview_path (str): Path to the CSV file.

        Returns:
            Self: The updated instance with loaded questions.
        """
        interview_questions: List[InterviewItem] = []

        if interview_path == None and interview_dataframe is None:
            raise ValueError("Either a path or a dataframe have to be provided")

        if interview_dataframe is not None:
            df = interview_dataframe
        else:
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
        Prepare the interview by assigning question stems, answer options, and prefilled responses.

        Args:
            question_stem (str or List[str], optional): Single or list of question stems.
            answer_options (AnswerOptions or Dict[int, AnswerOptions], optional): Answer options for all or per question.
            global_options (bool): If True, the answer options will be specified once at the end of the task instructions. Otherwise, they will be specified once per question.
            prefilled_responses (Dict[int, str], optional): If you provide prefilled responses, they will be used 
            to fill the answers instead of prompting the LLM for that question.
            randomized_item_order (bool): If True, randomize the order of questions.

        Returns:
            Self: The updated instance with prepared questions.
        """
        interview_questions: List[InterviewItem] = self._questions

        prompt_list = isinstance(question_stem, list)
        if prompt_list:
            assert len(question_stem) == len(
                interview_questions
            ), "If a list of question stems is given, length of prompt and survey questions have to be the same"

        options_dict = False

        if answer_options == None:
            self._global_options = None
        elif isinstance(answer_options, AnswerOptions):
            options_dict = False
            if global_options:
                self._global_options = answer_options
            else:
                self._global_options = None
        elif isinstance(answer_options, Dict):
            options_dict = True
            self._global_options = None

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
        Generate the prompt string for a single interview question.

        Args:
            interview_question (InterviewItem): The question to prompt.

        Returns:
            str: The formatted prompt for the question.
        """
        if interview_question.question_stem:
            if constants.QUESTION_CONTENT_PLACEHOLDER in interview_question.question_stem:
                question_prompt = interview_question.question_stem.format(
                    **{
                        constants.QUESTION_CONTENT_PLACEHOLDER: interview_question.question_content
                    }
                )
            else:
                question_prompt = f"""{interview_question.question_stem} {interview_question.question_content}"""
        else:
             question_prompt = f"""{interview_question.question_content}"""
        if interview_question.answer_options:
            _options_str = interview_question.answer_options.create_options_str()
            if _options_str is not None:
                question_prompt = '\n'.join([question_prompt, _options_str])
        return question_prompt

    def _generate_inference_options(
        self,
    ):
        """
        Internal method to generate inference options for the interview.

        Returns:
            InferenceOptions: Object containing prompts, answer options, and order.
        """
        interview_questions = self._questions

        default_prompt = f"""{self.interview_instruction}"""

        if self._global_options:
            _options_str = self._global_options.create_options_str()
            if _options_str is not None:
                default_prompt = '\n'.join([default_prompt, _options_str])

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
            answer_options=answer_options,
            order=order,
        )
