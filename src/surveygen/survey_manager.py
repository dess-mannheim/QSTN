from typing import (
    List,
    Dict,
    Optional,
    Union,
    Any,
    overload,
    Self,
    Literal,
    Final,
)
from enum import Enum

from dataclasses import dataclass, replace
from string import ascii_lowercase, ascii_uppercase

from .utilities.prompt_creation import PromptCreation
from .utilities.survey_objects import (
    AnswerOptions,
    InterviewItem,
    QuestionLLMResponseTuple,
)
from .utilities import prompt_templates
from .utilities import constants


from .inference.survey_inference import batch_generation, batch_turn_by_turn_generation
from .inference.dynamic_pydantic import generate_pydantic_model

from vllm import LLM
from vllm.sampling_params import GuidedDecodingParams

from openai import AsyncOpenAI

from transformers import AutoTokenizer

import pandas as pd

import random

import copy

import tqdm


class SurveyOptionGenerator:
    """
    A class responsible for preparing optional answers for the LLM.
    """

    LIKERT_5: List[str] = [
        "disagree strongly",
        "disagree a little",
        "neither agree nor disagree",
        "agree a little",
        "agree strongly",
    ]
    LIKERT_NO_MIDDLE: List[str] = [
        "disagree strongly",
        "disagree a little",
        "agree a little",
        "agree strongly",
    ]

    LIKERT_IMPORTANCE_FROM_TO: List[str] = ["Not at all important", "Very Important"]
    LIKERT_JUSTIFIABLE_FROM_TO: List[str] = ["Never justifiable", "Always justifiable"]
    _IDX_TYPES = Literal["char_low", "char_upper", "integer"]

    @staticmethod
    def generate_likert_options(
        n: int,
        answer_texts: Optional[List[str]],
        only_from_to_scale: bool = False,
        random_order: bool = False,
        reversed_order: bool = False,
        even_order: bool = False,
        start_idx: int = 1,
        list_prompt_template: str = prompt_templates.LIST_OPTIONS_DEFAULT,
        scale_prompt_template: str = prompt_templates.SCALE_OPTIONS_DEFAULT,
        options_separator: str = ", ",
        idx_type: _IDX_TYPES = "integer",
    ) -> AnswerOptions:
        """Generates a set of options and a prompt for a Likert-style scale.

        This function creates a numeric or alphabetic scale of a specified size (n),
        optionally attaching textual labels to the scale. It provides
        extensive control over ordering, formatting, and the final prompt string.

        :param n: The number of points on the scale (e.g., 5 for a 1-5 scale).
        :type n: int
        :param answer_texts: A list of strings to use as labels for points on the scale.
                             For example, ["Agree", "Disagree", "Neither"] for a 3-point scale.
        :type answer_texts: Optional[List[str]]
        :param only_from_to_scale: If True, forces a 'range' style prompt (e.g., "from 1 to 5")
                                using `scale_prompt_template`.
        :type only_from_to_scale: bool
        :param random_order: If True, shuffles the final list of generated options.
                            Mutually exclusive with `reversed_order`.
        :type random_order: bool
        :param reversed_order: If True, reverses the natural order of the scale (e.g., 5, 4, 3, 2, 1).
                            Mutually exclusive with `random_order`.
        :type reversed_order: bool
        :param even_order: Removes the middle answer option of the scale, if the scale is odd.
        :type even_order: bool
        :param start_idx: The starting number for an integer-based scale (e.g., 1 or 0).
        :type start_idx: int
        :param list_prompt_template: The format string for a list-style prompt. Must contain `{options}`.
        :type list_prompt_template: str
        :param scale_prompt_template: The format string for a range-style prompt. Must contain `{start}`
                                    and `{end}`.
        :type scale_prompt_template: str
        :param idx_type: The type of index for the scale: "char_low", "char_upper" or "integer".
        :type idx_type: _IDX_TYPES

        :raises ValueError: If `n` is less than 2, if `random_order` and `reversed_order` are both True,
                            or if `len(anchor_labels)` > `n`.

        :return: An object containing `options` (the final list of option strings) and `prompt`
                (the formatted prompt string).
        :rtype: AnswerOptions"""

        # @TODO @Jens Instead of assertions we should probably raise Value errors
        if only_from_to_scale:
            assert (
                len(answer_texts) == 2
            ), "If from to scale, provide exactly two descriptions"
            assert (
                idx_type == "integer"
            ), "Index type must be integer, not lower/uppercase characters."
        else:
            if answer_texts:
                assert (
                    len(answer_texts) == n
                ), "Description list must be the same length as options"
        if even_order:
            assert n % 2 != 0, "There must be a odd number of options!"
            middle_index = n // 2
            answer_texts = (
                answer_texts[:middle_index] + answer_texts[middle_index + 1 :]
            )
            n = n - 1
        if random_order:
            assert (
                len(answer_texts) >= 2
            ), "There must be at least two answer options to reorder randomly."
            random.shuffle(
                answer_texts
            )  # no assignment needed because shuffles already inplace
        if reversed_order:
            assert (
                len(answer_option) >= 2
            ), "There must be at least two answer options to reverse options."
            answer_options = answer_options[::-1]

        answer_options = []
        if idx_type == "integer":
            for i in range(n):
                answer_code = i + start_idx
                answer_option = f"{answer_code}"
                if only_from_to_scale:
                    if i == 0:
                        answer_option = f"{answer_code}: {answer_texts[0]}"
                    elif i == (n - 1):
                        answer_option = f"{answer_code}: {answer_texts[1]}"
                elif answer_texts:
                    answer_option = f"{answer_code}: {answer_texts[i]}"
                answer_options.append(answer_option)
        else:
            # TODO @Jens add these to constants.py
            if idx_type == "char_low":
                for i in range(n):
                    answer_option = f"{ascii_lowercase[i]}: {answer_texts[i]}"
                    answer_options.append(answer_option)
            elif idx_type == "char_upper":
                for i in range(n):
                    answer_option = f"{ascii_uppercase[i]}: {answer_texts[i]}"
                    answer_options.append(answer_option)

        interview_option = AnswerOptions(
            answer_options,
            from_to_scale=only_from_to_scale,
            list_prompt_template=list_prompt_template,
            scale_prompt_template=scale_prompt_template,
            options_seperator=options_separator,
        )

        return interview_option

    @staticmethod
    def generate_generic_options(
        answer_texts: Dict,
        only_from_to_scale: bool = False,
        random_order: bool = False,
        reversed_order: bool = False,
        even_order: bool = False,
        to_lowercase: bool = False,
        to_uppercase: bool = False,
        to_integer: bool = False,
        list_prompt_template: str = prompt_templates.LIST_OPTIONS_DEFAULT,
        scale_prompt_template: str = prompt_templates.SCALE_OPTIONS_DEFAULT,
        options_separator: str = ", ",
    ):

        n = len(answer_texts.values())
        answer_codes = answer_texts.keys()
        answer_texts = answer_texts.values()
        # answer_options = descriptions

        if to_lowercase:
            if all(isinstance(item, int) for item in answer_codes):
                new_codes = []
                for i in answer_codes:
                    code = ascii_lowercase[i - 1]
                    new_codes.append(code)
                answer_codes = new_codes
            else:
                answer_codes = [s.lower() for s in answer_codes]
        if to_uppercase:
            if all(isinstance(item, int) for item in answer_codes):
                new_codes = []
                for i in answer_codes:
                    code = ascii_uppercase[i - 1]
                    new_codes.append(code)
                answer_codes = new_codes
            else:
                answer_codes = [s.upper() for s in answer_codes]
        if to_integer:
            answer_codes = range(1, len(answer_codes) + 1)

        answer_options = dict(zip(answer_codes, answer_texts))

        if only_from_to_scale:
            assert all(
                isinstance(item, int) for item in answer_codes
            ), "To use from-to scale you must have integer answer codes."

        if random_order:
            assert (
                n >= 2
            ), "There must be at least two answer options to reorder randomly."
            temp = list(answer_texts)
            random.shuffle(temp)
            # reassigning to keys
            answer_options = dict(zip(answer_codes, temp))
        if reversed_order:
            assert (
                n >= 2
            ), "There must be at least two answer options to reverse options."
            reversed_values = list(answer_texts)[::-1]
            answer_options = dict(zip(answer_codes, reversed_values))
        if even_order:
            assert n % 2 != 0, "There must be a odd number of options!"
            middle_index = n // 2
            # Get the key of the item to be removed
            key_to_remove = list(answer_codes)[middle_index]
            # Create a new dictionary, excluding the item with key_to_remove
            # This uses a dictionary comprehension.
            answer_options = {
                key: value
                for key, value in answer_options.items()
                if key != key_to_remove
            }
            if all(isinstance(element, int) for element in list(answer_codes)):
                first_part = list(answer_codes)[:middle_index]
                last_part = list(answer_codes)[middle_index + 1 :]
                last_part = [x - 1 for x in last_part]
                answer_options = dict(
                    zip(first_part + last_part, answer_options.values())
                )
            elif set(list(answer_codes)).issubset(list(ascii_lowercase)):
                first_part = list(answer_codes)[:middle_index]
                # print("First part:", first_part)
                last_part = list(answer_codes)[middle_index + 1 :]
                # print("Last part:", last_part)
                last_parts = []
                for i in range(middle_index + 1, len(list(answer_codes))):
                    part = list(ascii_lowercase)[i - 1]
                    last_parts.append(part)
                    # print("Last parts:", last_parts)
                answer_options = dict(
                    zip(first_part + last_parts, answer_options.values())
                )
            elif set(list(answer_codes)).issubset(list(ascii_uppercase)):
                first_part = list(answer_codes)[:middle_index]
                # print("First part:", first_part)
                last_part = list(answer_codes)[middle_index + 1 :]
                # print("Last part:", last_part)
                last_parts = []
                for i in range(middle_index + 1, len(list(answer_codes))):
                    part = list(ascii_uppercase)[i - 1]
                    last_parts.append(part)
                    # print("Last parts:", last_parts)
                answer_options = dict(
                    zip(first_part + last_parts, answer_options.values())
                )

        answer_options = [f"{key}: {val}" for key, val in answer_options.items()]
        print(answer_options)

        interview_option = AnswerOptions(
            answer_options,
            from_to_scale=only_from_to_scale,
            list_prompt_template=list_prompt_template,
            scale_prompt_template=scale_prompt_template,
            options_seperator=options_separator,
        )

        return interview_option


class InterviewType(Enum):
    QUESTION: Final[str] = "interview_type_question"
    CONTEXT: Final[str] = "interview_type_context"
    ONE_PROMPT: Final[str] = "interview_type_one_prompt"


@dataclass
class InferenceOptions:
    system_prompt: str
    task_instruction: str
    question_prompts: Dict[int, str]
    answer_options: List[AnswerOptions]
    # guided_decodings: Optional[Dict[int, GuidedDecodingParams]]
    # full_guided_decoding: Optional[GuidedDecodingParams]
    # json_structure: Optional[List[str]]
    # full_json_structure: Optional[List[str]]
    order: List[int]

    def create_single_question(
        self, question_id: int, task_instruction: bool = False
    ) -> str:
        if task_instruction:
            return f"""{self.task_instruction} 
{self.question_prompts[question_id]}""".strip()
        else:
            return f"""{self.question_prompts[question_id]}"""

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

    def json_system_prompt(self, json_options: List[str]) -> str:
        creator = PromptCreation()
        creator.set_ouput_format_json(
            json_attributes=json_options, json_explanation=None
        )
        json_appendix = creator.get_output_prompt()

        system_prompt = f"""{self.system_prompt}
{json_appendix}"""
        return system_prompt


DEFAULT_SYSTEM_PROMPT: str = (
    "You will be given questions and possible answer options for each. Please reason about each question before answering."
)
DEFAULT_TASK_INSTRUCTION: str = ""

DEFAULT_JSON_STRUCTURE: List[str] = ["reasoning", "answer"]

DEFAULT_CONSTRAINTS: Dict[str, List[str]] = {"answer": constants.OPTIONS_ADJUST}

DEFAULT_INTERVIEW_ID: str = "Interview"


class LLMInterview:
    """
    A class responsible for preparing and conducting surveys on LLMs.
    """

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
        whole_prompt = f"""SYSTEM PROMPT:
{self.system_prompt}
INTERVIEW INSTRUCTIONS:
{self.interview_instruction}{f"{chr(10)}{self._global_options.create_options_str()}" if self._global_options else ""}
FIRST QUESTION:
{self.generate_question_prompt(self._questions[0])}
"""
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
        if interview_type == InterviewType.QUESTION:

            whole_prompt = f"""{self.system_prompt}
{self.interview_instruction}{f"{chr(10)}{self._global_options.create_options_str()}" if self._global_options else ""}
{self.generate_question_prompt(self._questions[0])}
    """
        elif (
            interview_type == InterviewType.ONE_PROMPT or interview_type == InterviewType.CONTEXT
        ):
            questions: str = "\n".join(
                self.generate_question_prompt(question) for question in self._questions
            )
            whole_prompt = f"""{self.system_prompt}
{self.interview_instruction}{f"{chr(10)}{self._global_options.create_options_str()}" if self._global_options else ""}
{questions}"""
        tokens = tokenizer.encode(whole_prompt)

        return len(tokens) if interview_type != InterviewType.CONTEXT else len(tokens) * 3

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


@dataclass
class SurveyResult:
    survey: LLMInterview
    results: Dict[int, QuestionLLMResponseTuple]

    def to_dataframe(self) -> pd.DataFrame:
        answers = []
        for item_id, question_llm_response_tuple in self.results.items():
            answers.append((item_id, *question_llm_response_tuple))
        return pd.DataFrame(
            answers,
            columns=[constants.INTERVIEW_ITEM_ID, *question_llm_response_tuple._fields],
        )


def conduct_survey_question_by_question(
    model: Union[LLM, AsyncOpenAI],
    interviews: Union[LLMInterview, List[LLMInterview]],
    json_structured_output: bool = False,
    json_structure: Optional[List[str]] = DEFAULT_JSON_STRUCTURE,
    constraints: Optional[Dict[str, List[str]]] = DEFAULT_CONSTRAINTS,
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    seed: int = 42,
    **generation_kwargs: Any,
) -> List[SurveyResult]:
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
    if isinstance(interviews, LLMInterview):
        interviews = [interviews]

    inference_options: List[InferenceOptions] = []

    max_survey_length: int = 0

    question_llm_response_pairs: List[Dict[int, QuestionLLMResponseTuple]] = []

    # if print_progress:
    #     print("Constructing prompts")
    for i in range(len(interviews)):
        inference_option = interviews[i]._generate_inference_options()
        inference_options.append(inference_option)
        survey_length = len(inference_option.order)
        if survey_length > max_survey_length:
            max_survey_length = survey_length

        question_llm_response_pairs.append({})

    survey_results: List[SurveyResult] = []

    #TODO allow for different answer option constraints between surveys
    if constraints:
        for json_element in constraints.keys():
            if constraints[json_element] == constants.OPTIONS_ADJUST:
                constraints[json_element] = inference_options[0].answer_options[0].answer_text

    for i in (
        tqdm.tqdm(range(max_survey_length))
        if print_progress
        else range(max_survey_length)
    ):
        current_batch = [
            inference_option
            for inference_option in inference_options
            if len(inference_option.order) > i
        ]

        if json_structured_output:
            system_messages = [
                inference.json_system_prompt(json_options=json_structure)
                for inference in current_batch
            ]

        else:
            system_messages = [inference.system_prompt for inference in current_batch]
        prompts = [
            inference.create_single_question(inference.order[i], task_instruction=True)
            for inference in current_batch
        ]
        questions = [
            inference.create_single_question(inference.order[i], task_instruction=False)
            for inference in current_batch
        ]

        output = batch_generation(
            model=model,
            system_messages=system_messages,
            prompts=prompts,
            guided_decoding_options= "json" if json_structured_output else None,
            json_fields=json_structure,
            constraints=constraints,
            client_model_name=client_model_name,
            api_concurrency=api_concurrency,
            print_conversation=print_conversation,
            print_progress=print_progress,
            seed=seed,
            **generation_kwargs,
        )

        for survey_id, question, answer, item in zip(
            range(len(current_batch)), questions, output, current_batch
        ):
            question_llm_response_pairs[survey_id].update(
                {item.order[i]: QuestionLLMResponseTuple(question, answer)}
            )

    for i, survey in enumerate(interviews):
        survey_results.append(SurveyResult(survey, question_llm_response_pairs[i]))

    return survey_results


def conduct_whole_survey_one_prompt(
    model: Union[LLM, AsyncOpenAI],
    interviews: Union[LLMInterview, List[LLMInterview]],
    json_structured_output: bool = False,
    json_structure: Optional[List[str]] = DEFAULT_JSON_STRUCTURE,
    constraints: Optional[Dict[str, List[str]]] = DEFAULT_CONSTRAINTS,
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    seed: int = 42,
    **generation_kwargs: Any,
) -> List[SurveyResult]:
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
    if isinstance(interviews, LLMInterview):
        interviews = [interviews]
    inference_options: List[InferenceOptions] = []

    # We always conduct the survey in one prompt
    max_survey_length: int = 1

    question_llm_response_pairs: List[Dict[int, QuestionLLMResponseTuple]] = []

    # if print_progress:
    #     print("Constructing prompts")
    for i in range(len(interviews)):
        inference_option = interviews[i]._generate_inference_options()
        inference_options.append(inference_option)

        question_llm_response_pairs.append({})

    survey_results: List[SurveyResult] = []

    for i in (
        tqdm.tqdm(range(max_survey_length))
        if print_progress
        else range(max_survey_length)
    ):
        current_batch = [
            inference_option
            for inference_option in inference_options
            if len(inference_option.order) > i
        ]

        if json_structured_output:
            all_json_structures = []
            all_constraints = []
            for inference_option in current_batch:
                full_json_structure = []
                full_constraints = {}
                for i in range(len(inference_option.answer_options)):
                    for json_element in json_structure:
                        new_element = f"{json_element}{i}"
                        if constraints:
                            constraints_element = constraints.get(json_element)
                            if constraints_element == constants.OPTIONS_ADJUST:
                                full_constraints[new_element] = inference_option.answer_options[i].answer_text
                            elif constraints_element != None:
                                full_constraints[new_element] = constraints_element
                        full_json_structure.append(new_element)
                all_constraints.append(full_constraints)
                all_json_structures.append(full_json_structure)
                    
            system_messages = [
                inference.json_system_prompt(all_json_structures[num])
                for num, inference in enumerate(current_batch)
            ]

        else:
            system_messages = [inference.system_prompt for inference in current_batch]
        prompts = [inference.create_all_questions() for inference in current_batch]

        output = batch_generation(
            model=model,
            system_messages=system_messages,
            prompts=prompts,
            guided_decoding_options= "json" if json_structured_output else None,
            json_fields=all_json_structures,
            constraints=all_constraints if constraints else None,
            client_model_name=client_model_name,
            api_concurrency=api_concurrency,
            print_conversation=print_conversation,
            print_progress=print_progress,
            seed=seed,
            **generation_kwargs,
        )

        for survey_id, prompt, answer in zip(
            range(len(current_batch)), prompts, output
        ):
            question_llm_response_pairs[survey_id].update(
                {-1: QuestionLLMResponseTuple(prompt, answer)}
            )

    for i, survey in enumerate(interviews):
        survey_results.append(SurveyResult(survey, question_llm_response_pairs[i]))

    return survey_results


def conduct_survey_in_context(
    model: Union[LLM, AsyncOpenAI],
    interviews: Union[LLMInterview, List[LLMInterview]],
    json_structured_output: bool = False,
    json_structure: Optional[List[str]] = DEFAULT_JSON_STRUCTURE,
    constraints: Optional[Dict[str, List[str]]] = DEFAULT_CONSTRAINTS,
    client_model_name: Optional[str] = None,
    api_concurrency: int = 10,
    print_conversation: bool = False,
    print_progress: bool = True,
    seed: int = 42,
    **generation_kwargs: Any,
) -> List[SurveyResult]:
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
    if isinstance(interviews, LLMInterview):
        interviews = [interviews]

    inference_options: List[InferenceOptions] = []

    max_survey_length: int = 0

    question_llm_response: List[Dict[int, QuestionLLMResponseTuple]] = []

    # if print_progress:
    #     print("Constructing prompts")
    for i in range(len(interviews)):
        inference_option = interviews[i]._generate_inference_options()
        inference_options.append(inference_option)
        survey_length = len(inference_option.order)
        if survey_length > max_survey_length:
            max_survey_length = survey_length

        question_llm_response.append({})

    survey_results: List[SurveyResult] = []

    all_prompts: List[List[str]] = []
    assistant_messages: List[List[str]] = []

    if constraints:
        for json_element in constraints.keys():
            if constraints[json_element] == constants.OPTIONS_ADJUST:
                constraints[json_element] = inference_options[0].answer_options[0].answer_text

    for i in range(len(interviews)):
        assistant_messages.append([])
        all_prompts.append([])

    for i in (
        tqdm.tqdm(range(max_survey_length))
        if print_progress
        else range(max_survey_length)
    ):
        current_batch = [
            inference_option
            for inference_option in inference_options
            if len(inference_option.order) > i
        ]
        current_surveys = [surv for surv in interviews if len(surv._questions) > i]

        first_question: bool = i == 0

        prompts = [
            inference.create_single_question(
                inference.order[i], task_instruction=first_question
            )
            for inference in current_batch
        ]
        questions = [
            inference.create_single_question(inference.order[i], task_instruction=False)
            for inference in current_batch
        ]
        for c in range(len(current_surveys)):
            all_prompts[c].append(prompts[c])

        current_assistant_messages: List[str] = []

        missing_indeces = []

        for index, surv in enumerate(current_surveys):
            prefilled_answer = surv._questions[i].prefilled_response
            if prefilled_answer:
                current_assistant_messages.append(prefilled_answer)
                missing_indeces.append(index)

        current_batch = [
            item for a, item in enumerate(current_batch) if a not in missing_indeces
        ]

        if len(current_batch) == 0:
            for c in range(len(current_surveys)):
                assistant_messages[c].append(current_assistant_messages[c])
            for survey_id, question, llm_response, item in zip(
                range(len(current_surveys)),
                questions,
                current_assistant_messages,
                current_surveys,
            ):
                question_llm_response[survey_id].update(
                    {
                        item._questions[i].item_id: QuestionLLMResponseTuple(
                            question, llm_response
                        )
                    }
                )
            continue
        if json_structured_output:
            system_messages = [
                inference.json_system_prompt(json_options=json_structure)
                for inference in current_batch
            ]
        else:
            system_messages = [inference.system_prompt for inference in current_batch]

        output = batch_turn_by_turn_generation(
            model=model,
            system_messages=system_messages,
            prompts=all_prompts,
            assistant_messages=assistant_messages,
            guided_decoding_options= "json" if json_structured_output else None,
            json_fields=json_structure,
            constraints=constraints,
            client_model_name=client_model_name,
            api_concurrency=api_concurrency,
            print_conversation=print_conversation,
            print_progress=print_progress,
            seed=seed,
            **generation_kwargs,
        )

        for num, index in enumerate(missing_indeces):
            output.insert(index, current_assistant_messages[num])
        for survey_id, question, llm_response, item in zip(
            range(len(current_surveys)), questions, output, current_surveys
        ):
            question_llm_response[survey_id].update(
                {
                    item._questions[i].item_id: QuestionLLMResponseTuple(
                        question, llm_response
                    )
                }
            )

        for o in range(len(output)):
            assistant_messages[o].append(output[o])
        # assistant_messages.append(output)

    for i, survey in enumerate(interviews):
        survey_results.append(SurveyResult(survey, question_llm_response[i]))

    return survey_results


if __name__ == "__main__":
    pass
