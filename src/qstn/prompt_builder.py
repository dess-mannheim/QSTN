from typing import List, Dict, Optional, Union, overload, Self, Tuple, Literal
from string import ascii_lowercase, ascii_uppercase

from dataclasses import replace

from .utilities.survey_objects import AnswerOptions, QuestionnaireItem, AnswerTexts
from .utilities import constants, placeholder, prompt_templates
from .utilities.constants import QuestionnairePresentation
from .utilities.utils import safe_format_with_regex

from .utilities.prompt_perturbations import * 

from .inference.response_generation import ResponseGenerationMethod

import pandas as pd

import random

import copy


#from transformers import AutoTokenizer


class LLMPrompt:
    """
    Main class for setting up and managing the prompt in the LLM experiment.

    This class handles loading questions from a predefined questionnaire, preparing prompts, managing answer options,
    and generating prompt structures for different interview types.
    """

    DEFAULT_QUESTIONNAIRE_ID: str = "Questionnaire"

    DEFAULT_SYSTEM_PROMPT: str = (
        "You will be given questions and possible answer options for each. Please reason about each question before answering."
    )
    DEFAULT_TASK_INSTRUCTION: str = ""

    DEFAULT_JSON_STRUCTURE: List[str] = ["reasoning", "answer"]

    DEFAULT_PROMPT_STRUCTURE: str = (
        f"{placeholder.PROMPT_QUESTIONS}\n{placeholder.PROMPT_OPTIONS}"
    )

    def __init__(
        self,
        questionnaire_source=Union[str, pd.DataFrame],
        questionnaire_name: str = DEFAULT_QUESTIONNAIRE_ID,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        prompt: str = DEFAULT_PROMPT_STRUCTURE,
        verbose=False,
        seed: int = 42,
    ):
        """
        Initialize an LLMPrompt instance. Either a path to a csv file or a pandas dataframe has to be provided to structure the questionnaire.

        Args:
            questionnaire_source (str/pd.Dataframe): Path to the CSV file containing the questionnaire structure and questions.
            questionnaire_name (str): Name/ID for the questionnaire.
            system_prompt (str): System prompt for all questions.
            prompt (str): Prompt for all questions.
            verbose (bool): If True, enables verbose output.
            seed (int): Random seed for reproducibility.
          """
        random.seed(seed)

        if questionnaire_source is None:
            raise ValueError("Either a path or a dataframe have to be provided")

        self.load_questionnaire_format(questionnaire_source=questionnaire_source)

        self.verbose: bool = verbose

        self.questionnaire_name: str = questionnaire_name

        self.system_prompt: str = system_prompt
        self.prompt: str = prompt

        self._same_options = False

    def duplicate(self):
        """
        Create a deep copy of the current interview instance.

        Returns:
            LLMQuestionnaire: A deep copy of the current object.
        """
        return copy.deepcopy(self)

    def get_prompt_for_questionnaire_type(
        self,
        questionnaire_type: QuestionnairePresentation = QuestionnairePresentation.SINGLE_ITEM,
        item_id: int = 0,
        item_separator: str = "\n",
    ) -> Tuple[str, str]:
        """
        Generate the full prompt for a given questionnaire presentation.

        Args:
            quesitonnaire_type (QuestionnairePresentation): The type of questionnaire prompt to generate.

        Returns:
            str: The constructed prompt for the questionnaire presentation.
        """
        options = ""
        automatic_output_instructions = ""
        if (
            questionnaire_type == QuestionnairePresentation.SINGLE_ITEM
            or questionnaire_type == QuestionnairePresentation.SEQUENTIAL
        ):
            question = self.generate_question_prompt(self._questions[item_id])

            if self._questions[item_id].answer_options:
                options = self._questions[item_id].answer_options.create_options_str()

                rgm = self._questions[item_id].answer_options.response_generation_method
                if rgm is None:  # by default, no response generation method is required
                    automatic_output_instructions = ""
                else:
                    automatic_output_instructions: str = rgm.get_automatic_prompt()
            else:
                options = ""
                automatic_output_instructions = ""

            format_dict = {
                placeholder.PROMPT_QUESTIONS: question,
                placeholder.PROMPT_OPTIONS: options,
                placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS: automatic_output_instructions,
            }

        elif questionnaire_type == QuestionnairePresentation.BATTERY:
            all_questions: List[str] = []
            for question in self._questions:
                current_question_prompt = self.generate_question_prompt(question)

                if question.answer_options:
                    options = question.answer_options.create_options_str()
                else:
                    options = ""
                format_dict = {
                    placeholder.PROMPT_OPTIONS: options,
                }
                current_question_prompt = safe_format_with_regex(
                    current_question_prompt, format_dict
                )
                all_questions.append(current_question_prompt)

            all_questions_str = item_separator.join(all_questions)
            if self._questions[item_id].answer_options:
                options = self._questions[item_id].answer_options.create_options_str()
                rgm = self._questions[item_id].answer_options.response_generation_method

                if rgm is None:  # by default, no response generation method is required
                    automatic_output_instructions = ""
                else:
                    automatic_output_instructions: str = rgm.get_automatic_prompt(
                        questions=self._questions
                    )
            else:
                options = ""
                automatic_output_instructions = ""

            format_dict = {
                placeholder.PROMPT_QUESTIONS: all_questions_str,
                placeholder.PROMPT_OPTIONS: options,
                placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS: automatic_output_instructions,
            }

        system_prompt = safe_format_with_regex(self.system_prompt, format_dict)
        prompt = safe_format_with_regex(self.prompt, format_dict)

        return system_prompt, prompt

    # def calculate_input_token_estimate(
    #     self, model_id: str, questionnaire_type: QuestionnairePresentation = QuestionnairePresentation.SINGLE_ITEM
    # ) -> int:
    #     """
    #     Estimate the number of input tokens for the prompt, given a model and questionnaire type.
    #     Remember that the model also has to have enough context length to fit its own response
    #     in case of CONTEXT and ONE_PROMPT type.

    #     Args:
    #         model_id (str): Huggingface model id.
    #         questionnaire_type (QuestionnairePresentation): Type of questionnaire prompt.

    #     Returns:
    #         int: Estimated number of input tokens.
    #     """
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     system_prompt, prompt = self.get_prompt_for_questionnaire_type(
    #         questionnaire_type=questionnaire_type
    #     )
    #     system_tokens = tokenizer.encode(system_prompt)
    #     tokens = tokenizer.encode(prompt)
    #     total_tokens = len(system_tokens) + len(tokens)

    #     return (
    #         total_tokens
    #         if questionnaire_type != QuestionnairePresentation.SEQUENTIAL
    #         else len(total_tokens) * 3
    #     )

    def get_questions(self) -> List[QuestionnaireItem]:
        """
        Get the list of loaded interview questions.

        Returns:
            List[QuestionnaireItem]: The loaded questions.
        """
        return self._questions

    def load_questionnaire_format(self, questionnaire_source: Union[str, pd.DataFrame]) -> Self:
        """
        Load the questionnaire format from a CSV file.

        The CSV should have columns: questionnaire_item_id, question_content
        Optionally it can also have question_stem.

        Args:
            questionnaire_source (str or pd.Dataframe): Path to a valid CSV file or pd.Dataframe.

        Returns:
            Self: The updated instance with loaded questions.
        """
        questionnaire_questions: List[QuestionnaireItem] = []

        if questionnaire_source is None:
            raise ValueError("Either a path or a dataframe have to be provided")

        if type(questionnaire_source) == pd.DataFrame:
            df = questionnaire_source
        else:
            df = pd.read_csv(questionnaire_source)

        for _, row in df.iterrows():
            questionnaire_item_id = row[constants.QUESTIONNAIRE_ITEM_ID]
            # if constants.QUESTION in df.columns:
            #     question = row[constants.QUESTION]
            if constants.QUESTION_CONTENT in df.columns:
                questionnaire_question_content = row[constants.QUESTION_CONTENT]
            else:
                questionnaire_question_content = None

            if constants.QUESTION_STEM in df.columns:
                question_stem = row[constants.QUESTION_STEM]
            else:
                question_stem = None

            generated_questionnaire_question = QuestionnaireItem(
                item_id=questionnaire_item_id,
                question_content=questionnaire_question_content,
                question_stem=question_stem,
            )
            questionnaire_questions.append(generated_questionnaire_question)

        self._questions = questionnaire_questions
        return self

    # TODO Item order could be given by ids
    @overload
    def prepare_prompt(
        self,
        question_stem: Optional[str] = None,
        answer_options: Optional[AnswerOptions] = None,
        prefilled_responses: Optional[Dict[int, str]] = None,
        randomized_item_order: bool = False,
    ) -> Self: ...

    @overload
    def prepare_prompt(
        self,
        question_stem: Optional[List[str]] = None,
        answer_options: Optional[Dict[int, AnswerOptions]] = None,
        prefilled_responses: Optional[Dict[int, str]] = None,
        randomized_item_order: bool = False,
    ) -> Self: ...

    def prepare_prompt(
        self,
        question_stem: Optional[Union[str, List[str]]] = None,
        answer_options: Optional[Union[AnswerOptions, Dict[int, AnswerOptions]]] = None,
        prefilled_responses: Optional[Dict[int, str]] = None,
        randomized_item_order: bool = False,
    ) -> Self:
        """
        Prepare the interview by assigning question stems, answer options, and prefilled responses.

        Args:
            question_stem (str or List[str], optional): Single or list of question stems.
            answer_options (AnswerOptions or Dict[int, AnswerOptions], optional): Answer options for all or per question.
            prefilled_responses (Dict[int, str], optional): If you provide prefilled responses, they will be used
            to fill the answers instead of prompting the LLM for that question.
            randomized_item_order (bool): If True, randomize the order of questions.
        Returns:
            Self: The updated instance with prepared questions.
        """
        questionnaire_questions: List[QuestionnaireItem] = self._questions

        prompt_list = isinstance(question_stem, list)
        if prompt_list:
            assert len(question_stem) == len(
                questionnaire_questions
            ), "If a list of question stems is given, length of prompt and survey questions have to be the same"

        options_dict = False

        if isinstance(answer_options, AnswerOptions):
            self._same_options = True
            options_dict = False
        elif isinstance(answer_options, Dict):
            self._same_options = False
            options_dict = True

        updated_questions: List[QuestionnaireItem] = []

        if not prefilled_responses:
            prefilled_responses = {}
            # for survey_question in survey_questions:
            # prefilled_answers[survey_question.question_id] = None

        if not prompt_list and not options_dict:
            updated_questions = []
            for i in range(len(questionnaire_questions)):
                new_questionnaire_question = replace(
                    questionnaire_questions[i],
                    question_stem=(
                        question_stem
                        if question_stem
                        else questionnaire_questions[i].question_stem
                    ),
                    answer_options=answer_options,
                    prefilled_response=prefilled_responses.get(
                        questionnaire_questions[i].item_id
                    ),
                )
                updated_questions.append(new_questionnaire_question)

        elif not prompt_list and options_dict:
            for i in range(len(questionnaire_questions)):
                new_questionnaire_question = replace(
                    questionnaire_questions[i],
                    question_stem=(
                        question_stem
                        if question_stem
                        else questionnaire_questions[i].question_stem
                    ),
                    answer_options=answer_options.get(questionnaire_questions[i].item_id),
                    prefilled_response=prefilled_responses.get(
                        questionnaire_questions[i].item_id
                    ),
                )
                updated_questions.append(new_questionnaire_question)

        elif prompt_list and not options_dict:
            for i in range(len(questionnaire_questions)):
                new_questionnaire_question = replace(
                    questionnaire_questions[i],
                    question_stem=(
                        question_stem[i]
                        if question_stem
                        else questionnaire_questions[i].question_stem
                    ),
                    answer_options=answer_options,
                    prefilled_response=prefilled_responses.get(
                        questionnaire_questions[i].item_id
                    ),
                )
                updated_questions.append(new_questionnaire_question)
        elif prompt_list and options_dict:
            for i in range(len(questionnaire_questions)):
                new_questionnaire_question = replace(
                    questionnaire_questions[i],
                    question_stem=(
                        question_stem[i]
                        if question_stem
                        else questionnaire_questions[i].question_stem
                    ),
                    answer_options=answer_options.get(questionnaire_questions[i].item_id),
                    prefilled_response=prefilled_responses.get(
                        questionnaire_questions[i].item_id
                    ),
                )
                updated_questions.append(new_questionnaire_question)
        
        if randomized_item_order:
            random.shuffle(updated_questions)

        self._questions = updated_questions
        return self

    def generate_question_prompt(self, questionnaire_items: QuestionnaireItem) -> str:
        """
        Generate the prompt string for a single interview question.

        Args:
            questionnaire_items (InterviewItem): The question to prompt.

        Returns:
            str: The formatted prompt for the question.
        """

        if questionnaire_items.question_stem:
            if placeholder.QUESTION_CONTENT in questionnaire_items.question_stem:
                format_dict = {
                    placeholder.QUESTION_CONTENT: questionnaire_items.question_content
                }
                question_prompt = safe_format_with_regex(
                    questionnaire_items.question_stem, format_dict
                )
            else:
                question_prompt = f"""{questionnaire_items.question_stem} {questionnaire_items.question_content}"""
        else:
            question_prompt = f"""{questionnaire_items.question_content}"""
        
        
        if questionnaire_items.answer_options:
            _options_str = questionnaire_items.answer_options.create_options_str()
            if _options_str is not None:
                safe_formatter = {placeholder.PROMPT_OPTIONS: _options_str}
                question_prompt = safe_format_with_regex(
                    question_prompt, safe_formatter
                )
        
        return question_prompt
    

_IDX_TYPES = Literal["char_lower", "char_upper", "integer", "no_index"]

def generate_likert_options(
    n: int,
    answer_texts: Optional[List[str]],
    only_from_to_scale: bool = False,
    random_order: bool = False,
    reversed_order: bool = False,
    even_order: bool = False,
    add_middle_category: bool = False,
    str_middle_cat: str = "Neutral",
    add_refusal: bool = False,
    refusal_code: str = "-99",
    start_idx: int = 1,
    list_prompt_template: str = prompt_templates.LIST_OPTIONS_DEFAULT,
    scale_prompt_template: str = prompt_templates.SCALE_OPTIONS_DEFAULT,
    index_answer_separator: str = ": ",
    options_separator: str = ", ",
    idx_type: _IDX_TYPES = "integer",
    response_generation_method: Optional[ResponseGenerationMethod] = None,
) -> AnswerOptions:
    """Generates a set of options and a prompt for a Likert-style scale.

    This function creates a numeric or alphabetic scale of a specified size (n),
    optionally attaching textual labels to the scale. It provides
    extensive control over ordering, formatting, and the final prompt string.

    Args:
        n (int): The number of options to generate (e.g., 5 for a 5-point scale).
        answer_texts (Optional[List[str]]): A list of text labels for each option.
            Its length must equal `n` if provided.
        only_from_to_scale (bool, optional): If True, the prompt will only show the
            min and max of the scale (e.g., "1 to 5"). Defaults to False.
        random_order (bool, optional): If True, the options are randomized. Defaults to False.
        reversed_order (bool, optional): If True, the options are in reversed input order.
            Defaults to False.
        even_order (bool, optional): If True, options the center option will be removed.
            E.g., for n=5: 1, 2, 4, 5
        add_middle_category (bool, optional): If True, a middle category will be added. The name can be specified,
            by default it is "Neutral". E.g., for n=4: 1, 2, 3: Neutral, 4, 5
        str_middle_cat (str, optional): The label for the middle category if `add_middle_category` is True.
            Defaults to "Neutral".
        add_refusal (bool, optional): If True, an additional option for "Don't know / Refuse to answer" will be added.
            Defaults to False.
        refusal_code (str, optional): The code assigned to the refusal option if `add_refusal` is True.
            Defaults to "-99".
        start_idx (int, optional): The starting index for the scale (usually 0 or 1).
            Defaults to 1.
        list_prompt_template (str, optional): The template for prompts that list all options.
        scale_prompt_template (str, optional): The template for prompts that only show the range.
        index_answer_separator (str, optional): The string used to separate an index from its
            text label (e.g., "1: Strongly Agree"). Defaults to ": ".
        options_separator (str, optional): The string used to separate options when listed
            in the prompt. Defaults to ", ".
        idx_type (_IDX_TYPES, optional): The type of index to use: "integer", "upper" (A, B, C),
            or "lower" (a, b, c). Defaults to "integer".
        response_generation_method (Optional[ResponseGenerationMethod], optional): An object
            controlling how the final response object is generated. Defaults to None.

    Raises:
        ValueError: If `answer_texts` is provided and its length does not match `n`.

    Returns:
        AnswerOptions: An object containing the generated list of option strings and the
        final formatted prompt ready for display.

    Example:
        .. code-block:: python

            # Generate a classic 5-point "Strongly Disagree" to "Strongly Agree" scale
            labels = [
                "Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"
            ]
            options = SurveyOptionGenerator.generate_likert_options(n=5, answer_texts=labels)
    """

    if only_from_to_scale:
        if len(answer_texts) != 2:
            raise ValueError(
                f"From-To scales require exactly 2 descriptions, but answer_texts was set to '{answer_texts}'."
            )
        if idx_type != "integer":
            raise ValueError(
                f"From-To scales require an integer scale index, but idx_type was set to '{idx_type}'."
            )
    else:
        if answer_texts:
            if len(answer_texts) != n:
                raise ValueError(
                    f"answer_texts and n need to be the same length, but answer_texts has length {len(answer_texts)} and n was given as {n}."
                )
    if even_order:
        if n % 2 == 0:
            raise ValueError(
                "If you want to turn a scale even, it should be odd before."
            )
        middle_index = n // 2
        answer_texts = (
            answer_texts[:middle_index] + answer_texts[middle_index + 1 :]
        )
        n = n - 1
    if add_middle_category:
        if n % 2 != 0:
            raise ValueError(
                "If you want to add a middle category, it should be even before."
            )
        middle_index = n // 2
        answer_texts = answer_texts[:middle_index] + [str_middle_cat] + answer_texts[middle_index :]
        n = n + 1

    if random_order:
        if len(answer_texts) < 2:
            raise ValueError(
                "There must be at least two answer options to reorder randomly."
            )
        random.shuffle(
            answer_texts
        )  # no assignment needed because shuffles already inplace
    if reversed_order:
        if len(answer_texts) < 2:
            raise ValueError(
                "There must be at least two answer options to reorder in reverse."
            )
        answer_texts = answer_texts[::-1]
    
    if add_refusal:
        answer_texts.append("Don't know / Refuse to answer")
        n += 1

    answer_option_indices = []
    if idx_type == "no_index":
        # no index, just the answer options directly
        answer_option_indices = None
    elif idx_type == "integer":
        if add_refusal: # if refusal is added, assign it a common code -99
            for i in range(n - 1):
                answer_code = i + start_idx
                answer_option_indices.append(str(answer_code))
            answer_option_indices.append(refusal_code)  # common code for refusal
        else:
            for i in range(n):
                answer_code = i + start_idx
                answer_option_indices.append(str(answer_code))
    else:
        # TODO @Jens add these to constants.py
        if idx_type == "char_lower":
            for i in range(n):
                answer_option_indices.append(ascii_lowercase[(i + start_idx) % 26])
        elif idx_type == "char_upper":
            for i in range(n):
                answer_option_indices.append(ascii_uppercase[(i + start_idx) % 26])


    
    answer_texts_object = AnswerTexts(
        answer_texts=answer_texts,
        indices=answer_option_indices,
        index_answer_seperator=index_answer_separator,
        option_seperators=options_separator,
        only_scale=only_from_to_scale,
    )

    questionnaire_options = AnswerOptions(
        answer_texts=answer_texts_object,
        from_to_scale=only_from_to_scale,
        list_prompt_template=list_prompt_template,
        scale_prompt_template=scale_prompt_template,
        response_generation_method=response_generation_method,
    )

    return questionnaire_options