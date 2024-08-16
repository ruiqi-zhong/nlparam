import os
import numpy as np
from typing import List, Iterator
from dataclasses import dataclass
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
import torch
from tqdm import trange, tqdm
from nlparam import TEMPLATE_DIRECTORY, DEFAULT_VALIDATOR_NAME, logger
import nlparam.llm.query as query
import time
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TARGET_LENGTH = 2
YES_NO_TOK_IDX = [150, 4273]
MAX_SOURCE_LENGTH = 1024
TEMPERATURE = 0.001
sm = torch.nn.Softmax(dim=-1)
GPT_TEMPLATE = os.path.join(TEMPLATE_DIRECTORY, "gpt_validator.txt")
T5_TEMPLATE = os.path.join(TEMPLATE_DIRECTORY, "t5_validator.txt")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
T5_MULTI_ASSIGNER_TEMPLATE = os.path.join(TEMPLATE_DIRECTORY, "multi_assigner.txt")


def truncate_text(text, max_len=256):
    tokens = gpt2_tokenizer(text)["input_ids"]
    if len(tokens) > max_len:
        return gpt2_tokenizer.decode(tokens[:max_len])
    else:
        return text


@dataclass
class ValidatorInput:
    """A validator input, consisting of a hypothesis and a text."""

    hypothesis: str
    text: str


class Validator:
    """A validator to validate a hypothesis given a text; abstract class."""

    def obtain_scores(self, validator_inputs: List[ValidatorInput]):
        raise NotImplementedError

    def validate(self, texts: List[str], hypotheses: List[str], verbose=False):
        """
        Given a list of texts and a list of hypotheses, return a list of scores, which the i-th score is how well the i-th hypothesis satisfies the i-th text.

        Parameters
        ----------
        texts : List[str]
            A list of texts.
        hypotheses : List[str]
            A list of hypotheses.

        Returns
        -------
        List[float]
            A list of scores, which the i-th score is how well the i-th hypothesis satisfies the i-th text.
        """
        assert len(texts) == len(hypotheses)

        validator_inputs = [
            ValidatorInput(hypothesis=hypothesis, text=text)
            for hypothesis, text in zip(hypotheses, texts)
        ]
        result = list(self.obtain_scores(validator_inputs, verbose=verbose))
        return result

    def stop_count(self):
        self.counting = False

    def start_count(self):
        self.counting = True


@dataclass
class MultiAssignerInput:
    """A assigner input, consisting of a list of candidate_explanation and a text."""

    candidate_explanation: List[str]
    text: str


class DummyValidator(Validator):
    def __init__(self):
        self.count = 0
        self.name = "validator_dummy"
        self.counting = True
        self.total_time_spent = 0

    def validate(self, texts: List[str], hypotheses: List[str]):
        result = [1.0 if h in t.split() else 0.0 for t, h in zip(texts, hypotheses)]
        if self.counting:
            self.count += len(result)
        return result

    def get_multi_text_description_matching(
        self, descriptions, texts, add_null_description=False, verbose=False
    ):
        results = []
        for text in texts:
            v = []
            one_added = False
            for description in descriptions:
                if not one_added and description in text:
                    v.append(1)
                    one_added = True
                else:
                    v.append(0)
            if not one_added:
                v[-1] = 1
            results.append(v)
        return results

    def obtain_scores(self, validator_inputs: List[ValidatorInput], verbose=False):
        for input in validator_inputs:
            if self.counting:
                self.count += 1
            yield 1.0 if input.hypothesis in input.text.split() else 0.0

    def clear_count(self):
        self.count = 0

def parse_template(template: str) -> str:
    """
    A helper function to parse the template, which can be either a string or a path to a file.
    """
    if os.path.exists(template):
        with open(template, "r") as f:
            return f.read()
    else:
        return template


def create_prompt_inputs_for_multi_assigner(
    template: str,
    assigner_inputs: List[MultiAssignerInput],
    add_null_description: bool = True,
):
    """
    A helper function to create the prompt inputs for multi assigner.

    Parameters
    ----------
    template: str
        The template used for assignment
    assigner_inputs : List[MultiAssignerInput]
        A list of MultiAssignerInput.
    add_null_description : bool
        Whether to add a null description to the candidate_explanation.

    Returns
    -------
    List[str]
        A list of prompts.
    """
    template = parse_template(template)
    prompts = [
        template.format(
            descriptions_with_index="\n".join(
                [
                    f"{i}. {description}"
                    for i, description in enumerate(
                        (input.candidate_explanation + ["none of the above"])
                        if add_null_description
                        else input.candidate_explanation
                    )
                ]
            ),
            text=input.text,
        )
        for input in assigner_inputs
    ]
    return prompts


def parse_mutli_assigner_output(response, num_descriptions):
    """
    A parser for multi assigner, which parses the response into a list of 0/1, where 1 means the description is satisfied.
    The expected format of the response is a list of integers, stringified.

    Parameters
    ----------
    response : str
        The response from the model to be parsed.
    num_descriptions : int
        The number of descriptions, for the backup choice of empty parsed result.

    Returns
    -------
    List[int]
        A list of 0/1, where 1 means the description is satisfied.
    """
    try:
        # this can deal with some errors
        opening_bracket = response.find("[")
        closing_bracket = response.find("]")
        response = response[opening_bracket : closing_bracket + 1]
        answer = json.loads(response)
        assert isinstance(answer, list)
        answer = list(map(int, answer))
        matched = [0] * num_descriptions
        for i in answer:
            if i < num_descriptions:
                matched[i] = 1
        return matched
    except Exception as e:
        # empty is kind of frequent, so let's not alert in this case
        if response.strip() == "":
            return [0] * num_descriptions
        print(
            "Assigner failed to parse the response. Treating it as empty. Error message: ",
            e,
        )
        print("The response to parse:", response)
        return [0] * num_descriptions


class D5Validator(Validator):
    """A validator based on T5 model to validate a hypothesis given a text"""

    BATCH_SIZE = 16
    with open(T5_TEMPLATE, "r") as f:
        DEFAULT_VALIDATOR_TEMPLATE = f.read()

    def __init__(
        self,
        model_path: str = DEFAULT_VALIDATOR_NAME,
        batch_size: int = BATCH_SIZE,
        template: str = DEFAULT_VALIDATOR_TEMPLATE,
        multi_assigner_template_path: str = T5_MULTI_ASSIGNER_TEMPLATE,
    ):
        """
        Initialize the validator

        Parameters
        ----------
        model_path : str
            The path to the T5 model weights used for validation
        batch_size : int
            The batch size used for validation
        template : str
            The template used for validation
        """

        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        print("loading model weights")
        self.model = (
            T5ForConditionalGeneration.from_pretrained(
                model_path  # , device_map="balanced"
            )
            .to(torch.bfloat16)
            .to(device)
        )
        self.model_name = os.path.basename(model_path)
        print("done")
        self.validator_template = template

        with open(multi_assigner_template_path, "r") as f:
            self.multi_assigner_template = f.read()
        self.batch_size = batch_size
        self.prompt2result = {}
        self.count = 0
        self.total_time_spent = 0
        self.name = f"validator_{self.model_name}"
        self.counting = True
        self.verbose = True

    def clear_count(self):
        self.count = 0
        self.prompt2result = {}

    def get_prompts_from_validator_inputs(self, validator_inputs: List[ValidatorInput]):
        prompts = []
        for validator_dict in validator_inputs:
            prompt = self.validator_template.format(
                hypothesis=validator_dict.hypothesis, text=validator_dict.text
            )
            prompts.append(prompt)
        return prompts

    def batch_inference(self, prompts: List[str], max_new_tokens=1, verbose=False):
        """
        Given a list of prompts, return a list of generated results in a batch manner.

        Parameters
        ----------
        prompts : List[str]
            A list of prompts.
        max_new_tokens : int
            The maximum number of tokens to be generated.

        Returns
        -------
        List[str]
            A list of generated results.

        """
        with torch.no_grad():
            self.model.eval()
            num_batches = (len(prompts) - 1) // self.batch_size + 1

            pbar = trange(num_batches) if verbose else range(num_batches)
            for batch_idx in pbar:
                input_prompts = prompts[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                inputs = self.tokenizer(
                    input_prompts,
                    return_tensors="pt",
                    padding="longest",
                    max_length=MAX_SOURCE_LENGTH,
                    truncation=True,
                ).to(device)
                generation_result = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=True,
                    temperature=0.001,
                    max_new_tokens=10,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                yield generation_result

    def get_multi_text_description_matching(
        self, descriptions, texts, add_null_description=False, verbose=False
    ):
        multi_assigner_inputs = [
            MultiAssignerInput(descriptions, text) for text in texts
        ]
        return self.obtain_multi_assigner_scores(
            multi_assigner_inputs, add_null_description, verbose
        )

    def obtain_multi_assigner_scores(
        self,
        assigner_inputs: List[MultiAssignerInput],
        add_null_description: bool = True,
        verbose=True,
    ) -> Iterator[List[int]]:
        num_descriptions = len(assigner_inputs[0].candidate_explanation)
        # assert all(
        #     len(input.candidate_explanation) == num_descriptions
        #     for input in assigner_inputs
        # )

        prompts = create_prompt_inputs_for_multi_assigner(
            self.multi_assigner_template, assigner_inputs, add_null_description
        )
        if verbose:
            print("First prompt as an example to assigner:")
            print(prompts[0])
        for generation_result in self.batch_inference(
            prompts, max_new_tokens=10, verbose=verbose
        ):
            responses = self.tokenizer.batch_decode(
                generation_result.sequences, skip_special_tokens=True
            )
            for response in responses:
                yield parse_mutli_assigner_output(response, num_descriptions)

    def obtain_scores(
        self, validator_inputs: List[ValidatorInput], verbose: bool = False
    ) -> Iterator[float]:
        """
        Given a list of ValidatorInput, return a list of scores, which the i-th score is how well the i-th ValidatorInput satisfies the description.

        Parameters
        ----------
        validator_inputs : List[ValidatorInput]
            A list of ValidatorInput.
        verbose : bool
            Whether to show a progress bar.

        Returns
        -------
        List[float]
            A list of scores, which the i-th score is how well the i-th ValidatorInput satisfies the description.
        """

        prompts = self.get_prompts_from_validator_inputs(validator_inputs)
        with torch.no_grad():
            self.model.eval()

            uncomputed_prompts = list(
                {p for p in prompts if p not in self.prompt2result}
            )
            if self.counting:
                self.count += len(uncomputed_prompts)

            if len(uncomputed_prompts) > 0:
                num_batches = (len(uncomputed_prompts) - 1) // self.batch_size + 1

                if verbose:
                    pbar = trange(num_batches)
                    pbar.set_description("inference")
                else:
                    pbar = range(num_batches)

                for batch_idx in pbar:
                    start_time = time.time()
                    input_prompts = uncomputed_prompts[
                        batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                    ]
                    inputs = self.tokenizer(
                        input_prompts,
                        return_tensors="pt",
                        padding="longest",
                        max_length=MAX_SOURCE_LENGTH,
                        truncation=True,
                    ).to(device)
                    generation_result = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        do_sample=True,
                        temperature=0.001,
                        max_new_tokens=2,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    decoded_strs = self.tokenizer.batch_decode(
                        generation_result.sequences,
                        skip_special_tokens=True,
                        return_dict_in_generate=True,
                        clean_up_tokenization_spaces=False,
                    )
                    uncomputed_scores = [
                        1 if "yes" in t.lower() else 0.0 for t in decoded_strs
                    ]
                    assert len(uncomputed_scores) == len(input_prompts)
                    end_time = time.time()
                    if self.counting:
                        self.total_time_spent += end_time - start_time
                    for i in range(len(input_prompts)):
                        self.prompt2result[input_prompts[i]] = uncomputed_scores[i]

            scores = [self.prompt2result[p] for p in prompts]
            for s in scores:
                yield s

    def set_oracle(self, descriptions, texts, responses):
        responses = np.array(responses)
        assert responses.shape == (len(texts), len(descriptions))
        response_flattened = []
        validator_inputs = []

        for i in range(len(texts)):
            for j in range(len(descriptions)):
                response_flattened.append(responses[i, j])
                validator_inputs.append(ValidatorInput(descriptions[j], texts[i]))

        prompts = self.get_prompts_from_validator_inputs(validator_inputs)
        for i in range(len(prompts)):
            self.prompt2result[prompts[i]] = response_flattened[i]
        logger.debug(f"Set response for {len(prompts)} prompts")


class GPTValidator(Validator):
    def __init__(
        self, model: str, template_path: str = None, multi_assigner_template_path=None
    ):
        """
        Parameters
        ----------
        model : str
            The GPT model to use.
        """
        super().__init__()
        self.model = model
        self.model_name = model

        if template_path is None:
            if "gpt" in model:
                template_path = GPT_TEMPLATE
            else:
                raise ValueError(f"Unknown model {model}")

        if multi_assigner_template_path is None:
            if "gpt" in model:
                multi_assigner_template_path = T5_MULTI_ASSIGNER_TEMPLATE
            else:
                raise ValueError(f"Unknown model {model}")

        with open(template_path, "r") as f:
            self.template = f.read()

        with open(multi_assigner_template_path, "r") as f:
            self.multi_assigner_template = f.read()
        self.count = 0
        self.prompt2restuls = {}

    def obtain_scores(
        self, validator_inputs: List[ValidatorInput], verbose: bool = False
    ):
        """
        Given a list of ValidatorInput, return a list of scores, which the i-th score is how well the i-th ValidatorInput satisfies the description.

        Parameters
        ----------
        validator_inputs : List[ValidatorInput]
            A list of ValidatorInput.
        verbose : bool
            Whether to print progress bar

        Returns
        -------
        List[float]
            A list of scores, which the i-th score is how well the i-th ValidatorInput satisfies the description.
        """

        # construct the prompts
        prompts = [
            self.template.format(hypothesis=input.hypothesis, text=input.text)
            for input in validator_inputs
        ]
        unprocessed_prompts = list(
            {p for p in prompts if p not in self.prompt2restuls}
        )
        num_processes = 10
        answers = []
        for response in query.query_wrapper(
            unprocessed_prompts,
            model=self.model,
            num_processes=num_processes,
            temperature=0.0,
            progress_bar=verbose,
            max_tokens=3,
        ):
            self.count += 1
            answers.append(1 if "yes" in response.lower() else 0)
        for prompt, answer in zip(unprocessed_prompts, answers):
            self.prompt2restuls[prompt] = answer
        
        for prompt in prompts:
            yield self.prompt2restuls[prompt]

    def get_multi_text_description_matching(
        self, descriptions, texts, add_null_description=False, verbose=False
    ):
        multi_assigner_inputs = [
            MultiAssignerInput(descriptions, text) for text in texts
        ]
        return self.obtain_multi_assigner_scores(
            multi_assigner_inputs, add_null_description, verbose
        )

    def obtain_multi_assigner_scores(
        self,
        assigner_inputs: List[MultiAssignerInput],
        add_null_description: bool = True,
        verbose=True,
    ) -> Iterator[List[int]]:
        num_descriptions = len(assigner_inputs[0].candidate_explanation)
        assert all(
            len(input.candidate_explanation) == num_descriptions
            for input in assigner_inputs
        )

        prompts = create_prompt_inputs_for_multi_assigner(
            self.multi_assigner_template, assigner_inputs, add_null_description
        )

        responses = query.query_wrapper(
            prompts,
            model=self.model,
            num_processes=5,
            temperature=0.0,
            progress_bar=verbose,
            max_tokens=3,
        )
        for response in responses:
            yield parse_mutli_assigner_output(response, num_descriptions)


def get_validator_by_name(model_name, **kwargs):
    if "t5" in model_name:
        return D5Validator(model_name)
    elif "gpt" in model_name:
        return GPTValidator(model_name)
    elif model_name == "dummy":
        return DummyValidator()
    else:
        raise ValueError(f"Unknown validator {model_name}")


def validate_descriptions(
    descriptions: List[str],
    texts: List[str],
    validator: Validator,
    progress_bar: bool = False,
) -> np.ndarray:
    """
    Given a list of descriptions and a list of texts, return a matrix of scores, which the i-th row and j-th column is how well the i-th text satisfies the j-th description.

    Parameters
    ----------
    descriptions : List[str]
        A list of descriptions to be validated.
    texts : List[str]
        A list of texts to be validated.
    validator : Validator
        A validator that can validate a list of ValidatorInput. Could be either a T5Validator or a GPT35Validator.
    progress_bar : bool, optional
        Whether to show a progress bar, by default False

    Returns
    -------
    np.ndarray
        A matrix of scores, which the i-th row and j-th column is how well the i-th text satisfies the j-th description.
    """
    # aggregate all the validator inputs

    # check whether the validator is a subclass of Validator
    if issubclass(type(validator), Validator):
        validator_inputs = []
        deduped_texts = list(set(texts))
        text2idx = {t: i for i, t in enumerate(deduped_texts)}
        orig_idxes = [text2idx[t] for t in texts]
        for text in tqdm(deduped_texts, desc="Generating validator inputs"):
            truncated_text = truncate_text(text)
            for description in descriptions:
                validator_inputs.append(
                    ValidatorInput(hypothesis=description, text=truncated_text)
                )

        # obtain the scores
        # scores = list(validator.obtain_scores(validator_inputs))
        scores = []
        start_time = time.time()
        for score in validator.obtain_scores(validator_inputs, verbose=progress_bar):
            scores.append(score)
        end_time = time.time()
        duration = int(end_time - start_time)
        logger.debug(f"Validation duration: {duration}s")

        # reshape the scores into a matrix
        # the i-th row and j-th column is how well the i-th text satisfies the j-th description
        scores = np.array(list(scores)).reshape(len(deduped_texts), len(descriptions))
        scores = scores[orig_idxes]
        return scores
    else:
        raise ValueError(f"Unknown validator {validator}")
