import random
import nlparam.llm.query as query
from nlparam import TEMPLATE_DIRECTORY
from typing import List, Dict
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from collections import defaultdict
import numpy as np
import os
from nlparam import logger
import time
from itertools import chain


NUM_MAX_RETRY = 2
PROPOSER_MAX_PER_SAMPLE_LENGTH = 256
MUTE_KEYINFO_IN_PROPOSER = True
DEFAULT_GOAL = "I want to guess what type of text is more likely to appear later within this particular the prompt"
TEMPLATE_DICTs = {}
for with_score in [True, False]:
    for simple in [True, False]:
        template_path = os.path.join(
            TEMPLATE_DIRECTORY,
            f"{'no_score_' if not with_score else 'reg_'}proposer{'_simple' if simple else ''}.txt",
        )
        with open(template_path, "r") as f:
            TEMPLATE_DICTs[(with_score, simple)] = f.read()
with open(os.path.join(TEMPLATE_DIRECTORY, "reg_proposer_application.txt")) as f:
    TEMPLATE_DICTs["application"] = f.read()


def dedup_by_prefix(ls):
    prefix2s = {l.split(";")[0]: l for l in sorted(ls)}
    return list(prefix2s.values())


def strip_doublelinebreaks(text: str) -> str:
    """
    Strip double line breaks from text.

    Parameters
    ----------
    text : str
        The text to strip double line breaks from.

    Returns
    -------
    str
        The text with double line breaks stripped.
    """
    return text.replace("\n\n", "\n")


def filter_header(s):
    lines = s.split("\n")
    if len(lines) > 0:
        line = lines[0].strip()
        if len(line) > 0 and line[0] == ":":
            lines = lines[1:]
    return "\n".join(lines)


def clean_description(x):
    x = x.strip()
    if len(x) > 0:
        if x[0] == "-":
            x = x[1:]

    x = x.strip()
    if len(x) > 0:
        if x[0] == '"':
            x = x[1:]

    x = x.strip()
    return x


def is_header(s):
    header_text = "here are"
    s = s.lower()
    if s[: len(header_text)] == header_text:
        return True
    if s.strip()[-1] == ":":
        return True
    return False


def parse_description_responses(response: str) -> List[str]:
    """
    Parse the description responses from the proposer model.

    Parameters
    ----------
    response : str
        The response from the proposer model, each description is separated by a newline, surrounded by quotes. We will extract the description within the quotes for each line.

    Returns
    -------
    List[str]
        A list of descriptions.
    """
    response = filter_header(response)
    descriptions = []
    for line_id, line in enumerate(response.split("\n-")):
        # find the two quotes
        start, end = (line.find('"') if line_id != 0 else -1), line.rfind('"')
        description = line[start + 1 : end]
        if description != "":
            description = clean_description(description)
            if description != "" and not is_header(description):
                descriptions.append(description)
    descriptions = list(set(descriptions))
    return descriptions


def construct_no_score_proposer_prompt(
    text_samples: List[str],
    goal: str,
    example_descriptions: List[str],
    num_descriptions_per_prompt: int,
    template: str = None,
):
    text_samples = "\n".join(
        [
            f"Sample {i}: {query.truncate_based_on_gpt2_tokens(text, PROPOSER_MAX_PER_SAMPLE_LENGTH)}"
            for i, text in enumerate(text_samples)
        ]
    )
    example_description_in_prompt = ""

    if len(example_descriptions) > 0:
        example_description_in_prompt = "Here are some example hypotheses you have generated; please generate something in the same format but different in content:\n"
        example_description_in_prompt = (
            "\n"
            + "\n".join(
                f'"{example_description.lower()}"'
                for example_description in example_descriptions
            )
            + "\n"
        )

    prompt = template.format(
        goal=goal,
        samples_in_prompt=text_samples,
        example_description_in_prompt=example_description_in_prompt,
        num_descriptions_per_prompt=num_descriptions_per_prompt,
    )
    return prompt


def construct_proposer_prompt(
    text_samples: List[str],
    scores: List[float],
    goal: str,
    example_descriptions: List[str],
    num_descriptions_per_prompt: int,
    template: str = None,
) -> str:
    """
    Construct the prompt for the proposer model.

    Parameters
    ----------
    text_samples : List[str]
        The text samples to be included in the prompt.
    scores : List[float]
        The score for each text sample.
    goal : str
        The goal or objective the proposer model should follow.
    example_descriptions : List[str], optional
        A list of example descriptions provided for formatting reference.
    num_descriptions_per_prompt : int
        The number of descriptions the model should suggest.
    template : str, optional
        The template to use for the prompt, by default None

    Returns
    -------
    str
        The formatted prompt for the proposer model.
    """

    assert len(text_samples) == len(scores)

    # sort the text samples by score
    text_samples = [
        text
        for _, text in sorted(
            zip(scores, text_samples), key=lambda pair: pair[0], reverse=False
        )
    ]
    scores = sorted(scores, reverse=False)

    text_samples_with_scores = [
        f"Sample {i}: {query.truncate_based_on_gpt2_tokens(text, PROPOSER_MAX_PER_SAMPLE_LENGTH)} (score: {score:.1f})"
        for i, (text, score) in enumerate(zip(text_samples, scores))
    ]

    samples_in_prompt_w_scores = "\n".join(text_samples_with_scores)

    example_description_in_prompt = ""
    if len(example_descriptions) > 0:
        example_description_in_prompt = "Here are some example hypotheses you have generated; please generate something in the same format but different in content:\n"
        example_description_in_prompt = (
            "\n"
            + "\n".join(
                f'"{example_description.lower()}"'
                for example_description in example_descriptions
            )
            + "\n"
        )

    if goal == "":
        goal = DEFAULT_GOAL
    prompt = template.format(
        goal=goal,
        samples_in_prompt_w_scores=samples_in_prompt_w_scores,
        example_description_in_prompt=example_description_in_prompt,
        num_descriptions_per_prompt=num_descriptions_per_prompt,
    )
    return prompt


@dataclass_json
@dataclass
class RegProposerResponse:
    """
    The response from the proposer model.

    Attributes
    ----------
    descriptions : List[str]
        A list of descriptions for the difference between the two Corpora.
    proposer_prompt : str
        The prompt used for the proposer model.
    texts_subset : List[str]
        The text samples used in the prompt.
    target_subset : List[float]
        The target scores for the text samples.
    estimated_cost : float
        The estimated cost of running the proposer model.
    raw_responses: List[str]
        The raw responses from the proposer model, each being one raw response to the proposer prompt.
    original_descriptions : List[str]
        The original descriptions proposed by LLM
    """

    descriptions: List[str]
    proposer_prompt: str
    texts_subset: List[str]
    target_subset: List[float]
    raw_responses: List[str]
    original_descriptions: List[str] = None
    duration: float = 0.0


NULL_REG_PROPOSER_RESPONSE = RegProposerResponse(
    descriptions=[],
    proposer_prompt="",
    texts_subset=[],
    target_subset=[],
    raw_responses=[],
    original_descriptions=None,
)


CORPUS_PAIR_OVERHEAD = 1024
CORPUS_BUFFER_FRACTION = 0.25


def get_max_num_samples_in_proposer(texts, proposer_model: str) -> int:
    """
    Get the maximal number of in-context samples based on the context length.Leave a buffer of 25% of the relative context length and 1024 tokens for the absolute context length

    Parameters
    ----------
    problem : Problem
        The D5 problem to solve.
    proposer_model : str
        The model used to propose descriptions.

    Returns
    -------
    int
        The maximal number of in-context samples.
    """
    max_corpus_pair_length = (
        query.get_context_length(proposer_model) - CORPUS_PAIR_OVERHEAD
    ) * (1 - CORPUS_BUFFER_FRACTION)
    avg_length = min(query.get_avg_length(texts), PROPOSER_MAX_PER_SAMPLE_LENGTH) + 10
    max_num_samples = int(max_corpus_pair_length / (avg_length + 10))
    return max_num_samples


fencing_phrases = ["unfortunately", "enough information", "i apologize"]


def is_fencing(response):
    response = response.lower()
    for phrase in fencing_phrases:
        if phrase in response:
            return True
    return False


def propose_descriptions_no_score(
    texts: List[str] = None,
    goal: str = None,
    example_descriptions: List[str] = [],
    model: str = "gpt-4",
    num_descriptions_per_prompt: int = 5,
    num_samples: int = -1,
    random_seed: int = 0,
    debug: bool = False,
    num_rounds: int = 1,
    template: str = None,
):
    random.seed(random_seed)

    if debug:
        all_tokens = {tok for text in texts for tok in text.split()}
        selected_descriptions = list(all_tokens)[
            : num_descriptions_per_prompt * num_rounds
        ]
        response = RegProposerResponse(
            descriptions=selected_descriptions,
            proposer_prompt="",
            texts_subset=None,
            target_subset=None,
            raw_responses=[""],
            original_descriptions=None,
        )
        return response

    if num_samples == -1:
        num_samples = get_max_num_samples_in_proposer(texts, model)

    sampled_idxes = random.sample(range(len(texts)), min(num_samples, len(texts)))

    samples_in_prompt = [texts[i] for i in sampled_idxes]
    prompt = construct_no_score_proposer_prompt(
        text_samples=samples_in_prompt,
        goal=goal,
        example_descriptions=example_descriptions,
        num_descriptions_per_prompt=num_descriptions_per_prompt,
        template=template,
    )
    prompt = strip_doublelinebreaks(prompt)

    start_time = time.time()
    raw_responses = []

    for _ in range(NUM_MAX_RETRY):
        raw_responses_ = query.query_wrapper(
            [prompt] * num_rounds,
            model=model,
            temperature=1.0,
            num_processes=min(4, num_rounds),
        )
        raw_responses_ = [
            r for r in raw_responses_ if r is not None and not is_fencing(r)
        ]
        raw_responses += raw_responses_
        if len(raw_responses) >= num_rounds:
            raw_responses = raw_responses[:num_rounds]
            break

    if len(raw_responses) == 0:
        logger.debug("Proposer failed to generate any response")
        return NULL_REG_PROPOSER_RESPONSE
    end_time = time.time()
    duration = int(end_time - start_time)
    logger.debug(f"Proposer duration: {duration}s")

    original_descriptions = list(
        chain(
            *[
                parse_description_responses(raw_response)
                for raw_response in raw_responses
            ]
        )
    )

    logger.debug("Proposer prompt and response:")
    logger.debug(prompt)
    logger.debug(original_descriptions)


    return RegProposerResponse(
        descriptions=original_descriptions[: num_descriptions_per_prompt * num_rounds],
        proposer_prompt=prompt,
        texts_subset=samples_in_prompt,
        target_subset=None,
        raw_responses=[""],
        original_descriptions=original_descriptions,
        duration=duration,
    )


def propose_descriptions(
    texts: List[str] = None,
    target: List[float] = None,
    goal: str = None,
    example_descriptions: List[str] = [],
    model: str = "gpt-4",
    num_descriptions_per_prompt: int = 5,
    num_samples: int = -1,
    random_seed: int = 0,
    num_extreme_examples: int = 5,
    debug: bool = False,
    num_rounds: int = 1,
    template: str = None,
) -> RegProposerResponse:
    """
    Propose descriptions for a given problem.

    Parameters
    ----------
    texts : List[str]
        The text samples to be included in the prompt.
    target : List[float]
        The target score for each text sample.
    goal : str
        The goal or objective the proposer model should follow.
    example_descriptions : List[str]
        A list of example descriptions provided for formatting reference.
    model : str
        The model to use for proposing descriptions.
    num_descriptions_per_prompt : int
        The number of descriptions the model should suggest.
    num_samples : int
        The number of text samples to be included in the prompt.
    random_seed : int
        The random seed for sampling text samples.
    num_extreme_examples : int
        The number of extreme examples to include in the prompt, by default 5
    debug : bool
        Whether to print debug information. Return the most frequent tokens in the text samples.

    Returns
    -------
    RegProposerResponse
        The response from the proposer model.
    """
    # set the random seed
    random.seed(random_seed)
    all_texts = texts

    assert len(all_texts) == len(target)

    if debug:
        token2score = defaultdict(list)
        all_tokens = list(set(tok for text in all_texts for tok in text.split()))

        X = np.zeros((len(all_texts), len(all_tokens)))

        for i in range(len(all_texts)):
            text = all_texts[i]
            toks = text.split()
            for tok in toks:
                X[i, all_tokens.index(tok)] += 1
        Y = np.array(target)
        Y -= np.mean(Y)
        X -= np.mean(X, axis=0)

        token2score = {tok: X[:, all_tokens.index(tok)].dot(Y) for tok in all_tokens}

        toks = list(token2score.keys())
        random.shuffle(toks)
        sorted_toks = sorted(toks, key=lambda tok: token2score[tok], reverse=True)
        response = RegProposerResponse(
            descriptions=sorted_toks[:num_descriptions_per_prompt],
            proposer_prompt="",
            texts_subset=all_texts,
            target_subset=target,
            raw_responses=[""],
        )
        return response

    if num_samples == -1:
        num_samples = get_max_num_samples_in_proposer(all_texts, model)

    num_extreme_examples = min(num_extreme_examples, num_samples // 2)

    num_datapoints = len(target)
    sample_idxes_sorted_by_score = sorted(
        range(num_datapoints), key=lambda i: target[i], reverse=True
    )

    top_idxes = sample_idxes_sorted_by_score[:num_extreme_examples]
    bottom_idxes = sample_idxes_sorted_by_score[-num_extreme_examples:]

    num_sampled_random_idxes = min(
        num_datapoints, max(num_samples - 2 * num_extreme_examples, 0)
    )

    random_idxes = random.sample(range(num_datapoints), num_sampled_random_idxes)

    all_idxes_in_prompt = top_idxes + bottom_idxes + random_idxes

    text_samples_in_prompt = [all_texts[idx] for idx in all_idxes_in_prompt]
    scores_in_prompt = [target[idx] for idx in all_idxes_in_prompt]

    # construct the prompt based on the text samples and the goal
    proposer_prompt = construct_proposer_prompt(
        text_samples=text_samples_in_prompt,
        scores=scores_in_prompt,
        goal=goal,
        num_descriptions_per_prompt=num_descriptions_per_prompt,
        example_descriptions=example_descriptions,
        template=template,
    )
    proposer_prompt = strip_doublelinebreaks(proposer_prompt)

    start_time = time.time()
    # get the response from the model

    for _ in range(NUM_MAX_RETRY):
        raw_responses = query.query_wrapper(
            [proposer_prompt] * num_rounds,
            model=model,
            temperature=1.0,
            num_processes=min(4, num_rounds),
        )
        raw_responses = [
            r for r in raw_responses if r is not None and not is_fencing(r)
        ]
        if len(raw_responses) >= num_rounds:
            raw_responses = raw_responses[:num_rounds]
            break
    if len(raw_responses) == 0:
        logger.debug("Proposer failed to generate any response")
        return NULL_REG_PROPOSER_RESPONSE
    end_time = time.time()
    duration = int(end_time - start_time)
    logger.debug(f"Proposer duration: {duration}s")
    # parse the response to get the descriptions
    # each description is separated by a newline, surrounded by quotes according to the prompt
    original_descriptions = list(
        chain(
            *[
                parse_description_responses(raw_response)
                for raw_response in raw_responses
            ]
        )
    )

    logger.debug("Proposer prompt and response:")
    logger.debug(proposer_prompt)
    logger.debug(original_descriptions)


    return RegProposerResponse(
        descriptions=original_descriptions[: num_descriptions_per_prompt * num_rounds],
        proposer_prompt=proposer_prompt,
        texts_subset=text_samples_in_prompt,
        target_subset=scores_in_prompt,
        raw_responses=raw_responses,
        original_descriptions=original_descriptions,
        duration=duration,
    )


def transform_gpt_template_to_claude_template(gpt_template):
    claude_template = "\n\nHuman: " + gpt_template
    claude_template = claude_template.replace(
        "Your responses are:", "\n\nAssistant: Here is a list of my guesses:"
    )
    return claude_template


class Proposer:
    def __init__(
        self,
        model_name: str,
        num_descriptions_per_prompt: int = 5,
        num_samples: int = -1,
        random_seed: int = 0,
        num_extreme_examples: int = 5,
        num_rounds: int = 1,
        simple_predicate=False,
        root_template_path=None,
        extract_application=False,
        **kwargs,
    ):
        self.model = model_name
        self.num_descriptions_per_prompt = num_descriptions_per_prompt
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.num_extreme_examples = num_extreme_examples
        self.num_rounds = num_rounds
        self.debug = model_name == "dummy"
        self.simple_predicate = simple_predicate
        self.name = (
            f"proposer_{model_name}_{'simple' if simple_predicate else 'complex'}"
        )
        self.goal2oracle_descriptions = defaultdict(list)
        self.total_time_spent = 0
        self.count = 0
        self.extract_application = extract_application

        if "claude" in model_name:
            self.model_family = "claude"
        elif "gpt" in model_name:
            self.model_family = "gpt"
        elif model_name == "dummy":
            self.model_family = "dummy"
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.root_template_path = root_template_path

    def set_oracle(self, goal2oracle_descriptions: Dict[str, List[str]]):
        self.goal2oracle_descriptions = goal2oracle_descriptions

    def propose_descriptions(
        self,
        texts: List[str],
        target: List[float],
        goal: str,
    ) -> RegProposerResponse:
        if not self.extract_application:
            template = TEMPLATE_DICTs[(True, self.simple_predicate)]
        else:
            template = TEMPLATE_DICTs["application"]
        if self.model_family == "claude":
            template = transform_gpt_template_to_claude_template(template)

        if self.root_template_path is not None:
            with open(self.root_template_path, "r") as f:
                template = f.read()
        proposer_response = propose_descriptions(
            texts=texts,
            target=target,
            goal=goal,
            model=self.model,
            num_descriptions_per_prompt=self.num_descriptions_per_prompt,
            num_samples=self.num_samples,
            random_seed=self.random_seed,
            num_extreme_examples=self.num_extreme_examples,
            num_rounds=self.num_rounds,
            debug=self.debug,
            template=template,
        )
        proposer_response.descriptions += self.goal2oracle_descriptions[goal]
        self.total_time_spent += proposer_response.duration
        self.count += self.num_rounds
        logger.debug(f"Proposer response: {proposer_response.descriptions}")
        return proposer_response

    def propose_descriptions_no_score(
        self, texts: List[str], goal: str
    ) -> RegProposerResponse:
        template = TEMPLATE_DICTs[(False, self.simple_predicate)]
        if self.model_family == "claude":
            template = transform_gpt_template_to_claude_template(template)
        return propose_descriptions_no_score(
            texts=texts,
            goal=goal,
            model=self.model,
            num_descriptions_per_prompt=self.num_descriptions_per_prompt,
            num_samples=self.num_samples,
            random_seed=self.random_seed,
            num_rounds=self.num_rounds,
            debug=self.debug,
            template=template,
        )
