from typing import List, Union
from tqdm import tqdm
import time
import openai
import os
from copy import deepcopy
import concurrent.futures
import numpy as np
from transformers import GPT2Tokenizer
import random
import concurrent.futures

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OPENAI_ORG"]
GPT2TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")

MAX_LIMIT = 5
MAX_RETRIES = 20
BATCH_SIZE = 20

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MESSAGE = [
    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
    {"role": "user", "content": None},
]


def get_length_in_gpt2_tokens(text: str) -> int:
    """
    Get the length of a text in GPT2 tokens.

    Parameters
    ----------
    text : str
        The text.

    Returns
    -------
    int
        The length of the text in GPT2 tokens.
    """
    return len(GPT2TOKENIZER.encode(text))


def truncate_based_on_gpt2_tokens(text: str, max_length: int) -> str:
    tokens = GPT2TOKENIZER.encode(text)
    if len(tokens) <= max_length:
        return text
    else:
        return GPT2TOKENIZER.decode(tokens[:max_length])


def get_avg_length(texts: List[str], max_num_samples=500) -> float:
    """
    Get the average length of texts in a list of texts.

    Parameters
    ----------
    texts : List[str]
        A list of texts.
    max_num_samples : int
        The maximum number of texts to sample to compute the average length.

    Returns
    -------
    float
        The average length of texts.
    """
    if len(texts) > max_num_samples:
        sampled_texts = random.sample(texts, max_num_samples)
    else:
        sampled_texts = texts
    avg_length = np.mean([get_length_in_gpt2_tokens(t) for t in sampled_texts])
    return avg_length


# hyperparameters
# in expectation the prompt will have the length (CONTEXT_LENGTH - CORPUS_OVERHEAD) * (1 - CORPUS_BUFFER_FRACTION) to leave room for the overflow and the completion
CORPUS_OVERHEAD = 1024
CORPUS_BUFFER_FRACTION = 0.25


def get_max_num_samples_in_proposer(texts: List[str], proposer_model: str) -> int:
    """
    Get the maximal number of in-context samples based on the context length.Leave a buffer of 25% of the relative context length and 1024 tokens for the absolute context length

    Parameters
    ----------
    texts : List[str]
        A list of texts.

    proposer_model : str
        The model used to propose descriptions.

    Returns
    -------
    int
        The maximal number of in-context samples.
    """
    max_corpus_pair_length = (get_context_length(proposer_model) - CORPUS_OVERHEAD) * (
        1 - CORPUS_BUFFER_FRACTION
    )
    avg_length = get_avg_length(texts)
    max_num_samples = int(max_corpus_pair_length / avg_length)
    return max_num_samples


def get_context_length(model: str) -> int:
    """
    Get the context length for the given model.

    Parameters
    ----------
    model : str
        The model in the API to be used.

    Returns
    -------
    int
        The context length.
    """

    if "gpt-4" in model:
        return 16000
    elif model == "gpt-4-32k":
        return 32000
    elif "gpt-3.5-turbo" in model:
        return 4096
    else:
        raise ValueError(f"Unknown model {model}")


DEFAULT_MESSAGE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": None},
]


def single_chat_gpt_wrapper(args) -> Union[None, str]:
    if args.get("messages") is None:
        args["messages"] = deepcopy(DEFAULT_MESSAGE)
        if args.get("system_prompt") is not None:
            args["messages"][0]["content"] = args["system_prompt"]
        args["messages"][1]["content"] = args["prompt"]
        del args["prompt"]
        if args.get("system_prompt") is not None:
            del args["system_prompt"]

    client = openai.OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for _ in range(10):
        try:
            chat_completion = client.chat.completions.create(**args)
            text_content_response = chat_completion.choices[0].message.content
            return text_content_response
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print(e)
            time.sleep(10)

    return None


def chat_gpt_wrapper_parallel(
    prompts: List[str], num_processes: int = 1, progress_bar: bool = True, **args
) -> List[str]:
    def update_progress_bar(future):
        if progress_bar:
            pbar.update(1)

    if num_processes == 1:
        results = []
        pbar = tqdm(total=len(prompts), desc="Processing") if progress_bar else None
        for prompt in prompts:
            result = single_chat_gpt_wrapper({**args, "prompt": prompt})
            if progress_bar:
                pbar.update(1)
            results.append(result)
        if progress_bar:
            pbar.close()
        return results

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = [
            executor.submit(single_chat_gpt_wrapper, {**args, "prompt": prompt})
            for prompt in prompts
        ]
        pbar = tqdm(total=len(tasks), desc="Processing") if progress_bar else None
        for task in concurrent.futures.as_completed(tasks):
            if progress_bar:
                task.add_done_callback(update_progress_bar)
        results = [task.result() for task in tasks]
    if progress_bar:
        pbar.close()
    return results


def query_wrapper(
    prompts: List[str],
    model: str = "gpt-3.5-turbo", 
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    num_processes: int = 1,
    progress_bar: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[str]:
    """
    Wrapper for querying all LLM APIs.

    Parameters
    ----------
    prompts : List[str]
        List of prompts to query the model with.
    model : str, optional
        Model to query, by default "gpt-3.5-turbo"
    max_tokens : int, optional
        Maximum number of tokens to generate, by default 128
    temperature : float, optional
        Temperature for sampling, by default 0.7
    top_p : float, optional
        Top p for sampling, by default 1.0
    num_processes : int, optional
        Number of processes to use, by default 1

    Returns
    -------
    List[str]
        List of generated texts.
    """
    assert type(prompts) == list
    assert model.startswith("gpt")

    args = {
        "temperature": temperature,
        "top_p": top_p,
        "prompts": prompts,
        "model": model,
        "max_tokens": max_tokens,
        "num_processes": num_processes,
        "progress_bar": progress_bar,
        "system_prompt": system_prompt,
    }
    return chat_gpt_wrapper_parallel(**args)
