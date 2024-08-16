from pathlib import Path
from .nlparam_logging import logger
import numpy as np

# Get the current file's directory.
current_directory = Path(__file__)
TEMPLATE_DIRECTORY = current_directory.parent / "templates"
DATA_DIR = current_directory.parent.parent / "data"
DEFAULT_VALIDATOR_NAME = "gpt-4o-mini"
DEFAULT_EMBEDDER_NAME = "hkunlp/instructor-xl"
DEFAULT_PROPOSER_NAME = "gpt-4o"
EXPERIMENT_VALIDATOR_NAME = "google/flan-t5-xl"
EXPERIMENT_PROPOSER_NAME = "gpt-3.5-turbo-0613"
DEFAULT_LLM_CONFIG = {
    "validator": {"model_name": DEFAULT_VALIDATOR_NAME},
    "proposer": {"model_name": DEFAULT_PROPOSER_NAME},
    "embedder": {"model_name": DEFAULT_EMBEDDER_NAME},
}

from .llm.query import query_wrapper
from .llm.propose import Proposer
from .llm.validate import get_validator_by_name, validate_descriptions, Validator
from .llm.embedder import Embedder
from .models.cluster_model import ClusteringModel
from .models.classification_model import ClassificationModel
from .models.timeseries_model import TimeSeriesModel
from .models.abstract_model import ModelOutput

def continuous_arr(arr, lr=0.005):
    init = np.mean(arr[:50])
    cur = init
    continuous = [cur]
    for i in range(1, len(arr)):
        cur = (1-lr) * cur + lr * arr[i]
        continuous.append(cur)
    return continuous
