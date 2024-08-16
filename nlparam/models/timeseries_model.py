import random
import numpy as np
import torch
from nlparam.models.abstract_model import (
    AbstractModel,
    NUM_ITERATIONS,
    NUM_CANDIDATE_PREDICATES_TO_EXPLORE,
    NUM_SAMPLES_TO_COMPUTE_PERFORMANCE,
    TEMPERATURE,
    kmeans_pp_init,
)
from nlparam import Embedder, get_validator_by_name, Proposer, DEFAULT_EMBEDDER_NAME, DEFAULT_VALIDATOR_NAME, DEFAULT_PROPOSER_NAME

from typing import Set, List
from tqdm import trange
from nlparam import logger

device = "cuda" if torch.cuda.is_available() else "cpu"
lsm = torch.nn.LogSoftmax(dim=1)


def optimize_ts(
    text_idx_by_time: List[int],
    embeddings: np.ndarray,
    discontinuity_penalty: float,
    full_phi_denotation: np.ndarray,
    w_init: np.ndarray,
    optimize_w: bool,
    optimized_ks: Set[int],
    seed: int,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    assert full_phi_denotation.shape[0] == embeddings.shape[0]

    orig_denotation = full_phi_denotation.copy()
    orig_denotation[:, list(optimized_ks)] = 0
    orig_denotation = torch.tensor(orig_denotation, dtype=torch.float32).to(device)

    N, K = full_phi_denotation.shape
    N, D = embeddings.shape

    continuous_denotation_added_mask = np.zeros_like(full_phi_denotation)
    continuous_denotation_added_mask[:, list(optimized_ks)] = 1
    continuous_denotation_added_mask = torch.tensor(
        continuous_denotation_added_mask, dtype=torch.float32
    ).to(device)

    continuous_predicate_representation_init = kmeans_pp_init(embeddings, K)

    embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
    continuous_predicate_representation_parameters = torch.nn.Parameter(
        torch.tensor(continuous_predicate_representation_init, dtype=torch.float32).to(
            device
        )
    )
    continuous_predicate_representation_parameters.requires_grad = True
    assert continuous_predicate_representation_parameters.shape == (K, D)

    if w_init is None:
        w_init = np.zeros((len(text_idx_by_time), K), dtype=np.float32)
    w = torch.tensor(w_init, dtype=torch.float32).to(device)
    w.requires_grad = optimize_w

    optim = torch.optim.Adam(
        [continuous_predicate_representation_parameters, w], lr=1e-3
    )

    pbar = trange(50000)

    fixed_denotation = orig_denotation.clone().detach()

    previous_loss, initial_loss = float("inf"), None
    for step in pbar:
        if len(optimized_ks) != 0:
            normalized_continuous_predicate_parameters = (
                continuous_predicate_representation_parameters
                / continuous_predicate_representation_parameters.norm(
                    dim=1, keepdim=True
                )
            )

            continuous_denotation = torch.matmul(
                embeddings, normalized_continuous_predicate_parameters.T
            )
            assert continuous_denotation.shape == (N, K)

            denotation = (
                orig_denotation
                + continuous_denotation * continuous_denotation_added_mask
            )
        else:
            denotation = fixed_denotation
        logits = torch.matmul(w, denotation[text_idx_by_time].T)
        perplexity_loss = -lsm(logits).diag().mean()

        discontinuity_loss = torch.mean((w[1:] - w[:-1]) ** 2)

        loss = perplexity_loss + discontinuity_penalty * discontinuity_loss
        loss.backward()
        optim.step()

        optim.zero_grad()

        current_loss = loss.item()
        pbar.set_description(f"Loss: {loss.item():.4f}")
        if initial_loss is None:
            initial_loss = current_loss

        if current_loss > previous_loss - 1e-5:
            break
        previous_loss = current_loss

    logger.debug(f"initial loss: {initial_loss}, final loss: {current_loss}")
    denotation = denotation.detach().cpu().numpy()
    return {
        "w": w.detach().cpu().numpy(),
        "full_phi_denotation": denotation,
        "loss": current_loss,
        "perplexity_loss": perplexity_loss.item(),
        "discontinuity_loss": discontinuity_loss.item(),
    }


class TimeSeriesModel(AbstractModel):

    def __init__(
        self,
        texts_by_time,
        K,
        goal,
        discontinuity_penalty = 1.0,
        embedder = None,
        validator = None,
        proposer = None,
        temperature=TEMPERATURE,
        num_samples_to_compute_performance=NUM_SAMPLES_TO_COMPUTE_PERFORMANCE,
        num_iterations=NUM_ITERATIONS,
        num_candidate_predicates_to_explore=NUM_CANDIDATE_PREDICATES_TO_EXPLORE,
        reference_phi_denotation=None,
        reference_phi_predicate_strings=None,
        random_update=False,
        dummy=None,
    ):

        # problem statement, these are cluster models' inputs
        self.texts_by_time = texts_by_time

        texts, reference_phi_denotation, _ = AbstractModel.reorder_texts(
            self.texts_by_time, reference_phi_denotation=reference_phi_denotation
        )

        self.texts = texts
        self.text2idx = {t: i for i, t in enumerate(self.texts)}
        self.text_idx_by_time = [self.text2idx[t] for t in self.texts_by_time]
        self.K = K
        self.discontinuity_penalty = discontinuity_penalty
        self.goal = goal

        # cluster model specific arguments
        self.commit = False
        self.absolute_correlation = True
        self.dummy = dummy

        super().__init__(
            embedder=embedder,
            validator=validator,
            proposer=proposer,
            temperature=temperature,
            num_samples_to_compute_performance=num_samples_to_compute_performance,
            num_iterations=num_iterations,
            num_candidate_predicates_to_explore=num_candidate_predicates_to_explore,
            reference_phi_denotation=reference_phi_denotation,
            reference_phi_predicate_strings=reference_phi_predicate_strings,
            random_update=random_update,
            dummy=dummy,
        )

    def optimize_w(self, phi_denotation, w_init=None):
        return optimize_ts(
            self.text_idx_by_time,
            self.embeddings,
            self.discontinuity_penalty,
            phi_denotation,
            w_init=w_init,
            optimize_w=True,
            optimized_ks=[],
            seed=0,
        )

    def optimize_tilde_phi(self, w, full_phi_denotation, optimized_ks):
        full_phi_denotation = optimize_ts(
            self.text_idx_by_time,
            self.embeddings,
            self.discontinuity_penalty,
            full_phi_denotation,
            w,
            False,
            optimized_ks,
            0,
        )["full_phi_denotation"]
        return full_phi_denotation

    def compute_fitness_from_phi_denotation(self, phi_denotation):
        return optimize_ts(
            self.text_idx_by_time,
            self.embeddings,
            self.discontinuity_penalty,
            phi_denotation,
            None,
            True,
            [],
            0,
        )["loss"]