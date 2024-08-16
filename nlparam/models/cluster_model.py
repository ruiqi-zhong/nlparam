import numpy as np
import torch
from nlparam.models.abstract_model  import (
    AbstractModel,
    NUM_ITERATIONS,
    NUM_CANDIDATE_PREDICATES_TO_EXPLORE,
    NUM_SAMPLES_TO_COMPUTE_PERFORMANCE,
    TEMPERATURE,
)
import json
from nlparam import Embedder, get_validator_by_name, Proposer, DEFAULT_EMBEDDER_NAME, DEFAULT_VALIDATOR_NAME, DEFAULT_PROPOSER_NAME

device = "cuda" if torch.cuda.is_available() else "cpu"


def np_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_centers_from_pos_neg(
    embeddings, pos_idxes_by_cluster, smoothing_prob, temperature
):
    pos_idxes_by_cluster = (
        torch.tensor(pos_idxes_by_cluster, dtype=torch.float32).to(device) == 1
    )
    assert len(embeddings) == len(pos_idxes_by_cluster[0])
    for pos_idxes in pos_idxes_by_cluster:
        if torch.sum(pos_idxes) == 0:
            pos_idxes[np.random.choice(len(pos_idxes))] = True

    centroids_param = torch.nn.Parameter(
        torch.tensor(
            [
                np.mean(embeddings[pos_idxes.detach().cpu().numpy()], axis=0)
                for pos_idxes in pos_idxes_by_cluster
            ],
            dtype=torch.float32,
        ).to(device)
    )
    embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam([centroids_param], lr=0.1)

    previous_loss = 1e10
    probability_lower_bound = smoothing_prob / len(embeddings)

    for _ in range(100):
        dot_products = torch.matmul(
            centroids_param / torch.norm(centroids_param, dim=1)[:, None],
            embeddings.transpose(1, 0),
        )
        probs = torch.softmax(dot_products / temperature, dim=1)
        probs = torch.clamp(probs, probability_lower_bound, 1.0)

        datapoint_probs = pos_idxes_by_cluster * probs
        # extract the non-zero elements
        datapoint_probs = datapoint_probs[datapoint_probs != 0]

        loss = -torch.mean(torch.log(datapoint_probs))
        loss.backward()

        if loss > previous_loss - 1e-5:
            break
        previous_loss = loss.item()
        optimizer.step()
        optimizer.zero_grad()

    centroids_param = centroids_param / torch.norm(centroids_param, dim=1)[:, None]

    return {
        "cluster_centers": centroids_param.detach().cpu().numpy(),
        "loss": loss.item(),
    }


class ClusteringModel(AbstractModel):

    def __init__(
        self,
        texts,
        K,
        goal,
        smoothing_prob = 1.0,
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
        dummy=False,
    ):

        # problem statement, these are cluster models' inputs

        texts, reference_phi_denotation, _ = AbstractModel.reorder_texts(
            texts, reference_phi_denotation=reference_phi_denotation
        )
        self.texts = texts
        self.K = K
        self.smoothing_prob = smoothing_prob
        self.goal = goal

        # cluster model specific arguments
        self.commit = True
        self.absolute_correlation = False
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
        logits = phi_denotation
        logits = logits / self.temperature
        probs = np_softmax(logits)
        cluster_idxes = np.argmax(probs, axis=1)
        return {
            "w": cluster_idxes,
            "logits": logits,
            "probs": probs,
        }

    def optimize_tilde_phi(self, w, full_phi_denotation, optimized_ks):
        assert type(optimized_ks) == list
        pos_idxes_by_cluster = np.array([w == k for k in optimized_ks])
        get_center_result = get_centers_from_pos_neg(
            self.embeddings,
            pos_idxes_by_cluster,
            self.smoothing_prob,
            temperature=self.temperature,
        )
        centroids, loss = (
            get_center_result["cluster_centers"],
            get_center_result["loss"],
        )
        tilde_phi_denotation = np.matmul(centroids, self.embeddings.transpose(1, 0)).T
        returned_full_phi_denotation = np.array(full_phi_denotation)
        for new_idx, k in enumerate(optimized_ks):
            returned_full_phi_denotation[:, k] = tilde_phi_denotation[:, new_idx]
        return returned_full_phi_denotation

    def compute_fitness_from_phi_denotation(self, phi_denotation):

        count_by_column = np.sum(phi_denotation, axis=0) + 1e-6
        phi_denotation = np.array(phi_denotation)
        probs = phi_denotation / count_by_column
        probs = np.maximum(probs, self.smoothing_prob / self.n_texts)
        probs = np.max(probs, axis=1)
        loss = np.mean(-np.log(probs))
        return loss
