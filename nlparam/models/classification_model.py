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
from sklearn.linear_model import LogisticRegression
from tqdm import trange

device = "cuda" if torch.cuda.is_available() else "cpu"
lsm = torch.nn.LogSoftmax(dim=1)


def lr(X, Y):
    clf = LogisticRegression(
        random_state=0,
        C=1e5,
        solver="lbfgs",
        max_iter=500,
        multi_class="multinomial",
        fit_intercept=False,
    )
    clf.fit(X, Y)
    w = clf.coef_
    probs = clf.predict_proba(X)
    loss = -np.mean(np.log(probs[np.arange(len(probs)), Y]))

    if w.shape[0] == 1:
        w = np.concatenate([np.zeros((1, w.shape[1])), w], axis=0)

    return {
        "w": w.T,
        "loss": loss,
    }


class ClassificationModel(AbstractModel):

    def __init__(
        self,
        texts,
        labels,
        K,
        goal,
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
        texts, reference_phi_denotation, labels = AbstractModel.reorder_texts(
            texts, reference_phi_denotation=reference_phi_denotation, labels=labels
        )

        # problem statement, these are cluster models' inputs
        self.texts = texts
        self.K = K
        self.goal = goal
        self.labels = labels

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

    # the dimension of w is K x C
    def optimize_w(self, phi_denotation, w_init=None):
        return lr(phi_denotation, self.labels)

    def optimize_tilde_phi(self, w, full_phi_denotation, optimized_ks):
        continuous_denotation_added_mask = np.zeros(
            full_phi_denotation.shape[1], dtype=np.float32
        )
        continuous_denotation_added_mask[optimized_ks] = 1.0
        continuous_denotation_added_mask = torch.tensor(
            continuous_denotation_added_mask, device=device
        )

        N, K = full_phi_denotation.shape
        N, D = self.embeddings.shape

        continuous_predicate_representation_init = kmeans_pp_init(self.embeddings, K)
        continuous_predicate_representation_parameters = torch.nn.Parameter(
            torch.tensor(continuous_predicate_representation_init).to(device)
        )
        continuous_predicate_representation_parameters.requires_grad = True
        assert continuous_predicate_representation_parameters.shape == (K, D)

        orig_denotation = torch.tensor(full_phi_denotation, device=device)
        orig_denotation[:, list(optimized_ks)] = 0.0

        w = torch.tensor(w, device=device)

        optimizer = torch.optim.Adam(
            [continuous_predicate_representation_parameters], lr=1e-2
        )

        previous_loss, initial_loss = float("inf"), None
        pbar = trange(1000)

        embeddings_torch = torch.tensor(self.embeddings, device=device)

        for step in pbar:
            normalized_continuous_predicate_parameters = (
                continuous_predicate_representation_parameters
                / continuous_predicate_representation_parameters.norm(
                    dim=1, keepdim=True
                )
            )

            continuous_denotation = torch.matmul(
                embeddings_torch, normalized_continuous_predicate_parameters.T
            )
            assert continuous_denotation.shape == (N, K)
            denotation = (
                orig_denotation
                + continuous_denotation * continuous_denotation_added_mask
            )
            logits = torch.matmul(denotation, w)  # N X C
            probs = lsm(logits)
            loss = -probs[torch.arange(len(probs)), self.labels].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if initial_loss is None:
                initial_loss = current_loss

            if previous_loss - current_loss < 1e-4:
                break

            previous_loss = current_loss

            pbar.set_description(f"Loss: {loss.item():.4f}")

        denotation = denotation.detach().cpu().numpy()
        return denotation

    def compute_fitness_from_phi_denotation(self, phi_denotation):
        return lr(phi_denotation, self.labels)["loss"]