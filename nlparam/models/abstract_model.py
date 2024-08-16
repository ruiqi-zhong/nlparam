from scipy.stats import pearsonr
from nlparam.llm.validate import validate_descriptions
from nlparam.eval_utils import compare_text_description_matching, get_similarity_scores, compare_tdm
from nlparam import logger
from nlparam import get_validator_by_name, Validator, Embedder, Proposer, DEFAULT_EMBEDDER_NAME, DEFAULT_VALIDATOR_NAME, DEFAULT_PROPOSER_NAME
from copy import deepcopy
import random
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelOutput:
    predicate2text2matching: dict
    aligned_predicates: list = None
    surface_similarity_score: float = None
    f1_similarity_score: float = None

NUM_SAMPLES_TO_COMPUTE_PERFORMANCE = 256
TEMPERATURE = 0.1
NUM_ITERATIONS = 10
NUM_CANDIDATE_PREDICATES_TO_EXPLORE = 5

def get_correlation(
    text_description_matching,
    target,
):
    text_description_matching_perturbed = text_description_matching + np.random.normal(
        0, 1e-6, size=text_description_matching.shape
    )
    target_perturbed = target + np.random.normal(0, 1e-6, size=target.shape)
    return pearsonr(target_perturbed, text_description_matching_perturbed)[0]


def kmeans_pp_init(embeddings, K):
    centroids = [embeddings[np.random.choice(len(embeddings))]]
    for _ in range(K - 1):
        dist_squared_for_different_centroids = []
        for c in centroids:
            dist_squared = np.linalg.norm(c - embeddings, axis=1)
            dist_squared_for_different_centroids.append(dist_squared)

        dist_squared = np.stack(dist_squared_for_different_centroids, axis=0)
        d_x = np.min(dist_squared, axis=0)
        prob_x = d_x / np.sum(d_x)
        centroids.append(embeddings[np.random.choice(range(len(embeddings)), p=prob_x)])
    return np.array(centroids)


class PredicateHandler:
    def __init__(
        self, validator, texts, subset_idxes, absolute_correlation, random_corr=False
    ):
        self.validator = validator
        self.all_predicate_strings = []
        self.all_predicate_denotation = np.zeros((len(subset_idxes), 0))
        self.texts = texts
        self.subset_idxes = subset_idxes
        self.N = len(texts)
        self.subset_texts = [texts[i] for i in subset_idxes]
        self.absolute_correlation = absolute_correlation
        self.random_corr = random_corr

    def add_predicates(self, new_predicate_strings):
        deduped_new_predicate_strings = []
        for predicate_string in new_predicate_strings:
            if predicate_string not in self.all_predicate_strings:
                deduped_new_predicate_strings.append(predicate_string)
                self.all_predicate_strings.append(predicate_string)

        if len(deduped_new_predicate_strings) == 0:
            return
        logger.debug(f"Adding {len(deduped_new_predicate_strings)} predicates")
        new_predicate_denotation = validate_descriptions(
            deduped_new_predicate_strings,
            self.subset_texts,
            self.validator,
            progress_bar=True,
        )
        self.all_predicate_denotation = np.concatenate(
            [self.all_predicate_denotation, new_predicate_denotation], axis=1
        )

    def get_predicates_by_correlation(self, logits):
        assert logits.shape[0] == self.N

        predicate_idx2correlation = {}
        for i in range(len(self.all_predicate_strings)):
            predicate_idx2correlation[i] = get_correlation(
                self.all_predicate_denotation[:, i],
                logits[self.subset_idxes],
            )
            if self.absolute_correlation:
                predicate_idx2correlation[i] = np.abs(predicate_idx2correlation[i])

        if not self.random_corr:
            predicate_idx_sorted_by_correlation = sorted(
                predicate_idx2correlation.keys(),
                key=lambda x: predicate_idx2correlation[x],
                reverse=True,
            )
            return [
                self.all_predicate_strings[i]
                for i in predicate_idx_sorted_by_correlation
            ]
        else:
            idx_shuffled = list(predicate_idx2correlation.keys())
            random.shuffle(idx_shuffled)
            return [self.all_predicate_strings[i] for i in idx_shuffled]


class AbstractModel:

    def __init__(
        self,
        embedder,
        validator,
        proposer,
        temperature,
        num_samples_to_compute_performance,
        num_iterations,
        num_candidate_predicates_to_explore,
        reference_phi_denotation=None,
        reference_phi_predicate_strings=None,
        random_update=False,
        dummy=False,
    ):
        self.embedder = embedder
        self.validator = validator
        self.proposer = proposer

        if not dummy:
            if self.validator is None:
                self.validator = get_validator_by_name(DEFAULT_VALIDATOR_NAME)
            if self.embedder is None:
                self.embedder = Embedder(DEFAULT_EMBEDDER_NAME)
            if self.proposer is None:
                self.proposer = Proposer(DEFAULT_PROPOSER_NAME)
        else:
            self.validator = get_validator_by_name("dummy")
            self.embedder = Embedder("dummy")
            self.proposer = Proposer("dummy")

        self.temperature = temperature
        self.num_samples_to_compute_performance = num_samples_to_compute_performance
        self.num_iterations = num_iterations
        self.num_candidate_predicates_to_explore = num_candidate_predicates_to_explore
        self.reference_phi_denotation = reference_phi_denotation
        self.reference_phi_predicate_strings = reference_phi_predicate_strings
        self.random_update = random_update
        if self.random_update:
            print(
                "Random update is enabled, the model will randomly update the predicates; this is for experimental purposes only"
            )

        self.post_init()

    @staticmethod
    def reorder_texts(texts, reference_phi_denotation=None, labels=None):
        sorted_idx_based_on_text = np.argsort(texts)
        texts = np.array(texts)[sorted_idx_based_on_text].tolist()

        if labels is not None:
            labels = np.array(labels)[sorted_idx_based_on_text].tolist()

        if reference_phi_denotation is not None:
            reference_phi_denotation = np.array(reference_phi_denotation)[
                sorted_idx_based_on_text
            ]

        return texts, reference_phi_denotation, labels

    def post_init(self):
        # compute other statistics that might be used
        assert self.texts == sorted(self.texts), "Texts should be sorted"
        assert len(self.texts) == len(set(self.texts)), "Texts should be unique"
        self.n_texts = len(self.texts)
        self.subsample_idxes = np.random.choice(
            self.n_texts,
            min(self.n_texts, self.num_samples_to_compute_performance),
            replace=False,
        )
        self.compute_embeddings()
        self.predicate_handler = PredicateHandler(
            self.validator,
            self.texts,
            self.subsample_idxes,
            self.absolute_correlation,
            random_corr=self.random_update,
        )

        self.init_predicate_strings = ["N/A" for _ in range(self.K)]
        self.phi_predicate_strings = ["N/A" for _ in range(self.K)]
        self.phi_predicate_denotation = np.zeros((self.n_texts, self.K))

        self.optimization_trajectory = []

    def compute_embeddings(self):
        embeddings = self.embedder.embed(self.texts)
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]

    def find_idxes_by_uselessness(self, phi_denotation):

        k2has_identical_ks = {}
        for k in range(self.K):
            for k_prime in range(self.K):
                if k != k_prime and np.all(
                    phi_denotation[:, k] == phi_denotation[:, k_prime]
                ):
                    k2has_identical_ks[k] = True
                    break

        current_loss = self.compute_fitness_from_phi_denotation(phi_denotation)
        increase_in_loss = []
        for k in range(self.K):
            logger.debug(f"Finding idxes by uselessness, k={k}")
            ablated_phi_denotation = np.array(phi_denotation)
            ablated_phi_denotation[:, k] = 0
            new_loss = self.compute_fitness_from_phi_denotation(ablated_phi_denotation)
            increase_in_loss.append(new_loss - current_loss)
        return sorted(
            range(self.K),
            key=lambda x: (
                float("-inf") if k2has_identical_ks.get(x) else increase_in_loss[x]
            ),
        )

    def initialize_tilde_phi(self):
        tilde_phi = kmeans_pp_init(self.embeddings, self.K)
        tilde_phi_denotation = np.matmul(tilde_phi, self.embeddings.transpose(1, 0)).T
        return self.alternatively_optimize_tilde_phi_and_w(
            phi_denotation=tilde_phi_denotation, optimized_ks=list(range(self.K))
        )

    def alternatively_optimize_tilde_phi_and_w(
        self, phi_denotation, optimized_ks, num_iterations=10
    ):
        logger.debug(f"Optimizing continous predicates with idxes {optimized_ks} now")
        logger.debug(f"Optimizing w now")
        w = self.optimize_w(phi_denotation)["w"]

        pbar = range(num_iterations)
        for step in pbar:
            logger.debug(f"Optimizing tilde_phi, step {step}")
            tilde_phi_denotation = self.optimize_tilde_phi(
                w, phi_denotation, optimized_ks
            )
            logger.debug(f"Optimizing w, step {step}")
            w = self.optimize_w(tilde_phi_denotation, w_init=w)["w"]
        return {
            "tilde_phi_denotation": tilde_phi_denotation,
            "w": w,
        }

    def eval_phi_predicate_strings(self, predicate_strings):

        return_dict = {
            "predicate_strings": predicate_strings,
        }

        predicted_predicate_denotation_before_commitment = validate_descriptions(
            predicate_strings, self.texts, self.validator, progress_bar=True
        )
        return_dict["before_commitment_denotation"] = (
            predicted_predicate_denotation_before_commitment
        )
        return_dict["before_commitment_predicate2text2matching"] = {
            predicate_strings[i]: {
                self.texts[j]: int(predicted_predicate_denotation_before_commitment[j, i])
                for j in range(self.n_texts)
            }
            for i in range(self.K)
        }
        return_dict["predicate2text2matching"] = return_dict["before_commitment_predicate2text2matching"]

        if self.reference_phi_denotation is not None:
            before_commitment_performance = compare_text_description_matching(
                self.reference_phi_denotation,
                predicted_predicate_denotation_before_commitment,
                hungarian_matching=False
            )
            return_dict["before_commitment"] = before_commitment_performance
            logger.debug(
                f"F1 score before commitment: {before_commitment_performance['main_metric']}",
            )

        if self.commit:
            predicted_predicate_denotation_after_commitment = np.array(
                list(
                    self.validator.get_multi_text_description_matching(
                        descriptions=predicate_strings,
                        texts=self.texts,
                        verbose=True,
                    )
                )
            )
            return_dict["after_commitment_denotation"] = (
                predicted_predicate_denotation_after_commitment
            )
            return_dict["after_commitment_predicate2text2matching"] = {
                predicate_strings[i]: {
                    self.texts[j]: int(predicted_predicate_denotation_after_commitment[j, i])
                    for j in range(self.n_texts)
                }
                for i in range(self.K)
            }
            return_dict["predicate2text2matching"] = return_dict["after_commitment_predicate2text2matching"]

            if self.reference_phi_denotation is not None:
                after_commitment_performance = compare_text_description_matching(
                    self.reference_phi_denotation,
                    predicted_predicate_denotation_after_commitment,
                    hungarian_matching=True
                )
                logger.debug(
                    f"F1 score after commitment: {after_commitment_performance['main_metric']}",
                )
                return_dict["after_commitment"] = after_commitment_performance

        if self.reference_phi_predicate_strings is not None:
            predicted_phi_denotation = (
                predicted_predicate_denotation_after_commitment
                if self.commit
                else predicted_predicate_denotation_before_commitment
            )
            matching_idxes = compare_tdm(
                self.reference_phi_denotation, predicted_phi_denotation
            )["matching_idxes"]
            aligned_predicted_predicate_strings = [
                predicate_strings[i] for i in matching_idxes
            ]
            return_dict["aligned_predicted_predicate_strings"] = (
                aligned_predicted_predicate_strings
            )

            similarity_scores = get_similarity_scores(
                self.reference_phi_predicate_strings,
                aligned_predicted_predicate_strings,
            )
            return_dict["surface_similarity_scores"] = similarity_scores
            return_dict["mean_surface_similarity_score"] = float(np.mean(similarity_scores))

        validator_inference_count = self.validator.count
        logger.debug(f"having done inference {validator_inference_count} times")
        return_dict["validator_inference_count"] = validator_inference_count
        return_dict["loss"] = self.compute_fitness_from_phi_denotation(
            predicted_predicate_denotation_before_commitment
        )
        return return_dict

    def log_optimization_trajectory(self, tag):
        log_dict = {
            "tag": tag,
            "predicate_strings": list(self.phi_predicate_strings),
            "eval_phi_predicate_strings": self.eval_phi_predicate_strings(
                self.phi_predicate_strings
            ),
        }
        self.optimization_trajectory.append(log_dict)
        self.phi_predicate_denotation = log_dict["eval_phi_predicate_strings"][
            "before_commitment_denotation"
        ]

    def full_optimization_loop(self):
        init_tilde_phi = self.initialize_tilde_phi()
        tilde_phi_denotation = init_tilde_phi["tilde_phi_denotation"]

        for k in range(self.K):
            continuous_denotation = tilde_phi_denotation[:, k]
            proposer_response = self.proposer.propose_descriptions(
                texts=self.texts, target=continuous_denotation, goal=self.goal
            )
            self.predicate_handler.add_predicates(proposer_response.descriptions)

        for k in range(self.K):
            best_predicate = self.predicate_handler.get_predicates_by_correlation(
                tilde_phi_denotation[:, k]
            )[0]
            self.init_predicate_strings[k] = best_predicate
            self.phi_predicate_strings[k] = best_predicate

        self.log_optimization_trajectory("init")
        for iteration in range(self.num_iterations):
            any_update_from_this_iteration = False
            current_loss = self.compute_fitness_from_phi_denotation(
                self.phi_predicate_denotation
            )
            logger.debug(
                f"Iteration {iteration}, current loss {current_loss}, current predicate strings {self.phi_predicate_strings}"
            )
            predicate_idxes_to_be_updated = self.find_idxes_by_uselessness(
                self.phi_predicate_denotation
            )
            logger.debug(
                f"Predicate idxes sorted by uselessness: {predicate_idxes_to_be_updated}"
            )

            for k in predicate_idxes_to_be_updated:
                # optimize the predicate string for k
                new_continuous_denotation = self.alternatively_optimize_tilde_phi_and_w(
                    self.phi_predicate_denotation, [k]
                )["tilde_phi_denotation"][:, k]

                new_candidate_descriptions = self.proposer.propose_descriptions(
                    texts=self.texts, target=new_continuous_denotation, goal=self.goal
                ).descriptions
                self.predicate_handler.add_predicates(new_candidate_descriptions)
                top_predicates = self.predicate_handler.get_predicates_by_correlation(
                    new_continuous_denotation
                )[: self.num_candidate_predicates_to_explore]
                logger.debug(f"Top predicates for predicate {k}: {top_predicates}")

                best_new_loss, best_new_predicate_idx = current_loss, None

                for new_predicate_idx in range(len(top_predicates)):
                    new_predicate_strings = deepcopy(self.phi_predicate_strings)
                    new_predicate_strings[k] = top_predicates[new_predicate_idx]
                    new_predicate_denotation = validate_descriptions(
                        new_predicate_strings,
                        self.texts,
                        self.validator,
                        progress_bar=True,
                    )
                    new_loss = self.compute_fitness_from_phi_denotation(
                        new_predicate_denotation
                    )
                    logger.debug(f"new_loss: {new_loss}")
                    if new_loss < best_new_loss:
                        best_new_loss, best_new_predicate_idx = (
                            new_loss,
                            new_predicate_idx,
                        )

                if best_new_predicate_idx is not None:
                    self.phi_predicate_strings[k] = top_predicates[
                        best_new_predicate_idx
                    ]
                    any_update_from_this_iteration = True

                if any_update_from_this_iteration:
                    self.log_optimization_trajectory(f"iteration_{iteration}")
                    break

            if not any_update_from_this_iteration:
                break

        return self.optimization_trajectory
    
    def fit(self) -> ModelOutput:
        result = self.full_optimization_loop()
        predicate2text2matching = result[-1]["eval_phi_predicate_strings"]["predicate2text2matching"]
        aligned_predicates, surface_similarity_score, f1_similarity_score = None, None, None

        if self.reference_phi_predicate_strings is not None:
            commitment_key = "after_commitment" if self.commit else "before_commitment"
            aligned_predicates = result[-1]["eval_phi_predicate_strings"]["aligned_predicted_predicate_strings"]
            surface_similarity_score = result[-1]["eval_phi_predicate_strings"]["mean_surface_similarity_score"]
            f1_similarity_score = result[-1]["eval_phi_predicate_strings"][commitment_key]["main_metric"]
        
        return ModelOutput(
            predicate2text2matching=predicate2text2matching,
            aligned_predicates=aligned_predicates,
            surface_similarity_score=surface_similarity_score,
            f1_similarity_score=f1_similarity_score
        )

    
    def optimize_w(self, phi_denotation, w_init=None):
        """
        Optimize w given phi_denotation
        
        Args:
            phi_denotation: np.array of shape (n_texts, K), where the kth column is the (continuous) denotation of the kth predicate
            w_init: the initial value of w
            
        Returns:
            dict: {
                "w": the optimized w,
                ...
            }
        """
        raise NotImplementedError


    def optimize_tilde_phi(self, w, full_phi_denotation, optimized_ks):
        """
        Optimize tilde_phi given w
        
        Args:
            w: the optimized w
            full_phi_denotation: np.array of shape (n_texts, K), where the kth column is the (continuous) denotation of the kth predicate
            optimized_ks: list of indices of predicates to be optimized
        
        Returns:
            np.array: (n_texts, K), where the kth column is the (continuous) denotation of the kth predicate
        """
        raise NotImplementedError
    
    
    def compute_fitness_from_phi_denotation(self, phi_denotation):
        """
        Compute the fitness from phi_denotation
        
        Args:
            phi_denotation: np.array of shape (n_texts, K), where the kth column is the (continuous) denotation of the kth predicate
        
        Returns:
            float: the fitness
        """
        raise NotImplementedError