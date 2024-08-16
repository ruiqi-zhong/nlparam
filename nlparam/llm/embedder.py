from transformers import T5EncoderModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from typing import List
import torch
import numpy as np
from tqdm import trange
import os
import pickle as pkl

device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_EMBEDDING_DIM = 50
CACHE_DIR = "embedding_cache/"


class Embedder:
    def __init__(self, model_name, dimension=None, sentence_transformer=False, disable_cache=False):
        self.is_dummy = model_name == "dummy"
        self.is_one_hot = model_name == "one_hot"

        self.model_name = model_name
        self.name = f"embedder_{model_name.replace('/', '')}"
        self.dimension = dimension
        self.cache_path = os.path.join(CACHE_DIR, self.name)
        self.rand_project = False
        self.sentence_transformer = sentence_transformer
        self.disable_cache = disable_cache

        self.text2embedding = {}
        if self.is_one_hot:
            self.text2idx = None
        elif self.is_dummy:
            self.dimension = DEFAULT_EMBEDDING_DIM
            self.vocab_size = 500
            self.embedding_matrix = np.random.randn(self.vocab_size, self.dimension)
            self.projector = np.eye(self.dimension)
        else:
            if not sentence_transformer:
                self.model = T5EncoderModel.from_pretrained(model_name).to(device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model_dim = AutoConfig.from_pretrained(model_name).d_model
            else:
                self.embedder = SentenceTransformer(model_name).to(device)
                self.model_dim = 768

            if dimension is None:
                self.dimension = self.model_dim
                self.projector = np.eye(self.model_dim)
                if os.path.exists(self.cache_path) and not disable_cache:
                    with open(self.cache_path, "rb") as f:
                        self.text2embedding = pkl.load(f)
            else:
                self.projector = np.random.randn(self.model_dim, self.dimension)
                projector_norm = np.linalg.norm(self.projector, axis=0)
                self.projector = self.projector / projector_norm
                self.cache_path = os.path.join(CACHE_DIR, self.name)
                self.rand_project = True

    def set_one_hot_texts(self, texts: List[str]):
        assert self.is_one_hot
        self.text2idx = {text: i for i, text in enumerate(texts)}
        self.dimension = len(texts)

    def embed_batch(self, texts: List[str]):
        if not self.sentence_transformer:
            encoded_input = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                output = self.model(**encoded_input)
            # return the average embedding of the last layer after removing the padding
            result = []
            for i in range(len(texts)):
                v = (
                    output.last_hidden_state[i, encoded_input["attention_mask"][i] == 1]
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                ) @ self.projector
                v = v / np.linalg.norm(v)
                result.append(v)
            return np.array(result)
        else:
            raw_embeddings = self.embedder.encode(texts)
            return raw_embeddings @ self.projector

    def embed(self, texts: List[str], bsize=16, use_cache: bool = True) -> np.ndarray:
        if self.is_dummy:
            result_embedding = []
            for t in texts:
                toks = t.split()
                v = np.sum(
                    [
                        self.embedding_matrix[hash(tok) % self.vocab_size]
                        for tok in toks
                    ],
                    axis=0,
                )
                v = v / np.linalg.norm(v)
                result_embedding.append(v)
            result_embedding = np.array(result_embedding)

            return result_embedding @ self.projector
        elif self.is_one_hot:
            self.set_one_hot_texts(texts)
            assert self.text2idx is not None

            result_embedding = np.zeros((len(texts), len(self.text2idx)))
            text_idxes = [self.text2idx[t] for t in texts]
            result_embedding[np.arange(len(texts)), text_idxes] = 1.0

            return result_embedding

        else:
            dedup_texts = list(
                set([t for t in texts if t not in self.text2embedding or not use_cache])
            )

            if len(dedup_texts) != 0:
                result = []
                pbar = trange(0, len(dedup_texts), bsize, desc="Embedding")
                for i in pbar:
                    result.append(self.embed_batch(dedup_texts[i : i + bsize]))
                result = np.concatenate(result)

                for text, embedding in zip(dedup_texts, result):
                    self.text2embedding[text] = embedding

                if self.cache_path is not None and not self.rand_project and not self.disable_cache:
                    with open(self.cache_path, "wb") as f:
                        pkl.dump(self.text2embedding, f)

            result_embeddings = [self.text2embedding[t] for t in texts]
            result = np.array(result_embeddings)
            return result
