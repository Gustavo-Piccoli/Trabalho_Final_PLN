"""Carregamento e pre-processamento do GoEmotions para classificacao multi-label.

Fornece:
  - load_goemotions(): baixa o dataset (HuggingFace), valida a taxonomia,
    constroi vetores de rotulo multi-hot (28 dimensoes, float) e tokeniza.
  - MultiLabelCollator: collator que faz padding dinamico dos textos e
    empilha os rotulos como tensores float (exigido pela BCEWithLogitsLoss).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch
from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from src.labels import GOEMOTIONS_LABELS, NUM_LABELS

# Nome canonico no Hub e a config "simplified" (rotulos ja filtrados por
# concordancia entre anotadores; formato multi-label).
_HF_DATASET = "google-research-datasets/go_emotions"
_HF_CONFIG = "simplified"


def load_goemotions(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 64,
    smoke: bool = False,
    cache_dir: str = "data/cache",
) -> DatasetDict:
    """Carrega e tokeniza o GoEmotions.

    Args:
        tokenizer: tokenizador do modelo (BERT ou RoBERTa).
        max_length: comprimento maximo em tokens (comentarios sao curtos).
        smoke: se True, usa um subconjunto minusculo para teste rapido.
        cache_dir: diretorio de cache do HuggingFace datasets.

    Returns:
        DatasetDict com splits 'train'/'validation'/'test' contendo
        input_ids, attention_mask e labels (vetor multi-hot float de 28 dims).
    """
    raw: DatasetDict = load_dataset(_HF_DATASET, _HF_CONFIG, cache_dir=cache_dir)

    # Valida a taxonomia carregada contra a ordem canonica de labels.py.
    dataset_labels = raw["train"].features["labels"].feature.names
    if list(dataset_labels) != GOEMOTIONS_LABELS:
        warnings.warn(
            "A ordem de rotulos do dataset difere de labels.py. "
            "Usando a ordem do dataset para os indices.",
            stacklevel=2,
        )

    if smoke:
        raw["train"] = raw["train"].select(range(min(200, len(raw["train"]))))
        raw["validation"] = raw["validation"].select(range(min(100, len(raw["validation"]))))
        raw["test"] = raw["test"].select(range(min(100, len(raw["test"]))))

    def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
        enc = tokenizer(batch["text"], truncation=True, max_length=max_length)
        # Converte a lista de indices em vetor multi-hot float.
        multi_hot = []
        for label_ids in batch["labels"]:
            vec = [0.0] * NUM_LABELS
            for idx in label_ids:
                vec[idx] = 1.0
            multi_hot.append(vec)
        enc["labels"] = multi_hot
        return enc

    tokenized = raw.map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizando e construindo rotulos multi-hot",
    )
    return tokenized


@dataclass
class MultiLabelCollator:
    """Faz padding dinamico dos textos e empilha rotulos como float.

    O padding dinamico (ao tamanho do maior exemplo do batch) e mais
    eficiente que padding fixo. Os rotulos multi-hot ja tem tamanho fixo
    (28) e sao convertidos para float, dtype exigido pela BCEWithLogitsLoss.
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float)
        return batch
