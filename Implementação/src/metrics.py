"""Metricas para classificacao multi-label de emocoes.

Com 28 classes desbalanceadas, a accuracy isolada nao e informativa. As
metricas principais sao baseadas em F1, tratando as classes de formas
distintas:

  - Macro F1   (PRINCIPAL):  media simples do F1 por classe; nao pondera
                             pelo tamanho -> justo sob desbalanceamento.
  - Micro F1:                agrega TP/FP/FN globais; padrao no paper original
                             do GoEmotions, dominado pelas classes frequentes.
  - Weighted F1 (COMPLEMENTAR): media do F1 ponderada pelo suporte de cada classe.
  - Subset accuracy (REFERENCIA): fracao de exemplos com TODOS os rotulos
                             corretos (exact match) - muito estrita em multi-label.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid numericamente estavel."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def logits_to_preds(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Converte logits em predicoes binarias (multi-hot) aplicando sigmoid + limiar."""
    probs = sigmoid(logits)
    return (probs >= threshold).astype(int)


def compute_multilabel_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcula o conjunto de metricas multi-label a partir dos logits."""
    preds = logits_to_preds(logits, threshold)
    labels = labels.astype(int)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "subset_accuracy": accuracy_score(labels, preds),
    }


def build_compute_metrics(threshold: float = 0.5):
    """Cria a funcao compute_metrics usada pelo Trainer do HuggingFace.

    O Trainer chama a funcao com um objeto contendo `predictions` (logits) e
    `label_ids` (rotulos verdadeiros).
    """

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(logits, tuple):  # alguns modelos retornam tupla
            logits = logits[0]
        return compute_multilabel_metrics(np.asarray(logits), np.asarray(labels), threshold)

    return compute_metrics
