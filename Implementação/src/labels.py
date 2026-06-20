"""Taxonomia de emocoes do GoEmotions.

O dataset possui 28 classes: 27 emocoes + 'neutral'. A ordem abaixo segue
o indice oficial usado pelo dataset no HuggingFace Hub. Ainda assim, o
modulo `dataset.py` valida essa lista contra os nomes reais carregados em
tempo de execucao, evitando divergencias silenciosas.

Tambem definimos o agrupamento por sentimento proposto por Demszky et al.
(2020), util para analises agregadas (ex.: matriz de confusao em nivel de
sentimento) na etapa de avaliacao.
"""

from __future__ import annotations

# Ordem canonica das 28 classes (indice 0..27).
GOEMOTIONS_LABELS: list[str] = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

NUM_LABELS: int = len(GOEMOTIONS_LABELS)  # 28

# Mapeamentos auxiliares.
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(GOEMOTIONS_LABELS)}
ID2LABEL: dict[int, str] = {i: name for i, name in enumerate(GOEMOTIONS_LABELS)}

# Agrupamento por sentimento (Demszky et al., 2020).
SENTIMENT_GROUPS: dict[str, list[str]] = {
    "positive": [
        "admiration", "amusement", "approval", "caring", "desire",
        "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief",
    ],
    "negative": [
        "anger", "annoyance", "disappointment", "disapproval", "disgust",
        "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
    ],
    "ambiguous": ["confusion", "curiosity", "realization", "surprise"],
    "neutral": ["neutral"],
}

# Mapa inverso emocao -> grupo de sentimento.
LABEL2SENTIMENT: dict[str, str] = {
    label: group
    for group, labels in SENTIMENT_GROUPS.items()
    for label in labels
}
