# Análise de Emoções em Textos — GoEmotions

Trabalho Final de PLN (PPGC/UFRGS). Classificação **multi-label** de emoções
de granularidade fina com Transformers, comparando **BERT** (baseline) e
**RoBERTa** (modelo principal) no dataset **GoEmotions** (28 classes:
27 emoções + neutro).

> GoEmotions é multi-label: um comentário pode expressar várias emoções
> simultaneamente. Usamos uma sigmoid independente por classe
> (`BCEWithLogitsLoss`) com limiar de decisão 0.5.

## Estrutura

```
.
├── configs/            # hiperparâmetros (bert.yaml, roberta.yaml)
├── src/
│   ├── labels.py       # taxonomia das 28 emoções + grupos de sentimento
│   ├── dataset.py      # carga/tokenização do GoEmotions (multi-hot)
│   ├── metrics.py      # Macro/Micro/Weighted F1, subset accuracy
│   ├── train.py        # fine-tuning (HuggingFace Trainer)
│   └── evaluate.py     # avaliação detalhada e análise de erros
├── notebooks/
│   └── colab_train.ipynb   # treino pesado no Colab (GPU T4)
├── results/            # métricas, checkpoints e gráficos (gerados)
├── data/cache/         # cache do dataset (gerado, ignorado pelo git)
└── requirements.txt
```

## Setup local (GPU NVIDIA)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# PyTorch com CUDA 12.1 (índice próprio):
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# Demais dependências:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Validar a GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Uso

Teste rápido (subset minúsculo, 1 época) para validar o pipeline:

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs/bert.yaml --smoke
```

Treino completo:

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs/bert.yaml
.\.venv\Scripts\python.exe -m src.train --config configs/roberta.yaml
```

Avaliação detalhada (gera relatórios e gráficos em `results/<modelo>/`):

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --model_dir results/bert/best_model --output_dir results/bert
.\.venv\Scripts\python.exe -m src.evaluate --model_dir results/roberta/best_model --output_dir results/roberta
```

## Métricas

Com 28 classes desbalanceadas, a accuracy isolada é pouco informativa:

| Métrica | Papel | Descrição |
|---|---|---|
| **Macro F1** | Principal | Média simples do F1 por classe (trata todas igualmente). |
| **Micro F1** | Comparação | Agrega TP/FP/FN globais; usado no paper original. |
| **Weighted F1** | Complementar | Média do F1 ponderada pelo suporte de cada classe. |
| **Subset accuracy** | Referência | Fração de exemplos com todos os rótulos corretos (exact match). |

## Referência

Demszky, D. et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* ACL.
