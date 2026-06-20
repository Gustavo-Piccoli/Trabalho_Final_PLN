"""Avaliacao detalhada e analise de erros de um modelo treinado.

Vai alem das metricas agregadas e produz os artefatos que sustentam a
analise do artigo:

  - classification_report.csv : precisao/recall/F1/suporte por classe.
  - per_class_f1.csv          : F1 por classe, ranqueado (emocoes mais dificeis).
  - confusion_gold_pred.(csv|png): matriz 28x28 de co-ativacao ouro-vs-predito,
                                  normalizada por linha -> revela quais emocoes
                                  sao confundidas entre si.
  - sentiment_confusion.(csv|png): mesma ideia agregada nos 4 grupos de sentimento.
  - error_examples.csv        : amostra de falsos positivos / negativos para
                                  analise qualitativa.

Uso:
    python -m src.evaluate --model_dir results/bert/best_model --output_dir results/bert
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.dataset import MultiLabelCollator, load_goemotions
from src.labels import GOEMOTIONS_LABELS, LABEL2SENTIMENT, SENTIMENT_GROUPS
from src.metrics import compute_multilabel_metrics, logits_to_preds

matplotlib.use("Agg")  # backend sem display (servidor/Colab)
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


@torch.no_grad()
def predict_logits(model, dataset, collator, batch_size: int, device: str) -> np.ndarray:
    """Roda inferencia em batches e retorna a matriz de logits (N, 28)."""
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    chunks = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        chunks.append(logits.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def save_heatmap(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    """Salva um heatmap (matriz quadrada rotulada) em PNG."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(7, n * 0.45)))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="viridis",
                vmin=0.0, vmax=1.0, square=True, ax=ax, cbar_kws={"shrink": 0.6})
    ax.set_title(title)
    ax.set_ylabel("Rotulo verdadeiro (ouro)")
    ax.set_xlabel("Rotulo predito")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliacao detalhada GoEmotions")
    parser.add_argument("--model_dir", required=True, help="Diretorio do modelo treinado")
    parser.add_argument("--output_dir", required=True, help="Onde salvar os artefatos")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] dispositivo: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    ds = load_goemotions(tokenizer, max_length=args.max_length, smoke=args.smoke)
    test = ds["test"]
    gold = np.asarray(test["labels"]).astype(int)

    logits = predict_logits(model, test, MultiLabelCollator(tokenizer), args.batch_size, device)
    preds = logits_to_preds(logits, args.threshold)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Metricas agregadas ---
    agg = compute_multilabel_metrics(logits, gold, args.threshold)
    print("\n[metricas agregadas]")
    for k, v in agg.items():
        print(f"  {k:18s}: {v:.4f}")

    # --- Relatorio por classe ---
    report = classification_report(
        gold, preds, target_names=GOEMOTIONS_LABELS,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out / "classification_report.csv")

    per_class = report_df.loc[GOEMOTIONS_LABELS, ["precision", "recall", "f1-score", "support"]]
    per_class = per_class.sort_values("f1-score", ascending=False)
    per_class.to_csv(out / "per_class_f1.csv")
    print("\n[5 emocoes mais faceis]")
    print(per_class.head(5).to_string())
    print("\n[5 emocoes mais dificeis]")
    print(per_class.tail(5).to_string())

    # --- Matriz de confusao ouro-vs-predito (co-ativacao) ---
    # C[i,j] = nº de exemplos com ouro=i e predito=j; normalizada por linha.
    cooc = gold.T @ preds  # (28, 28)
    row_sums = cooc.sum(axis=1, keepdims=True)
    cooc_norm = np.divide(cooc, row_sums, out=np.zeros_like(cooc, dtype=float),
                          where=row_sums != 0)
    pd.DataFrame(cooc_norm, index=GOEMOTIONS_LABELS, columns=GOEMOTIONS_LABELS) \
        .to_csv(out / "confusion_gold_pred.csv")
    save_heatmap(cooc_norm, GOEMOTIONS_LABELS,
                 "Confusao ouro-vs-predito (normalizada por linha)",
                 out / "confusion_gold_pred.png")

    # --- Agregacao por grupo de sentimento (4x4) ---
    groups = list(SENTIMENT_GROUPS.keys())
    g_index = {g: i for i, g in enumerate(groups)}
    gold_g = np.zeros((gold.shape[0], len(groups)), dtype=int)
    pred_g = np.zeros_like(gold_g)
    for j, emo in enumerate(GOEMOTIONS_LABELS):
        gi = g_index[LABEL2SENTIMENT[emo]]
        gold_g[:, gi] |= gold[:, j]
        pred_g[:, gi] |= preds[:, j]
    sent_cooc = gold_g.T @ pred_g
    rs = sent_cooc.sum(axis=1, keepdims=True)
    sent_norm = np.divide(sent_cooc, rs, out=np.zeros_like(sent_cooc, dtype=float),
                          where=rs != 0)
    pd.DataFrame(sent_norm, index=groups, columns=groups) \
        .to_csv(out / "sentiment_confusion.csv")
    save_heatmap(sent_norm, groups,
                 "Confusao por grupo de sentimento (normalizada por linha)",
                 out / "sentiment_confusion.png")

    # --- Exemplos de erro para analise qualitativa ---
    # load_goemotions remove a coluna 'text' (so mantem tensores); recarregamos
    # os textos crus do split de teste apenas para exibir os exemplos.
    from datasets import load_dataset
    raw_texts = load_dataset("google-research-datasets/go_emotions", "simplified",
                             cache_dir="data/cache")["test"]["text"]
    if args.smoke:
        raw_texts = raw_texts[:len(gold)]

    rows = []
    for i in range(len(gold)):
        g_set = {GOEMOTIONS_LABELS[j] for j in np.where(gold[i] == 1)[0]}
        p_set = {GOEMOTIONS_LABELS[j] for j in np.where(preds[i] == 1)[0]}
        if g_set != p_set:
            rows.append({
                "text": raw_texts[i],
                "gold": ", ".join(sorted(g_set)) or "(nenhum)",
                "pred": ", ".join(sorted(p_set)) or "(nenhum)",
                "false_negatives": ", ".join(sorted(g_set - p_set)),
                "false_positives": ", ".join(sorted(p_set - g_set)),
            })
    err_df = pd.DataFrame(rows)
    err_df.head(200).to_csv(out / "error_examples.csv", index=False)
    print(f"\n[ok] {len(err_df)} exemplos com erro "
          f"({100 * len(err_df) / len(gold):.1f}% do teste); "
          f"amostra salva em {out / 'error_examples.csv'}")
    print(f"[ok] artefatos de analise salvos em {out}")


if __name__ == "__main__":
    main()
