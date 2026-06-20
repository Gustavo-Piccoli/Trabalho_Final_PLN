"""Fine-tuning de modelos Transformer no GoEmotions (multi-label).

Usa AutoModelForSequenceClassification com problem_type=
"multi_label_classification", o que ativa automaticamente a
BCEWithLogitsLoss (uma sigmoid independente por classe).

Uso:
    python -m src.train --config configs/bert.yaml
    python -m src.train --config configs/roberta.yaml
    python -m src.train --config configs/bert.yaml --smoke   # teste rapido

O melhor checkpoint (maior Macro F1 na validacao) e mantido ao final, e as
metricas de validacao e teste sao salvas em <output_dir>/metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.dataset import MultiLabelCollator, load_goemotions
from src.labels import ID2LABEL, LABEL2ID, NUM_LABELS
from src.metrics import build_compute_metrics


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning GoEmotions multi-label")
    parser.add_argument("--config", required=True, help="Caminho do YAML de configuracao")
    parser.add_argument("--smoke", action="store_true", help="Teste rapido com subset minusculo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    use_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"[info] CUDA disponivel: {use_cuda} | dispositivo: {device_name}")

    # --- Tokenizador e dados ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    ds = load_goemotions(tokenizer, max_length=int(cfg["max_length"]), smoke=args.smoke)
    print(f"[info] splits: train={len(ds['train'])} "
          f"val={len(ds['validation'])} test={len(ds['test'])}")

    # --- Modelo ---
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # --- Argumentos de treino ---
    tcfg = cfg["training"]
    fp16 = bool(tcfg.get("fp16", False)) and use_cuda
    epochs = 1 if args.smoke else float(tcfg["num_train_epochs"])

    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        run_name=cfg.get("run_name"),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(tcfg["learning_rate"]),
        num_train_epochs=epochs,
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.0)),
        fp16=fp16,
        logging_steps=int(tcfg.get("logging_steps", 50)),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=MultiLabelCollator(tokenizer),
        compute_metrics=build_compute_metrics(float(cfg["threshold"])),
    )

    # --- Treino ---
    trainer.train()

    # --- Avaliacao (melhor modelo ja carregado) ---
    val_metrics = trainer.evaluate(ds["validation"], metric_key_prefix="val")
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_name": cfg["model_name"],
        "smoke": args.smoke,
        "threshold": float(cfg["threshold"]),
        "max_length": int(cfg["max_length"]),
        "epochs": epochs,
        "validation": val_metrics,
        "test": test_metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Salva melhor modelo + tokenizador para a etapa de avaliacao detalhada.
    best_dir = out_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    print("\n===== RESULTADOS =====")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[ok] Modelo salvo em: {best_dir}")
    print(f"[ok] Metricas salvas em: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
