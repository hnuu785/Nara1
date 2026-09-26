#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score a dev-set submission and save per-item metrics and error cases.

This utility is kept outside submit/ so it is not included in the competition ZIP.
It uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ITEMS = [f"v{i}" for i in range(1, 25)]


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"CSV에 id 열이 없습니다: {path}")
        rows: dict[str, dict[str, str]] = {}
        for line, row in enumerate(reader, start=2):
            record_id = (row.get("id") or "").strip()
            if not record_id:
                raise ValueError(f"빈 id: {path}:{line}")
            if record_id in rows:
                raise ValueError(f"중복 id {record_id}: {path}:{line}")
            rows[record_id] = row
    return rows


def binary(row: dict[str, str], item: str, record_id: str, path: Path) -> int:
    value = (row.get(item) or "").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{path}: id={record_id}, {item} 값이 0/1이 아닙니다: {value!r}")
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="dev 예측의 항목별 F1 및 오탐·누락 사례 계산")
    parser.add_argument("--pred", required=True, type=Path, help="추론 결과 submission.csv")
    parser.add_argument("--labels", type=Path, default=Path("dev_labels.csv"))
    parser.add_argument("--report-dir", type=Path, default=Path("output_dev"))
    args = parser.parse_args()

    predictions = read_rows(args.pred)
    labels = read_rows(args.labels)
    missing = sorted(labels.keys() - predictions.keys())
    extra = sorted(predictions.keys() - labels.keys())
    if missing or extra:
        raise ValueError(
            f"ID가 일치하지 않습니다. 예측 누락={missing[:10]} (총 {len(missing)}건), "
            f"예측에만 있음={extra[:10]} (총 {len(extra)}건)"
        )

    per_item: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in ITEMS:
        tp = fp = fn = 0
        for record_id, truth_row in labels.items():
            truth = binary(truth_row, item, record_id, args.labels)
            pred_row = predictions[record_id]
            prediction = binary(pred_row, item, record_id, args.pred)
            if truth == 1 and prediction == 1:
                tp += 1
            elif truth == 0 and prediction == 1:
                fp += 1
            elif truth == 1 and prediction == 0:
                fn += 1
            if truth != prediction:
                kind = "FP" if prediction == 1 else "FN"
                number = item[1:]
                errors.append({
                    "id": record_id,
                    "item": item,
                    "error": kind,
                    "true": truth,
                    "predicted": prediction,
                    "gold_evidence": truth_row.get(f"e{number}", "") or "",
                    "predicted_evidence": pred_row.get(f"e{number}", "") or "",
                })

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_item.append({
            "item": item,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    macro_f1 = sum(row["f1"] for row in per_item) / len(ITEMS)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.report_dir / "dev_metrics.json"
    errors_path = args.report_dir / "dev_error_cases.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({
            "records": len(labels),
            "macro_f1": macro_f1,
            "items": per_item,
            "error_cases": errors,
        }, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with errors_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["id", "item", "error", "true", "predicted", "gold_evidence", "predicted_evidence"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)

    print(f"records={len(labels)}  macro_f1={macro_f1:.6f}  errors={len(errors)}")
    print("item  TP  FP  FN  precision  recall  F1")
    for row in per_item:
        print(
            f"{row['item']:>3}  {row['tp']:>2}  {row['fp']:>2}  {row['fn']:>2}  "
            f"{row['precision']:.3f}      {row['recall']:.3f}  {row['f1']:.3f}"
        )
    print(f"metrics: {metrics_path}")
    print(f"FP/FN cases: {errors_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
