"""선폭 측정 3방법 성능 평가를 DagsHub MLflow(IDA_Width_Measurement)에 방법별 Run으로 올린다.

  보정(val)  : 노트북1 노광 사진 15장, 사람 정답 450개 → val/ 지표
  평가(test) : 노트북2 사진 15장, 사람 정답 450개   → test/ 지표 (보정에 안 쓴 사진)
  안정성     : 안정성 사진 27장 (반복 촬영·조명 3단계) → stability/ 지표

usage:
  python log_width_benchmark.py --predictions <pred_*.csv 폴더> --ground-truth <filled.csv>
         --owner <이름> [--attach 파일 ...] [--dry-run]
pred 파일 이름: pred_<laptop1|laptop2|stability>_<unet|canny|threshold_0.57>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking.width_run_info import WidthRunInfo  # noqa: E402

UM_PER_PX = 0.1592
MAX_DIST_PX = 30.0
REPEAT_DIST_PX = 20.0
Y_FRACTIONS = "0.25,0.45,0.65"
# 결정용 핵심 지표: 정확도, 반복성, 조명 민감도, 성공률(이물 많은 사진 포함), 소요시간
CORE_TEST = ("mae_um", "runtime_median_ms")
CORE_STABILITY = ("repeat_target_sd_um", "lighting_image_mean_range_um", "success_rate")
METHODS = {
    "unet": {"model_name": "unet-sidewall", "edge_parameter": "probability_threshold=0.50"},
    "canny": {"model_name": "canny-sobel-gradient", "edge_parameter": "gradient_peak, offset=0"},
    "threshold_0.57": {"model_name": "threshold-green", "edge_parameter": "dark_band_fraction=0.57"},
}


def load(folder: Path, split: str, key: str) -> pd.DataFrame:
    d = pd.read_csv(folder / f"pred_{split}_{key}.csv", encoding="utf-8-sig")
    d["ok"] = d.status.eq("ok") & d.width_px.notna()
    d["y_fraction"] = d.y_fraction.round(2)
    return d


def nearest(ref: pd.DataFrame, other: pd.DataFrame, limit: float) -> dict:
    pool, pairs = other.copy(), {}
    for i, r in ref.iterrows():
        if pool.empty:
            break
        dist = (pool.center_x_px - r.center_x_px).abs()
        j = dist.idxmin()
        if dist[j] <= limit:
            pairs[i] = j
            pool = pool.drop(j)
    return pairs


def accuracy(pred: pd.DataFrame, gt: pd.DataFrame) -> dict:
    ok = pred[pred.ok]
    errors = []
    for (img, kind, y), g in gt.groupby(["image_id", "measurement_type", "y_fraction"]):
        cand = ok[(ok.image_id == img) & (ok.measurement_type == kind) & (ok.y_fraction == y)]
        pairs = nearest(g, cand, MAX_DIST_PX)
        for i, r in g.iterrows():
            pred_um = cand.loc[pairs[i], "width_px"] * UM_PER_PX if i in pairs else np.nan
            errors.append({"image_id": img, "type": kind, "err": pred_um - r.gt_um})
    e = pd.DataFrame(errors)
    valid = e.err.dropna()
    image_err = e.groupby(["image_id", "type"]).err.mean().abs()
    runtime = pd.to_numeric(pred.drop_duplicates("image_id").runtime_ms, errors="coerce").dropna()
    return {
        "mae_um": valid.abs().mean(),
        "bias_line_um": e[e.type == "line"].err.mean(),
        "bias_gap_um": e[e.type == "gap"].err.mean(),
        "p95_abs_err_um": valid.abs().quantile(0.95),
        "mae_after_bias_removal_um": (valid - valid.mean()).abs().mean(),
        "outlier_rate_over_1um": (valid.abs() > 1.0).mean(),
        "image_mean_abs_err_um": image_err.mean(),
        "image_mean_max_abs_err_um": image_err.max(),
        "gt_coverage_rate": e.err.notna().mean(),
        "success_rate": pred.ok.mean(),
        "runtime_median_ms": runtime.median(),
    }


def stability(pred: pd.DataFrame) -> dict:
    p = pred.copy()
    p["folder"] = p.image_id.str.split("__").str[0]
    shot = p.image_id.str.split("__").str[1]
    p["sample"], p["level"] = shot.str[:-1], shot.str[-1].astype(int)
    repeat_sd, repeat_img, light_range, light_img = [], [], [], []
    for (folder, _sample), g in p.groupby(["folder", "sample"]):
        good = g[g.ok]
        lighting = folder == "조명 비교"
        for (_kind, _y), gy in good.groupby(["measurement_type", "y_fraction"]):
            ref = gy[gy.level == 1]
            tracks = {i: [r.width_px] for i, r in ref.iterrows()}
            for level in (2, 3):
                for i, j in nearest(ref, gy[gy.level == level], REPEAT_DIST_PX).items():
                    tracks[i].append(gy.loc[j, "width_px"])
            for values in tracks.values():
                if len(values) == 3:
                    v = np.asarray(values) * UM_PER_PX
                    (light_range if lighting else repeat_sd).append(np.ptp(v) if lighting else v.std(ddof=1))
        means = good.groupby(["level", "measurement_type"]).width_px.mean().unstack() * UM_PER_PX
        for kind in means.columns:
            (light_img if lighting else repeat_img).append(
                np.ptp(means[kind]) if lighting else means[kind].std(ddof=1))
    return {
        "repeat_target_sd_um": np.mean(repeat_sd),
        "repeat_image_mean_sd_um": np.nanmean(repeat_img),
        "lighting_target_range_um": np.mean(light_range),
        "lighting_image_mean_range_um": np.nanmean(light_img),
        "success_rate": p.ok.mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--environment", default="windows11-cpu")
    parser.add_argument("--dataset-version", default="gt-line-gap-y25-45-65-v1")
    parser.add_argument("--attach", type=Path, nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--core-only", action="store_true",
                        help="결정용 핵심 지표 5개만 기록 (나머지는 첨부 Excel에)")
    args = parser.parse_args()

    gt = pd.read_csv(args.ground_truth, encoding="utf-8-sig")
    gt["y_fraction"] = gt.y_fraction.round(2)
    gt["center_x_px"] = (gt.left_x_px + gt.right_x_px) / 2
    gt["gt_um"] = (gt.right_x_px - gt.left_x_px) * UM_PER_PX
    splits = {name: gt[gt.image_id.str.startswith(name)] for name in ("laptop1", "laptop2")}

    results = {}
    for key in METHODS:
        results[key] = {
            "val": accuracy(load(args.predictions, "laptop1", key), splits["laptop1"]),
            "test": accuracy(load(args.predictions, "laptop2", key), splits["laptop2"]),
            "stability": stability(load(args.predictions, "stability", key)),
        }
    table = pd.DataFrame({k: {f"{s}/{m}": v for s, d in r.items() for m, v in d.items()} for k, r in results.items()})
    print(table.round(4).to_string())
    if args.dry_run:
        return

    from experiment_logger import log_file, log_metrics, log_test_metrics, tracked_run
    import mlflow

    for key, spec in METHODS.items():
        info = WidthRunInfo(
            owner=args.owner, environment=args.environment, model_name=spec["model_name"],
            dataset_version=args.dataset_version, seed=0,
            val_split_id="laptop1-exposure-15img", test_split_id="laptop2-15img",
        )
        params = {
            "method": key.split("_")[0], "edge_parameter": spec["edge_parameter"],
            "edge_definition": "inner edge (red sidewall to line/gap)",
            "y_fractions": Y_FRACTIONS, "targets_per_row": 5, "um_per_px": UM_PER_PX,
            "measurement_types": "line,gap", "gt_match_max_distance_px": MAX_DIST_PX,
            "calibration_set": "laptop1 15 images (450 GT)", "test_set": "laptop2 15 images (450 GT)",
            "stability_set": "27 images: repeat x2 folders, lighting x1 folder",
        }
        with tracked_run(info, params=params) as run:
            if args.core_only:
                test, stab = results[key]["test"], results[key]["stability"]
                log_test_metrics({k: test[k] for k in CORE_TEST}, split_id="laptop2-15img")
                log_metrics({k: stab[k] for k in CORE_STABILITY}, 0, prefix="stability")
                mlflow.set_tag("metric_set", "core5")
            else:
                log_metrics(results[key]["val"], 0, prefix="val")
                log_metrics(results[key]["stability"], 0, prefix="stability")
                log_test_metrics(results[key]["test"], split_id="laptop2-15img")
            mlflow.set_tags({"gt_annotator": "manual", "runtime_note": "same PC, CPU, first image excluded"})
            for split in ("laptop1", "laptop2", "stability"):
                log_file(args.predictions / f"pred_{split}_{key}.csv", artifact_path="predictions", required=True)
            log_file(args.ground_truth, artifact_path="ground_truth", required=True)
            for path in args.attach:
                log_file(path, artifact_path="results", required=True)
            print(f"logged {key}: run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
