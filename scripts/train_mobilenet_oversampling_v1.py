"""One fixed rare-class oversampling experiment, reusing the real-only baseline."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import traceback
import warnings

import train_mobilenet_real_only_v1 as baseline

ROOT=baseline.ROOT
BASE_REPORT=ROOT/"data/derived/mobilenet_baseline_v1"
BASE_RUN=ROOT/"runs/mobilenet/mobilenet_v3_small_real_only_v1"
REPORT=ROOT/"data/derived/mobilenet_oversampling_v1"
RUN=ROOT/"runs/mobilenet/mobilenet_v3_small_oversample_v1"
SUMMARY=REPORT/"mobilenet_oversampling_summary.json"
FACTORS={"p":1,"o":10,"s":5,"sc":2}
EFFECTIVE={"p":4352,"o":20,"s":25,"sc":798}
OriginalCropDataset=baseline.CropDataset


class OversampledCropDataset(OriginalCropDataset):
    """Expand Train indices only; Val/Test rows and file bytes are unchanged."""
    def __init__(self,rows,split,transform):
        super().__init__(rows,split,transform)
        if split=="train":
            self.rows=[row for row in self.rows for _ in range(FACTORS[row["class"]])]
            assert len(self.rows)==5195
            assert Counter(r["class"] for r in self.rows)==EFFECTIVE


def set_output_paths():
    # Output destinations only. The baseline source file/model/training loop are not edited.
    baseline.REPORT=REPORT
    baseline.RUN=RUN
    baseline.SUMMARY=SUMMARY


def verify_protected(protected):
    changed=[name for name,sha in protected.items()
             if not (ROOT/name).is_file() or baseline.digest(ROOT/name)!=sha]
    assert not changed,f"Protected inputs changed: {changed[:5]}"


def prepare():
    if REPORT.exists() or RUN.exists():
        raise FileExistsError("Never overwrite an existing oversampling experiment")
    base=json.loads((BASE_REPORT/"mobilenet_baseline_summary.json").read_text(encoding="utf-8"))
    base_config=json.loads((BASE_REPORT/"training_config.json").read_text(encoding="utf-8"))
    assert base["training_success"] and base["status"]=="COMPLETED"
    assert base["config"]==base_config and base["test_evaluation_calls"]==1
    assert base["script_sha256"]==baseline.digest(Path(baseline.__file__))
    assert baseline.digest(BASE_RUN/"best.pt")==base["best_checkpoint_sha256"]
    protected=dict(base["protected_input_sha256"])
    for folder in [BASE_REPORT,BASE_RUN]:
        for path in folder.rglob("*"):
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()]=baseline.digest(path)
    protected[Path(baseline.__file__).relative_to(ROOT).as_posix()]=baseline.digest(Path(baseline.__file__))
    verify_protected(protected)
    rows=baseline.read_rows()
    assert len(rows)==5953 and len({r["crop_filename"] for r in rows})==5953
    assert len({r["annotation_id"] for r in rows})==5953
    config=json.loads(json.dumps(base_config))
    assert config["class_order"]==["p","o","s","sc"] and config["device"]=="cpu"
    assert config["batch_size"]==32 and config["max_epochs"]==50
    assert config["early_stopping_patience"]==10 and config["seed"]==42
    assert config["optimizer"]=="AdamW" and config["learning_rate"]==1e-3
    assert config["weight_decay"]==1e-4 and config["class_weight"] is None
    assert config["loss"]=="CrossEntropyLoss" and config["label_smoothing"]==0
    assert config["environment"]["torch"]==str(baseline.torch.__version__)
    assert config["environment"]["torchvision"]==str(baseline.torchvision.__version__)
    assert config["environment"]["numpy"]==baseline.np.__version__
    assert config["torch_num_threads"]==baseline.torch.get_num_threads()
    assert config["torch_num_interop_threads"]==baseline.torch.get_num_interop_threads()
    assert not baseline.torch.cuda.is_available()
    assert baseline.digest(baseline.WEIGHT)==config["pretrained_sha256"]
    assert not any(config[k] for k in ["augmentation","weighted_sampler","undersampling","synthetic"])
    config.update(oversampling=True,sample_duplication=True)
    config_diff={k:{"baseline":base_config[k],"oversampling":v}
                 for k,v in config.items() if v!=base_config[k]}
    assert set(config_diff)=={"oversampling","sample_duplication"}
    manifest=[]
    source_splits={}
    transform=baseline.preprocess(config)
    for split in ["train","val","test"]:
        selected=[r for r in rows if r["split"]==split]
        assert Counter(r["class"] for r in selected)==Counter(baseline.EXPECTED[split])
        dataset=OriginalCropDataset(rows,split,transform)
        assert [r["crop_filename"] for r in dataset.rows]==[r["crop_filename"] for r in selected]
        for row in selected:
            source_splits.setdefault(row["source_image_id"],set()).add(split)
            path=baseline.DATA/row["crop_relative_path"]
            with baseline.Image.open(path) as image:
                image.load()
                assert image.mode=="RGB" and image.size==(int(row["crop_width"]),int(row["crop_height"]))
            if split=="train":
                factor=FACTORS[row["class"]]
                for repeat in range(1,factor+1):
                    manifest.append({
                        "effective_sample_index":len(manifest),"source_crop":row["crop_relative_path"],
                        "crop_filename":row["crop_filename"],"class":row["class"],
                        "repeat_index":repeat,"repeat_factor":factor,"split":"train",
                        "source_filename":row["source_filename"],"source_image_id":row["source_image_id"],
                        "annotation_id":row["annotation_id"],"crop_sha256":row["crop_file_sha256"],
                    })
    assert all(len(v)==1 for v in source_splits.values())
    assert len(manifest)==5195 and Counter(r["class"] for r in manifest)==EFFECTIVE
    repeats=Counter(r["source_crop"] for r in manifest)
    assert len(repeats)==4758
    for row in manifest:
        assert repeats[row["source_crop"]]==FACTORS[row["class"]]==row["repeat_factor"]
    import csv
    with (BASE_REPORT/"test_predictions.csv").open(encoding="utf-8-sig",newline="") as file:
        base_predictions=list(csv.DictReader(file))
    assert [r["crop_filename"] for r in rows if r["split"]=="test"]==[r["crop_filename"] for r in base_predictions]
    assert [r["class"] for r in rows if r["split"]=="test"]==[r["true_class"] for r in base_predictions]
    # Pure metric and deterministic preprocessing checks, not extra model/test evaluations.
    assert baseline.classification_metrics(baseline.np.eye(4,dtype=int))["macro_f1"]==1
    assert baseline.classification_metrics(baseline.np.diag([1,0,1,1]))["macro_f1"]==.75
    val_base=OriginalCropDataset(rows,"val",baseline.preprocess(base_config))
    val_os=OversampledCropDataset(rows,"val",transform)
    assert val_base.rows==val_os.rows
    assert baseline.torch.equal(val_base[0][0],val_os[0][0])
    train_os=OversampledCropDataset(rows,"train",transform)
    assert len(train_os)==5195 and Counter(r["class"] for r in train_os.rows)==EFFECTIVE
    REPORT.mkdir(parents=True)
    RUN.mkdir(parents=True)
    baseline.csv_write(REPORT/"oversampling_manifest.csv",manifest)
    baseline.json_write(REPORT/"training_config.json",config)
    baseline.json_write(RUN/"training_config.json",config)
    for path in [Path(__file__).resolve(),Path(baseline.__file__).resolve()]:
        shutil.copy2(path,RUN/path.name)
    unique_factors=Counter(FACTORS[r["class"]] for r in rows if r["split"]=="train")
    effective_factors=Counter(r["repeat_factor"] for r in manifest)
    summary={
        "status":"PREPARED","training_success":False,"dataset_preparation_success":True,"error":None,
        "model":config["model"],"pretrained":True,"config":config,
        "baseline_summary_path":str(BASE_REPORT/"mobilenet_baseline_summary.json"),
        "baseline_config_comparison":{"equal_except_oversampling":True,"differences":config_diff},
        "repeat_factors":FACTORS,"original_unique_train":4758,"effective_train":5195,
        "unique_class_counts":baseline.EXPECTED["train"],"effective_class_counts":EFFECTIVE,
        "repeat_factor_unique_crop_counts":dict(unique_factors),
        "repeat_factor_effective_sample_counts":dict(effective_factors),
        "sample_counts":{"train":5195,"val":597,"test":598},
        "unique_sample_counts":{"train":4758,"val":597,"test":598},
        "class_support":{"train":EFFECTIVE,"val":baseline.EXPECTED["val"],"test":baseline.EXPECTED["test"]},
        "oversampling_implementation":"Explicit deterministic Train index expansion; RandomSampler without replacement over expanded indices, no weighted sampling or PNG copies.",
        "morphology_note":"Repeated exposure to the same 2 open and 5 short crops; no new morphology or unique real samples.",
        "val_test_unchanged_before_training":True,"test_order_matches_baseline":True,
        "protected_input_sha256":protected,"dataset_readable":5953,"unreadable_samples":0,
        "loader_exclusions":0,"actual_epochs":0,"best_epoch":None,"best_val_macro_f1":None,
        "nan_detected":False,"test_evaluation_calls":0,"warnings":[],
        "script_sha256":baseline.digest(Path(__file__).resolve()),
        "baseline_helper_sha256":baseline.digest(Path(baseline.__file__).resolve()),
        "started_utc":datetime.now(timezone.utc).isoformat(),
    }
    baseline.json_write(SUMMARY,summary)
    print("OVERSAMPLING_PREPARED="+json.dumps({k:summary[k] for k in [
        "original_unique_train","effective_train","unique_class_counts","effective_class_counts",
        "repeat_factor_unique_crop_counts","repeat_factor_effective_sample_counts",
        "val_test_unchanged_before_training","baseline_config_comparison"]}),flush=True)
    return summary


def write_results(summary,history,predictions):
    test=summary["test"]
    base=json.loads((BASE_REPORT/"mobilenet_baseline_summary.json").read_text(encoding="utf-8"))
    baseline.csv_write(REPORT/"classification_report.csv",[
        {"class":name,**test["per_class"][name]} for name in baseline.CLASS_ORDER])
    baseline.csv_write(REPORT/"confusion_matrix.csv",[
        {"true_class":name,**dict(zip(baseline.CLASS_ORDER,row))}
        for name,row in zip(baseline.CLASS_ORDER,test["confusion_matrix"])])
    baseline.csv_write(REPORT/"test_predictions.csv",predictions)
    baseline.make_plots(summary,history)
    overall_keys=["accuracy","macro_precision","macro_recall","macro_f1","weighted_f1"]
    summary["baseline_comparison"]={
        "overall":{k:{"baseline":base["test"][k],"oversampling":test[k],
                      "delta":test[k]-base["test"][k]} for k in overall_keys},
        "per_class":{name:{k:{
            "baseline":base["test"]["per_class"][name][k],"oversampling":test["per_class"][name][k],
            "delta":test["per_class"][name][k]-base["test"]["per_class"][name][k]}
            for k in ["precision","recall","f1"]} for name in baseline.CLASS_ORDER},
        "confusion_matrix_delta":(baseline.np.array(test["confusion_matrix"])-
                                  baseline.np.array(base["test"]["confusion_matrix"])).tolist(),
        "sc_to_p":{"baseline":base["test"]["confusion_matrix"][3][0],"oversampling":test["confusion_matrix"][3][0]},
        "p_prediction_fraction":{"baseline":base["majority_class_evidence"]["predicted_particle_fraction"],
                                 "oversampling":summary["majority_class_evidence"]["predicted_particle_fraction"]},
    }
    lines=[
        "# MobileNetV3-Small rare-class oversampling v1","",
        "- Official ImageNet pretrained; full-network fine-tuning; classes [p,o,s,sc].",
        "- Unique Train crops: 4758; effective Train indices: 5195; Val: 597; Test: 598.",
        "- Train repeat factors: p x1, o x10, s x5, sc x2. Effective counts: p4352, o20, s25, sc798.",
        "- No crop files copied/changed; no new morphology. Same 2 open and 5 short crops repeated.",
        "- The only differences from baseline config are oversampling=True and sample_duplication=True.",
        "- Seed42; batch32; max50 epochs; patience10; AdamW lr1e-3 weight_decay1e-4; unweighted CrossEntropyLoss.",
        "- Same deterministic RGB / direct bilinear224x224 / tensor / ImageNet normalization; no augmentation.",
        "- Same CPU, pretrained SHA256, model, optimizer, class order, Val/Test bytes, order and preprocessing.",
        f"- Actual epochs: {summary['actual_epochs']}; best epoch: {summary['best_epoch']}; best Val macro-F1: {summary['best_val_macro_f1']:.8f}.",
        "- Selection: maximize Val macro-F1, strict improvement, earliest tie, patience10. Test evaluated once after selection.",
        "- All four classes included with zero_division=0. Val o support0 contributes F1=0 (macro-F1 upper bound0.75).",
        "- Online Train metrics use ordinary training-pass predictions; Val/Test metrics use inference mode.","",
        "## Final best-checkpoint Validation and Test","",
        "| Split | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name,metrics in [("Val",summary["best_val"]),("Test",test)]:
        lines.append("| "+name+" | "+" | ".join(f"{metrics[k]:.8f}" for k in overall_keys)+" |")
    lines+=["","## Test per-class metrics","",
            "| Class | Support | Predicted count | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|"]
    for name,item in test["per_class"].items():
        lines.append(f"| {name} | {item['support']} | {item['predicted_count']} | "
                     f"{item['precision']:.8f} | {item['recall']:.8f} | {item['f1']:.8f} |")
    lines+=["","## Test comparison (oversampling minus baseline)","",
            "| Metric | Baseline | Oversampling | Delta |","|---|---:|---:|---:|"]
    for name,item in summary["baseline_comparison"]["overall"].items():
        lines.append(f"| {name} | {item['baseline']:.8f} | {item['oversampling']:.8f} | {item['delta']:+.8f} |")
    for name in baseline.CLASS_ORDER:
        for metric,item in summary["baseline_comparison"]["per_class"][name].items():
            lines.append(f"| {name} {metric} | {item['baseline']:.8f} | {item['oversampling']:.8f} | {item['delta']:+.8f} |")
    lines+=["","## Test confusion matrix","",
            "Rows GT, columns prediction; order p,o,s,sc.","",
            "| GT | p | o | s | sc |","|---|---:|---:|---:|---:|"]
    for name,row in zip(baseline.CLASS_ORDER,test["confusion_matrix"]):
        lines.append("| "+name+" | "+" | ".join(map(str,row))+" |")
    comparison=summary["baseline_comparison"]
    lines+=["","## Observed cases","",
            f"- sc -> p: {comparison['sc_to_p']['baseline']} -> {comparison['sc_to_p']['oversampling']}.",
            f"- p prediction fraction: {comparison['p_prediction_fraction']['baseline']:.8f} -> {comparison['p_prediction_fraction']['oversampling']:.8f}.",
            f"- Predicted o count: {test['per_class']['o']['predicted_count']}; predicted s count: {test['per_class']['s']['predicted_count']}."]
    for item in summary["rare_test_predictions"]:
        lines.append(f"- GT {item['true_class']} -> {item['predicted_class']}, confidence={item['confidence']:.8f}: {item['crop_filename']}.")
    lines+=["",
            "- Test open and short each have one GT example; these are individual observations, not evidence of generalization.",
            "- Accuracy alone is insufficient because particle represents about91% of Test crops.",
            "- This is GT-bbox classification, not whole-image defect detection; classification F1 is not YOLO AP.","",
            "## Traceability","",
            f"- Best checkpoint: {RUN/'best.pt'}.",
            f"- Training minutes: {summary['training_time_min']:.3f}.",
            f"- Protected original crops, fixed split and baseline artifacts unchanged: {summary['protected_inputs_unchanged']}.",
            f"- NaN/Inf: {summary['nan_detected']}; unreadable: {summary['unreadable_samples']}; excluded: {summary['loader_exclusions']}.",
            "- Warnings: "+("; ".join(summary["warnings"]) if summary["warnings"] else "none")+".",
            f"- Error: {summary['error']}.",""]
    (REPORT/"mobilenet_oversampling_results.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--prepare-only",action="store_true")
    group.add_argument("--train-prepared",action="store_true")
    args=parser.parse_args()
    set_output_paths()
    if args.train_prepared:
        summary=json.loads(SUMMARY.read_text(encoding="utf-8"))
        assert summary["status"]=="PREPARED" and not (REPORT/"training.log").exists()
        assert summary["script_sha256"]==baseline.digest(Path(__file__).resolve())
        assert summary["baseline_helper_sha256"]==baseline.digest(Path(baseline.__file__).resolve())
        verify_protected(summary["protected_input_sha256"])
    else:
        summary=prepare()
    if args.prepare_only:
        return
    with (REPORT/"training.log").open("x",encoding="utf-8",buffering=1) as logfile:
        with contextlib.redirect_stdout(baseline.Tee(sys.stdout,logfile)),contextlib.redirect_stderr(baseline.Tee(sys.stderr,logfile)):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    baseline.CropDataset=OversampledCropDataset
                    try:
                        history,predictions=baseline.train(summary)
                    finally:
                        baseline.CropDataset=OriginalCropDataset
                    summary["warnings"]=list(dict.fromkeys(str(w.message) for w in caught))
                    write_results(summary,history,predictions)
                    summary["warnings"]=list(dict.fromkeys(str(w.message) for w in caught))
                summary.update(status="COMPLETED",training_success=True,error=None)
                baseline.json_write(SUMMARY,summary)
                print("OVERSAMPLING_COMPLETED="+json.dumps({k:summary[k] for k in
                      ["training_success","actual_epochs","best_epoch","best_val_macro_f1",
                       "test","baseline_comparison","best_checkpoint","warnings"]}),flush=True)
            except Exception as error:
                summary.update(status="FAILED",training_success=False,error=repr(error),
                               nan_detected=isinstance(error,FloatingPointError))
                baseline.json_write(SUMMARY,summary)
                traceback.print_exc()
                raise


if __name__=="__main__":
    main()
