"""One fixed ResNet18 rare-class oversampling run; same baseline training loop."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import contextlib
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import shutil
import sys
import traceback
import warnings

import train_resnet18_baseline_v1 as baseline

common=baseline.common
ROOT=baseline.ROOT
DATA=baseline.DATA
SCRIPT=Path(__file__).resolve()
BASE_RUN=ROOT/"runs/resnet/resnet18_real_only_v1"
BASE_REPORT=ROOT/"data/derived/resnet18_baseline_v1"
RUN=ROOT/"runs/resnet/resnet18_oversample_v1"
REPORT=ROOT/"data/derived/resnet18_oversampling_v1"
SUMMARY=REPORT/"resnet18_oversampling_summary.json"
MANIFEST=REPORT/"oversampling_manifest.csv"
ORDER=["p","o","s","sc"]
FACTORS={"p":1,"o":10,"s":5,"sc":2}
EFFECTIVE={"p":4352,"o":20,"s":25,"sc":798}
OriginalCropDataset=common.CropDataset


def read_manifest():
    with MANIFEST.open(encoding="utf-8-sig",newline="") as file:
        return list(csv.DictReader(file))


class OversampledCropDataset(OriginalCropDataset):
    """Use saved explicit Train exposure indices; preserve Val/Test rows exactly."""
    def __init__(self,rows,split,transform):
        super().__init__(rows,split,transform)
        if split!="train":
            return
        by_path={row["crop_relative_path"]:row for row in self.rows}
        indices=read_manifest()
        assert len(by_path)==4758 and len(indices)==5195
        expected=[(row["crop_relative_path"],repeat)
                  for row in self.rows for repeat in range(1,FACTORS[row["class"]]+1)]
        assert [(row["source_crop"],int(row["repeat_index"])) for row in indices]==expected
        for row in indices:
            source=by_path[row["source_crop"]]
            assert row["split"]=="train" and row["class"]==source["class"]
            assert int(row["repeat_factor"])==FACTORS[source["class"]]
            assert row["crop_sha256"]==source["crop_file_sha256"]
        self.rows=[by_path[row["source_crop"]] for row in indices]
        assert Counter(row["class"] for row in self.rows)==EFFECTIVE


def set_output_paths():
    # In-memory experiment destinations/loader only; existing source files are not edited.
    baseline.RUN=RUN
    baseline.REPORT=REPORT
    baseline.SUMMARY=SUMMARY
    baseline.SCRIPT=SCRIPT


def prepare():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("Never overwrite an existing oversampling experiment")
    old=json.loads((BASE_REPORT/"resnet18_baseline_summary.json").read_text(encoding="utf-8"))
    base_config=json.loads((BASE_RUN/"training_config.json").read_text(encoding="utf-8"))
    assert old["status"]=="COMPLETED" and old["training_success"] and old["test_evaluation_calls"]==1
    assert base_config==old["config"]
    assert base_config==json.loads((BASE_REPORT/"training_config.json").read_text(encoding="utf-8"))
    assert common.digest(Path(baseline.__file__))==old["script_sha256"]
    assert common.digest(Path(common.__file__))==old["helper_sha256"]
    assert common.digest(BASE_RUN/"best.pt")==old["best_checkpoint_sha256"]
    protected=dict(old["protected_input_sha256"])
    for folder in [BASE_RUN,BASE_REPORT]:
        for path in folder.rglob("*"):
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()]=common.digest(path)
    for path in [Path(baseline.__file__).resolve(),Path(common.__file__).resolve()]:
        protected[path.relative_to(ROOT).as_posix()]=common.digest(path)
    baseline.verify_protected(protected)
    rows=common.read_rows()
    assert len(rows)==5953 and len({r["crop_filename"] for r in rows})==5953
    assert len({r["annotation_id"] for r in rows})==5953
    with (ROOT/"data/derived/fixed_split_v1/split_manifest.csv").open(encoding="utf-8-sig") as file:
        fixed={r["image_id"]:r for r in csv.DictReader(file)}
    assert Counter(r["split"] for r in fixed.values())=={"train":505,"val":62,"test":62}
    source_splits=defaultdict(set)
    group_splits=defaultdict(set)
    all_paths=set()
    exposures=[]
    for index,row in enumerate(rows,1):
        assert row["class"] in ORDER and row["split"] in common.EXPECTED
        assignment=fixed[row["source_image_id"]]
        assert assignment["split"]==row["split"] and assignment["filename"]==row["source_filename"]
        assert assignment["group_key"]==row["capture_group"]
        source_splits[row["source_image_id"]].add(row["split"])
        group_splits[row["capture_group"]].add(row["split"])
        path=(DATA/row["crop_relative_path"]).resolve()
        assert path.parent==DATA.resolve()/row["split"]/row["class"]
        assert path.is_file() and path.stat().st_size>0
        assert common.digest(path)==row["crop_file_sha256"]
        with common.Image.open(path) as image:
            image.load()
            assert image.mode=="RGB" and image.size==(int(row["crop_width"]),int(row["crop_height"]))
        all_paths.add(path)
        if row["split"]=="train":
            factor=FACTORS[row["class"]]
            for repeat in range(1,factor+1):
                exposures.append({
                    "effective_sample_index":len(exposures),"source_crop":row["crop_relative_path"],
                    "class":row["class"],"repeat_factor":factor,"repeat_index":repeat,"split":"train",
                    "crop_filename":row["crop_filename"],"source_filename":row["source_filename"],
                    "source_image_id":row["source_image_id"],"annotation_id":row["annotation_id"],
                    "crop_sha256":row["crop_file_sha256"],
                })
        if index%1500==0:
            print(f"RESNET18_OS_CROP_QA={index}/5953",flush=True)
    assert all(len(splits)==1 for splits in source_splits.values())
    assert all(len(splits)==1 for splits in group_splits.values())
    assert all_paths=={p.resolve() for split in common.EXPECTED for name in ORDER
                      for p in (DATA/split/name).glob("*.png")}
    for split in common.EXPECTED:
        assert Counter(r["class"] for r in rows if r["split"]==split)==Counter(common.EXPECTED[split])
    assert len(exposures)==5195 and Counter(r["class"] for r in exposures)==EFFECTIVE
    per_source=Counter(r["source_crop"] for r in exposures)
    assert len(per_source)==4758
    assert all(per_source[r["source_crop"]]==r["repeat_factor"] for r in exposures)
    with (BASE_REPORT/"test_predictions.csv").open(encoding="utf-8-sig",newline="") as file:
        baseline_predictions=list(csv.DictReader(file))
    assert [(r["crop_filename"],r["class"]) for r in rows if r["split"]=="test"]==[
        (r["crop_filename"],r["true_class"]) for r in baseline_predictions]
    config=json.loads(json.dumps(base_config))
    assert config["requested_weights_enum"]=="ResNet18_Weights.DEFAULT"
    assert config["weights_enum"]=="ResNet18_Weights.IMAGENET1K_V1"
    assert common.digest(baseline.WEIGHT)==config["pretrained_sha256"]
    assert str(baseline.WEIGHTS)==config["weights_enum"]
    assert config["class_order"]==ORDER and config["normalization_mean"]==[.485,.456,.406]
    assert config["normalization_std"]==[.229,.224,.225]
    assert (config["seed"],config["batch_size"],config["max_epochs"],config["early_stopping_patience"])==(42,32,50,10)
    assert (config["optimizer"],config["learning_rate"],config["weight_decay"])==("AdamW",.001,.0001)
    assert config["class_weight"] is None and config["label_smoothing"]==0
    assert config["learning_rate_scheduler"] is None and config["num_workers"]==0
    assert config["deterministic_algorithms"] and not config["freeze_backbone"]
    assert not any(config[k] for k in ["augmentation","undersampling","weighted_sampler","synthetic","mixed_precision"])
    assert config["device"]=="cpu" and not common.torch.cuda.is_available()
    assert config["torch_num_threads"]==common.torch.get_num_threads()
    assert config["torch_num_interop_threads"]==common.torch.get_num_interop_threads()
    current={"python":platform.python_version(),"torch":str(common.torch.__version__),
             "torchvision":str(common.torchvision.__version__),"numpy":common.np.__version__}
    assert all(config["environment"][k]==v for k,v in current.items())
    config.update(
        oversampling=True,sample_duplication=True,
        script_path=str(SCRIPT),script_sha256=common.digest(SCRIPT),
        run_path=str(RUN),result_path=str(REPORT),oversampling_manifest_path=str(MANIFEST),
        training_helper_path=str(Path(baseline.__file__).resolve()),
        training_helper_sha256=common.digest(Path(baseline.__file__).resolve()),
        oversampling_implementation="Saved explicit manifest indices; ordinary RandomSampler without replacement over the expanded Train dataset.",
    )
    difference={k:{"baseline":base_config.get(k),"oversampling":v}
                for k,v in config.items() if v!=base_config.get(k)}
    allowed={"oversampling","sample_duplication","script_path","script_sha256","run_path","result_path",
             "oversampling_manifest_path","training_helper_path","training_helper_sha256","oversampling_implementation"}
    assert set(difference)==allowed
    assert all(config[k]==v for k,v in base_config.items() if k not in allowed)
    assert common.classification_metrics(common.np.diag([1,0,1,1]))["macro_f1"]==.75
    transform=common.preprocess(config)
    for split in ["val","test"]:
        original=OriginalCropDataset(rows,split,common.preprocess(base_config))
        unchanged=OversampledCropDataset(rows,split,transform)
        assert original.rows==unchanged.rows
    val_a=OriginalCropDataset(rows,"val",common.preprocess(base_config))[0][0]
    val_b=OversampledCropDataset(rows,"val",transform)[0][0]
    assert common.torch.equal(val_a,val_b)
    RUN.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    common.csv_write(MANIFEST,exposures)
    assert len(read_manifest())==5195
    expanded=OversampledCropDataset(rows,"train",transform)
    assert len(expanded)==5195 and Counter(r["class"] for r in expanded.rows)==EFFECTIVE
    manifest_sha=common.digest(MANIFEST)
    protected[MANIFEST.relative_to(ROOT).as_posix()]=manifest_sha
    common.json_write(RUN/"training_config.json",config)
    common.json_write(REPORT/"training_config.json",config)
    for path in [SCRIPT,Path(baseline.__file__).resolve(),Path(common.__file__).resolve()]:
        shutil.copy2(path,RUN/path.name)
    unique_factors=Counter(FACTORS[r["class"]] for r in rows if r["split"]=="train")
    effective_factors=Counter(r["repeat_factor"] for r in exposures)
    summary={
        "status":"PREPARED","dataset_preparation_success":True,"training_success":False,"error":None,
        "model":"resnet18","pretrained":True,"config":config,"script_path":str(SCRIPT),
        "script_sha256":common.digest(SCRIPT),"resnet_baseline_helper_sha256":common.digest(Path(baseline.__file__)),
        "data_helper_sha256":common.digest(Path(common.__file__)),
        "execution_command":f"{sys.executable} -X utf8 -B -u {SCRIPT} --train-prepared",
        "original_unique_train":4758,"effective_train":5195,"repeat_factors":FACTORS,
        "unique_train_class_counts":common.EXPECTED["train"],"effective_train_class_counts":EFFECTIVE,
        "sample_counts":{"train":5195,"val":597,"test":598},
        "unique_sample_counts":{"train":4758,"val":597,"test":598},
        "class_support":{"train":EFFECTIVE,"val":common.EXPECTED["val"],"test":common.EXPECTED["test"]},
        "repeat_factor_unique_crop_counts":dict(unique_factors),
        "repeat_factor_effective_sample_counts":dict(effective_factors),
        "oversampling_manifest_path":str(MANIFEST),"oversampling_manifest_sha256":manifest_sha,
        "baseline_config_comparison":{"same_training_conditions":True,"differences":difference,
                                      "only_experimental_change":"Train repeat exposure; other differences are traceability/output metadata."},
        "val_test_unchanged":True,"test_order_matches_baseline":True,"source_split_leakage":0,
        "protected_input_sha256":protected,"dataset_readable":5953,"unreadable_samples":0,
        "loader_exclusions":0,"nan_detected":False,"test_evaluation_calls":0,"warnings":[],
        "actual_epochs":0,"best_epoch":None,"best_val_macro_f1":None,
        "morphology_note":"Only repeated exposure to the same 4758 unique real Train crops, including 2 open and 5 short crops. No new morphology.",
        "started_utc":datetime.now(timezone.utc).isoformat(),
    }
    common.json_write(SUMMARY,summary)
    print("RESNET18_OS_PREPARED="+json.dumps({k:summary[k] for k in [
        "original_unique_train","effective_train","unique_train_class_counts","effective_train_class_counts",
        "repeat_factor_unique_crop_counts","repeat_factor_effective_sample_counts",
        "val_test_unchanged","source_split_leakage","baseline_config_comparison"]}),flush=True)
    return summary


def write_results(summary,history,predictions):
    test=summary["test"]
    old=json.loads((BASE_REPORT/"resnet18_baseline_summary.json").read_text(encoding="utf-8"))
    keys=["accuracy","macro_precision","macro_recall","macro_f1","weighted_f1"]
    comparison={
        "overall":{k:{"baseline":old["test"][k],"oversampling":test[k],"delta":test[k]-old["test"][k]} for k in keys},
        "per_class":{c:{k:{"baseline":old["test"]["per_class"][c][k],"oversampling":test["per_class"][c][k],
                           "delta":test["per_class"][c][k]-old["test"]["per_class"][c][k]}
                         for k in ["precision","recall","f1"]} for c in ORDER},
        "confusion_matrix_delta":(common.np.array(test["confusion_matrix"])-common.np.array(old["test"]["confusion_matrix"])).tolist(),
    }
    for key,y,x in [("sc_to_p",3,0),("p_to_sc",0,3)]:
        before,after=old["test"]["confusion_matrix"][y][x],test["confusion_matrix"][y][x]
        comparison[key]={"baseline":before,"oversampling":after,"delta":after-before}
    before=old["majority_class_evidence"]["predicted_particle_fraction"]
    after=summary["majority_class_evidence"]["predicted_particle_fraction"]
    comparison["p_prediction_ratio"]={"baseline":before,"oversampling":after,"delta":after-before}
    comparison["predicted_o_count"]={"baseline":old["test"]["per_class"]["o"]["predicted_count"],"oversampling":test["per_class"]["o"]["predicted_count"]}
    comparison["predicted_s_count"]={"baseline":old["test"]["per_class"]["s"]["predicted_count"],"oversampling":test["per_class"]["s"]["predicted_count"]}
    summary["resnet18_baseline_comparison"]=comparison
    common.csv_write(REPORT/"classification_report.csv",[{"class":c,**test["per_class"][c]} for c in ORDER])
    common.csv_write(REPORT/"confusion_matrix.csv",[
        {"true_class":c,**dict(zip(ORDER,row))} for c,row in zip(ORDER,test["confusion_matrix"])])
    common.csv_write(REPORT/"test_predictions.csv",predictions)
    common.REPORT=REPORT
    common.make_plots(summary,history)
    for name in ["loss_curve.png","accuracy_macro_f1_curve.png"]:
        shutil.copy2(REPORT/name,RUN/name)
    lines=[
        "# ResNet18 rare-class oversampling v1","",
        f"- Actual execution script: {SCRIPT}.",
        f"- Script SHA256: {summary['script_sha256']}.",
        "- Same official ResNet18_Weights.DEFAULT = IMAGENET1K_V1 and pretrained hash as the ResNet18 real-only baseline.",
        "- Same full-network fine-tuning, FC4 and class order [p,o,s,sc].",
        "- Original unique Train4758 -> effective Train5195; Val597 and Test598 unchanged.",
        "- Repeat factors p x1 / o x10 / s x5 / sc x2; effective p4352/o20/s25/sc798.",
        f"- The Train loader reads saved indices from {MANIFEST}. No crop image copies or modifications.",
        "- Same deterministic RGB / direct bilinear224x224 / tensor / ImageNet normalization.",
        "- No augmentation, class weighting, weighted sampler, focal loss, undersampling or synthetic data.",
        "- Same seed42, batch32, max50 epochs, patience10, AdamW lr1e-3 weight_decay1e-4, workers0, deterministic algorithms.",
        "- Unweighted CrossEntropyLoss, label_smoothing0, scheduler=None. Other config changes are output/traceability metadata only.",
        f"- Device {summary['config']['device']}; torch {summary['config']['environment']['torch']}; torchvision {summary['config']['environment']['torchvision']}.",
        f"- Actual epochs {summary['actual_epochs']}; best epoch {summary['best_epoch']}; best Val macro-F1 {summary['best_val_macro_f1']:.8f}.",
        "- Best=max Val macro-F1, strict improvement, earliest tie. Macro includes all4 labels with zero_division=0.",
        "- Val open support0 contributes F1=0; maximum four-class Val macro-F1=0.75.",
        "- Train metrics are online training-pass metrics. Test evaluated once after Val-based selection.","",
        "## Best-checkpoint results","",
        "| Split | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name,metrics in [("Val",summary["best_val"]),("Test",test)]:
        lines.append("| "+name+" | "+" | ".join(f"{metrics[k]:.8f}" for k in keys)+" |")
    lines+=["","## Test class metrics","",
            "| Class | Support | Predicted count | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|"]
    for c in ORDER:
        r=test["per_class"][c]
        lines.append(f"| {c} | {r['support']} | {r['predicted_count']} | {r['precision']:.8f} | {r['recall']:.8f} | {r['f1']:.8f} |")
    lines+=["","## Test confusion matrix","",
            "Rows GT, columns prediction; order p,o,s,sc.","",
            "| GT | p | o | s | sc |","|---|---:|---:|---:|---:|"]
    for c,row in zip(ORDER,test["confusion_matrix"]):
        lines.append("| "+c+" | "+" | ".join(map(str,row))+" |")
    lines+=["","## ResNet18 baseline comparison (oversampling minus baseline)","",
            "| Metric | Baseline | Oversampling | Delta |","|---|---:|---:|---:|"]
    for key,r in comparison["overall"].items():
        lines.append(f"| {key} | {r['baseline']:.8f} | {r['oversampling']:.8f} | {r['delta']:+.8f} |")
    for c in ORDER:
        for key,r in comparison["per_class"][c].items():
            lines.append(f"| {c} {key} | {r['baseline']:.8f} | {r['oversampling']:.8f} | {r['delta']:+.8f} |")
    lines+=["",f"- sc -> p: {comparison['sc_to_p']['baseline']} -> {comparison['sc_to_p']['oversampling']}.",
            f"- p -> sc: {comparison['p_to_sc']['baseline']} -> {comparison['p_to_sc']['oversampling']}.",
            f"- p prediction ratio: {before:.8f} -> {after:.8f}.",
            f"- Predicted open: {test['per_class']['o']['predicted_count']}; short: {test['per_class']['s']['predicted_count']}.",
            "","## Rare Test observations",""]
    for r in summary["rare_test_predictions"]:
        lines.append(f"- GT {r['true_class']} -> {r['predicted_class']}; confidence={r['confidence']:.8f}; {r['crop_filename']}.")
    lines+=["",
        "- Train open2 and short5 are the same unique real crops after repetition; oversampling adds no new morphology.",
        "- Test open1/short1 are individual cases, not generalization evidence. Val open0/short1 are also sparse.",
        "- Classification uses GT boxes; these metrics are not equivalent to whole-image detection AP.","",
        "## Reproducibility and QA","",
        f"- Best checkpoint: {RUN/'best.pt'}.",
        f"- Last checkpoint: {RUN/'last.pt'}.",
        f"- Training minutes: {summary['training_time_min']:.3f}.",
        f"- Script/config verification: {summary['script_and_config_verified']}.",
        f"- Protected dataset, fixed split, manifest and baseline unchanged: {summary['protected_inputs_unchanged']}.",
        f"- Test evaluation calls: {summary['test_evaluation_calls']}.",
        f"- NaN: {summary['nan_detected']}; unreadable: {summary['unreadable_samples']}; loader exclusions: {summary['loader_exclusions']}.",
        "- Warnings: "+("; ".join(summary["warnings"]) if summary["warnings"] else "none")+".",
        f"- Error: {summary['error']}.",""]
    (REPORT/"resnet18_oversampling_results.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only",action="store_true")
    mode.add_argument("--train-prepared",action="store_true")
    args=parser.parse_args()
    set_output_paths()
    if args.train_prepared:
        summary=json.loads(SUMMARY.read_text(encoding="utf-8"))
        assert summary["status"]=="PREPARED" and not (REPORT/"training.log").exists()
        assert summary["script_sha256"]==common.digest(SCRIPT)
        assert summary["resnet_baseline_helper_sha256"]==common.digest(Path(baseline.__file__))
        assert summary["data_helper_sha256"]==common.digest(Path(common.__file__))
        baseline.verify_protected(summary["protected_input_sha256"])
    else:
        summary=prepare()
    if args.prepare_only:
        return
    with (REPORT/"training.log").open("x",encoding="utf-8",buffering=1) as logfile:
        with contextlib.redirect_stdout(common.Tee(sys.stdout,logfile)),contextlib.redirect_stderr(common.Tee(sys.stderr,logfile)):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    common.CropDataset=OversampledCropDataset
                    try:
                        history,predictions=baseline.train(summary)
                    finally:
                        common.CropDataset=OriginalCropDataset
                    assert common.digest(MANIFEST)==summary["oversampling_manifest_sha256"]
                    summary["warnings"]=list(dict.fromkeys(str(w.message) for w in caught))
                    write_results(summary,history,predictions)
                    summary["warnings"]=list(dict.fromkeys(str(w.message) for w in caught))
                summary.update(status="COMPLETED",training_success=True,error=None)
                common.json_write(SUMMARY,summary)
                print("RESNET18_OS_COMPLETED="+json.dumps({k:summary[k] for k in
                    ["training_success","actual_epochs","best_epoch","best_val_macro_f1",
                     "test","resnet18_baseline_comparison","best_checkpoint","warnings"]}),flush=True)
            except Exception as error:
                summary.update(status="FAILED",training_success=False,error=repr(error),
                               nan_detected=isinstance(error,FloatingPointError))
                common.json_write(SUMMARY,summary)
                traceback.print_exc()
                raise


if __name__=="__main__":
    main()
