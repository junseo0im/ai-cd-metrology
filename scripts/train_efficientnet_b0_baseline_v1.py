"""One EfficientNet-B0 real-only GT-bbox baseline; no tuning or extra experiments."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import contextlib
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback
import warnings

import train_mobilenet_real_only_v1 as common
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights

torch,np,nn=common.torch,common.np,common.nn
ROOT=common.ROOT
DATA=common.DATA
SCRIPT=Path(__file__).resolve()
RUN=ROOT/"runs/efficientnet/efficientnet_b0_real_only_v1"
REPORT=ROOT/"data/derived/efficientnet_b0_baseline_v1"
SUMMARY=REPORT/"efficientnet_b0_baseline_summary.json"
ORDER=["p","o","s","sc"]
EXPECTED=common.EXPECTED
WEIGHTS=EfficientNet_B0_Weights.DEFAULT
WEIGHT=ROOT/"outputs/pretrained"/WEIGHTS.url.rsplit("/",1)[1]
BASELINES={
    "mobilenet":(ROOT/"data/derived/mobilenet_baseline_v1/mobilenet_baseline_summary.json",
                 ROOT/"runs/mobilenet/mobilenet_v3_small_real_only_v1"),
    "resnet18":(ROOT/"data/derived/resnet18_baseline_v1/resnet18_baseline_summary.json",
                ROOT/"runs/resnet/resnet18_real_only_v1"),
}


def verify_protected(protected):
    changed=[name for name,sha in protected.items()
             if not (ROOT/name).is_file() or common.digest(ROOT/name)!=sha]
    assert not changed,f"Protected inputs changed: {changed[:5]}"


def prepare():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("Never overwrite an existing EfficientNet experiment")
    saved={name:json.loads(path.read_text(encoding="utf-8")) for name,(path,_) in BASELINES.items()}
    for name,old in saved.items():
        assert old["status"]=="COMPLETED" and old["training_success"] and old["test_evaluation_calls"]==1
        assert old["config"]==json.loads((BASELINES[name][1]/"training_config.json").read_text(encoding="utf-8"))
        assert old["config"]==json.loads((BASELINES[name][0].parent/"training_config.json").read_text(encoding="utf-8"))
        assert old["class_support"]==EXPECTED and old["sample_counts"]=={"train":4758,"val":597,"test":598}
    assert common.digest(Path(common.__file__))==saved["mobilenet"]["script_sha256"]
    assert common.digest(ROOT/"scripts/train_resnet18_baseline_v1.py")==saved["resnet18"]["script_sha256"]
    protected={}
    for old in saved.values():
        protected.update(old["protected_input_sha256"])
    prior_folders=[
        "data/derived/yolov8_baseline_v1","data/derived/yolov8_oversampling_v1",
        "data/derived/mobilenet_baseline_v1","data/derived/mobilenet_oversampling_v1",
        "data/derived/resnet18_baseline_v1","data/derived/resnet18_oversampling_v1",
        "runs/baseline/yolov8n_real_only_v1","runs/oversampling/yolov8n_oversample_v1",
        "runs/mobilenet/mobilenet_v3_small_real_only_v1","runs/mobilenet/mobilenet_v3_small_oversample_v1",
        "runs/resnet/resnet18_real_only_v1","runs/resnet/resnet18_oversample_v1","git_release_v1",
    ]
    for folder in prior_folders:
        for path in (ROOT/folder).rglob("*"):
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()]=common.digest(path)
    for path in [Path(common.__file__).resolve(),ROOT/"scripts/train_resnet18_baseline_v1.py"]:
        protected[path.relative_to(ROOT).as_posix()]=common.digest(path)
    verify_protected(protected)
    ds=json.loads((DATA/"dataset_summary.json").read_text(encoding="utf-8"))
    assert ds["success"] and ds["status"]=="COMPLETED" and ds["total_crops"]==5953
    assert sum(ds["qa_error_counts"].values())==0
    rows=common.read_rows()
    assert len(rows)==5953 and len({r["crop_filename"] for r in rows})==5953
    assert len({r["annotation_id"] for r in rows})==5953
    with (ROOT/"data/derived/fixed_split_v1/split_manifest.csv").open(encoding="utf-8-sig",newline="") as file:
        fixed={r["image_id"]:r for r in csv.DictReader(file)}
    assert Counter(r["split"] for r in fixed.values())=={"train":505,"val":62,"test":62}
    source_splits=defaultdict(set)
    group_splits=defaultdict(set)
    all_paths=set()
    for index,row in enumerate(rows,1):
        assignment=fixed[row["source_image_id"]]
        assert row["source_filename"]==assignment["filename"] and row["split"]==assignment["split"]
        assert row["capture_group"]==assignment["group_key"]
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
        if index%1500==0:
            print(f"EFFICIENTNET_B0_CROP_QA={index}/5953",flush=True)
    assert all(len(x)==1 for x in source_splits.values()) and all(len(x)==1 for x in group_splits.values())
    assert all_paths=={p.resolve() for split in EXPECTED for name in ORDER for p in (DATA/split/name).glob("*.png")}
    for split in EXPECTED:
        assert Counter(r["class"] for r in rows if r["split"]==split)==Counter(EXPECTED[split])
    for name,(summary_path,_) in BASELINES.items():
        with (summary_path.parent/"test_predictions.csv").open(encoding="utf-8-sig",newline="") as file:
            predictions=list(csv.DictReader(file))
        assert [(r["crop_filename"],r["class"]) for r in rows if r["split"]=="test"]==[
            (r["crop_filename"],r["true_class"]) for r in predictions]
    if not WEIGHT.exists():
        print(f"EFFICIENTNET_B0_OFFICIAL_WEIGHT_DOWNLOAD={WEIGHTS.url}",flush=True)
        torch.hub.load_state_dict_from_url(WEIGHTS.url,model_dir=str(WEIGHT.parent),
                                          map_location="cpu",progress=False,check_hash=True,weights_only=True)
    assert WEIGHT.is_file() and common.digest(WEIGHT).startswith("7f5810bc")
    assert str(WEIGHTS)=="EfficientNet_B0_Weights.IMAGENET1K_V1"
    protected[WEIGHT.relative_to(ROOT).as_posix()]=common.digest(WEIGHT)
    base_config=saved["resnet18"]["config"]
    assert list(WEIGHTS.transforms().mean)==base_config["normalization_mean"]==[.485,.456,.406]
    assert list(WEIGHTS.transforms().std)==base_config["normalization_std"]==[.229,.224,.225]
    config=json.loads(json.dumps(base_config))
    config.update(
        model="efficientnet_b0",requested_weights_enum="EfficientNet_B0_Weights.DEFAULT",
        weights_enum=str(WEIGHTS),weights_url=WEIGHTS.url,pretrained_path=str(WEIGHT),
        pretrained_sha256=common.digest(WEIGHT),
        architecture_dropout="Torchvision EfficientNet-B0 defaults: dropout0.2 and stochastic_depth_prob0.2; seed-controlled model regularization, not image augmentation.",
        pretrained_num_parameters=WEIGHTS.meta["num_params"],
        dataset_path=str(DATA),dataset_manifest=str(DATA/"crop_manifest.csv"),
        script_path=str(SCRIPT),script_sha256=common.digest(SCRIPT),
        helper_path=str(Path(common.__file__).resolve()),helper_sha256=common.digest(Path(common.__file__)),
        run_path=str(RUN),result_path=str(REPORT),device="cuda:0" if torch.cuda.is_available() else "cpu",
        torch_num_threads=torch.get_num_threads(),torch_num_interop_threads=torch.get_num_interop_threads(),
        environment={"python":platform.python_version(),"torch":str(torch.__version__),
                     "torchvision":str(common.torchvision.__version__),"numpy":np.__version__,
                     "platform":platform.platform(),"cuda_available":torch.cuda.is_available()},
    )
    unchanged_keys=[
        "class_order","class_to_idx","seed","input_size","batch_size","max_epochs",
        "early_stopping_patience","best_metric","best_mode","best_tie_policy","macro_policy",
        "optimizer","learning_rate","weight_decay","optimizer_betas","optimizer_eps","amsgrad",
        "loss","class_weight","label_smoothing","learning_rate_scheduler","freeze_backbone",
        "num_workers","deterministic_algorithms","deterministic_warn_only","cudnn_benchmark",
        "cudnn_deterministic","preprocessing","normalization_mean","normalization_std",
        "train_shuffle","train_sampler","evaluation_sampler","drop_last","augmentation",
        "oversampling","undersampling","weighted_sampler","synthetic","sample_duplication",
        "mixed_precision","train_metric_policy","test_policy",
    ]
    assert all(config[k]==old["config"][k] for old in saved.values() for k in unchanged_keys)
    assert (config["seed"],config["batch_size"],config["max_epochs"],config["early_stopping_patience"])==(42,32,50,10)
    assert (config["optimizer"],config["learning_rate"],config["weight_decay"])==("AdamW",.001,.0001)
    assert config["class_weight"] is None and config["label_smoothing"]==0 and config["learning_rate_scheduler"] is None
    assert not any(config[k] for k in ["augmentation","oversampling","undersampling","weighted_sampler","synthetic","sample_duplication","freeze_backbone"])
    assert common.classification_metrics(np.diag([1,0,1,1]))["macro_f1"]==.75
    transform=common.preprocess(config)
    sample=common.CropDataset(rows,"val",transform)
    assert torch.equal(sample[0][0],sample[0][0]) and sample[0][0].shape==(3,224,224)
    assert torch.equal(sample[0][0],common.CropDataset(rows,"val",common.preprocess(base_config))[0][0])
    RUN.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    common.json_write(RUN/"training_config.json",config)
    common.json_write(REPORT/"training_config.json",config)
    for path in [SCRIPT,Path(common.__file__).resolve()]:
        shutil.copy2(path,RUN/path.name)
    summary={
        "status":"PREPARED","training_success":False,"dataset_preparation_success":True,"error":None,
        "model":"efficientnet_b0","pretrained":True,"config":config,
        "script_path":str(SCRIPT),"script_sha256":common.digest(SCRIPT),
        "helper_sha256":common.digest(Path(common.__file__)),
        "execution_command":f"{sys.executable} -X utf8 -B -u {SCRIPT} --train-prepared",
        "sample_counts":{split:sum(cc.values()) for split,cc in EXPECTED.items()},"class_support":EXPECTED,
        "baseline_config_comparison":{"unchanged_training_and_preprocessing_keys":unchanged_keys,
                                     "all_equal":True,"intentional_difference":"EfficientNet-B0 architecture and its official pretrained weights"},
        "val_test_unchanged":True,"test_order_matches_both_baselines":True,"source_split_leakage":0,
        "protected_input_sha256":protected,"dataset_readable":5953,"unreadable_samples":0,
        "loader_exclusions":0,"nan_detected":False,"test_evaluation_calls":0,"warnings":[],
        "actual_epochs":0,"best_epoch":None,"best_val_macro_f1":None,
        "started_utc":datetime.now(timezone.utc).isoformat(),
    }
    common.json_write(SUMMARY,summary)
    print("EFFICIENTNET_B0_PREPARED="+json.dumps({k:summary[k] for k in [
        "sample_counts","class_support","source_split_leakage","baseline_config_comparison"]}),flush=True)
    return summary


def pass_epoch(model,loader,loss_fn,device,optimizer=None,epoch=None,retain_predictions=False):
    training=optimizer is not None
    model.train(training)
    cm=np.zeros((4,4),dtype=np.int64)
    loss_sum=0.
    seen=[]
    predictions=[]
    with torch.enable_grad() if training else torch.inference_mode():
        for step,(images,target,indices) in enumerate(loader,1):
            images,target=images.to(device),target.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            loss=loss_fn(logits,target)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("NaN/Inf loss or logits")
            if training:
                loss.backward()
                optimizer.step()
            prediction=logits.detach().argmax(dim=1)
            true_np=target.cpu().numpy()
            pred_np=prediction.cpu().numpy()
            cm+=np.bincount(true_np*4+pred_np,minlength=16).reshape(4,4)
            loss_sum+=float(loss.detach().cpu())*len(indices)
            seen.extend(indices.tolist())
            if retain_predictions:
                confidence=logits.detach().softmax(dim=1).max(dim=1).values.cpu().tolist()
                for index,t,p,c in zip(indices.tolist(),true_np.tolist(),pred_np.tolist(),confidence):
                    row=loader.dataset.rows[index]
                    predictions.append({
                        "crop_filename":row["crop_filename"],"source_filename":row["source_filename"],
                        "true_class":ORDER[t],"predicted_class":ORDER[p],"confidence":float(c),
                        "correct":t==p,"annotation_id":row["annotation_id"],"source_image_id":row["source_image_id"],
                    })
            if training and (step%15==0 or step==len(loader)):
                progress={"status":"TRAINING","epoch":epoch,"batch":step,"batches":len(loader),
                          "train_loss_so_far":loss_sum/len(seen)}
                common.json_write(RUN/"progress.json",progress)
                print("EFFICIENTNET_B0_BATCH="+json.dumps(progress),flush=True)
    assert len(seen)==len(loader.dataset) and sorted(seen)==list(range(len(loader.dataset))),"Loader duplication/exclusion"
    metrics=common.classification_metrics(cm)
    metrics["loss"]=loss_sum/len(seen)
    return metrics,predictions


def train(summary):
    config=summary["config"]
    random.seed(config["seed"]);np.random.seed(config["seed"]);torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    device=torch.device(config["device"])
    model=efficientnet_b0(weights=None)
    model.load_state_dict(torch.load(WEIGHT,map_location="cpu",weights_only=True),strict=True)
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,4)
    assert all(p.requires_grad for p in model.parameters()) and model.classifier[-1].out_features==4
    model=model.to(device)
    summary["trainable_parameters"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary["total_parameters"]=sum(p.numel() for p in model.parameters())
    assert summary["trainable_parameters"]==summary["total_parameters"]
    print("EFFICIENTNET_B0_MODEL="+json.dumps({"parameters":summary["total_parameters"],
          "device":str(device),"weights":str(WEIGHTS),"class_order":ORDER}),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],
                               weight_decay=config["weight_decay"],betas=tuple(config["optimizer_betas"]),
                               eps=config["optimizer_eps"],amsgrad=config["amsgrad"])
    loss_fn=nn.CrossEntropyLoss(weight=None,label_smoothing=0.)
    rows=common.read_rows()
    transform=common.preprocess(config)
    train_set=common.CropDataset(rows,"train",transform)
    val_set=common.CropDataset(rows,"val",transform)
    generator=torch.Generator().manual_seed(config["seed"])
    train_loader=common.DataLoader(train_set,batch_size=config["batch_size"],shuffle=True,
                                   num_workers=0,drop_last=False,generator=generator)
    val_loader=common.DataLoader(val_set,batch_size=config["batch_size"],shuffle=False,num_workers=0,drop_last=False)
    assert isinstance(train_loader.sampler,common.RandomSampler) and not train_loader.sampler.replacement
    assert isinstance(val_loader.sampler,common.SequentialSampler)
    history=[]
    best=-math.inf
    stale=0
    started=time.perf_counter()
    summary["status"]="TRAINING"
    common.json_write(SUMMARY,summary)
    for epoch in range(1,config["max_epochs"]+1):
        epoch_start=time.perf_counter()
        train_metrics,_=pass_epoch(model,train_loader,loss_fn,device,optimizer,epoch)
        val_metrics,_=pass_epoch(model,val_loader,loss_fn,device)
        row={"epoch":epoch,"duration_sec":time.perf_counter()-epoch_start}
        for split,metrics in [("train",train_metrics),("val",val_metrics)]:
            for key in ["loss","accuracy","macro_precision","macro_recall","macro_f1","weighted_f1"]:
                row[f"{split}_{key}"]=metrics[key]
        history.append(row)
        common.csv_write(RUN/"training_history.csv",history)
        checkpoint={"state_dict":model.state_dict(),"epoch":epoch,"class_order":ORDER,
                    "config":config,"val_metrics":val_metrics,"script_sha256":summary["script_sha256"]}
        torch.save(checkpoint,RUN/"last.pt")
        if val_metrics["macro_f1"]>best:
            best=val_metrics["macro_f1"];stale=0
            summary.update(best_epoch=epoch,best_val_macro_f1=best,best_val=val_metrics)
            torch.save(checkpoint,RUN/"best.pt")
        else:
            stale+=1
        summary.update(actual_epochs=epoch,epochs_without_improvement=stale,
                       training_time_min=(time.perf_counter()-started)/60)
        common.json_write(SUMMARY,summary)
        progress={"status":"EPOCH_COMPLETED",**row,"best_epoch":summary["best_epoch"],
                  "best_val_macro_f1":best,"stale":stale}
        common.json_write(RUN/"progress.json",progress)
        print("EFFICIENTNET_B0_EPOCH="+json.dumps(progress),flush=True)
        if stale>=config["early_stopping_patience"]:
            summary["stop_reason"]="early_stopping"
            break
    else:
        summary["stop_reason"]="max_epochs"
    summary["training_time_min"]=(time.perf_counter()-started)/60
    checkpoint=torch.load(RUN/"best.pt",map_location="cpu",weights_only=True)
    assert checkpoint["epoch"]==summary["best_epoch"] and checkpoint["class_order"]==ORDER
    assert checkpoint["config"]==config and checkpoint["script_sha256"]==common.digest(SCRIPT)
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    model.to(device)
    # No Test model inference or selection occurs before this point.
    test_set=common.CropDataset(rows,"test",transform)
    test_loader=common.DataLoader(test_set,batch_size=config["batch_size"],shuffle=False,num_workers=0,drop_last=False)
    assert summary["test_evaluation_calls"]==0
    summary.update(status="FINAL_TEST_EVALUATION",test_evaluation_calls=1)
    common.json_write(SUMMARY,summary)
    test,predictions=pass_epoch(model,test_loader,loss_fn,device,retain_predictions=True)
    assert len(predictions)==598 and len({r["crop_filename"] for r in predictions})==598
    assert {c:test["per_class"][c]["support"] for c in ORDER}==EXPECTED["test"]
    summary["test"]=test
    summary["rare_test_predictions"]=[r for r in predictions if r["true_class"] in {"o","s"}]
    assert Counter(r["true_class"] for r in summary["rare_test_predictions"])=={"o":1,"s":1}
    cm=np.asarray(test["confusion_matrix"])
    summary["majority_class_evidence"]={
        "gt_particle_fraction":int(cm[0].sum())/598,"predicted_particle_fraction":int(cm[:,0].sum())/598,
        "non_particle_to_particle":int(cm[1:,0].sum()),"non_particle_support":int(cm[1:].sum()),
    }
    verify_protected(summary["protected_input_sha256"])
    assert common.digest(SCRIPT)==summary["script_sha256"]
    assert common.digest(RUN/SCRIPT.name)==summary["script_sha256"]
    assert json.loads((RUN/"training_config.json").read_text(encoding="utf-8"))==config
    summary.update(protected_inputs_unchanged=True,script_and_config_verified=True,
                   best_checkpoint=str(RUN/"best.pt"),last_checkpoint=str(RUN/"last.pt"),
                   best_checkpoint_sha256=common.digest(RUN/"best.pt"),
                   last_checkpoint_sha256=common.digest(RUN/"last.pt"),
                   completed_utc=datetime.now(timezone.utc).isoformat())
    return history,predictions


def write_results(summary,history,predictions):
    test=summary["test"]
    overall=["accuracy","macro_precision","macro_recall","macro_f1","weighted_f1"]
    comparisons={}
    for name,(path,_) in BASELINES.items():
        old=json.loads(path.read_text(encoding="utf-8"))
        previous=old["test"]
        comparison={
            "overall":{k:{"baseline":previous[k],"efficientnet_b0":test[k],"delta":test[k]-previous[k]} for k in overall},
            "per_class":{c:{k:{"baseline":previous["per_class"][c][k],
                              "efficientnet_b0":test["per_class"][c][k],
                              "delta":test["per_class"][c][k]-previous["per_class"][c][k]}
                            for k in ["precision","recall","f1"]} for c in ORDER},
            "confusion_matrix_delta":(np.array(test["confusion_matrix"])-np.array(previous["confusion_matrix"])).tolist(),
        }
        for key,y,x in [("sc_to_p",3,0),("p_to_sc",0,3)]:
            before,after=previous["confusion_matrix"][y][x],test["confusion_matrix"][y][x]
            comparison[key]={"baseline":before,"efficientnet_b0":after,"delta":after-before}
        before=old["majority_class_evidence"]["predicted_particle_fraction"]
        after=summary["majority_class_evidence"]["predicted_particle_fraction"]
        comparison["p_prediction_ratio"]={"baseline":before,"efficientnet_b0":after,"delta":after-before}
        for c in ["o","s"]:
            comparison[f"{c}_predicted_count"]={
                "baseline":previous["per_class"][c]["predicted_count"],
                "efficientnet_b0":test["per_class"][c]["predicted_count"],
            }
        comparisons[name]=comparison
    summary["baseline_comparisons"]=comparisons
    common.csv_write(REPORT/"classification_report.csv",[{"class":c,**test["per_class"][c]} for c in ORDER])
    common.csv_write(REPORT/"confusion_matrix.csv",[
        {"true_class":c,**dict(zip(ORDER,row))} for c,row in zip(ORDER,test["confusion_matrix"])])
    common.csv_write(REPORT/"test_predictions.csv",predictions)
    prior_report=common.REPORT
    common.REPORT=REPORT
    try:
        common.make_plots(summary,history)
    finally:
        common.REPORT=prior_report
    for name in ["loss_curve.png","accuracy_macro_f1_curve.png"]:
        shutil.copy2(REPORT/name,RUN/name)
    lines=[
        "# EfficientNet-B0 real-only GT-bbox classification baseline","",
        f"- Executed script: {SCRIPT}.",
        f"- Script SHA256: {summary['script_sha256']}.",
        f"- Official requested EfficientNet_B0_Weights.DEFAULT; resolved {summary['config']['weights_enum']}.",
        f"- Pretrained SHA256: {summary['config']['pretrained_sha256']}.",
        f"- Total/trainable parameters: {summary['total_parameters']} / {summary['trainable_parameters']}.",
        f"- torch {summary['config']['environment']['torch']}; torchvision {summary['config']['environment']['torchvision']}.",
        f"- Device {summary['config']['device']}; full fine-tuning; classifier4; class order [p,o,s,sc].",
        f"- Dataset: {DATA}; Train4758 / Val597 / Test598; fixed source split unchanged.",
        "- Original imbalance p4352/o2/s5/sc399 retained; no sampling, weighting, augmentation or synthetic data.",
        "- Same RGB / deterministic direct bilinear antialiased224x224 / tensor / ImageNet mean/std as existing baselines.",
        "- Weight-enum default bicubic/resize256/center-crop transform is not used; only its normalization is checked.",
        "- Official B0 dropout0.2/stochastic-depth0.2 are seed-controlled architecture regularization, not data augmentation.",
        "- Same seed42; batch32; max50 epochs; patience10; AdamW lr1e-3/wd1e-4; workers0; deterministic algorithms.",
        "- Unweighted CrossEntropyLoss; label_smoothing0; scheduler=None; no backbone freeze.",
        "- Best=max Val macro-F1; strict improvement/earliest tie; all4 classes incl support0; zero_division=0.",
        "- Val o support0 contributes F1=0; maximum four-class Val macro-F1 is0.75.",
        f"- Actual epochs {summary['actual_epochs']}; best epoch {summary['best_epoch']}; best Val macro-F1 {summary['best_val_macro_f1']:.8f}.",
        "- Train metrics are online optimization-pass metrics; Test evaluated once after Val-only selection.","",
        "## Best-checkpoint results","",
        "| Split | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name,metrics in [("Val",summary["best_val"]),("Test",test)]:
        lines.append("| "+name+" | "+" | ".join(f"{metrics[k]:.8f}" for k in overall)+" |")
    lines+=["","## Test class results","",
            "| Class | Support | Predicted count | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|"]
    for c in ORDER:
        r=test["per_class"][c]
        lines.append(f"| {c} | {r['support']} | {r['predicted_count']} | {r['precision']:.8f} | {r['recall']:.8f} | {r['f1']:.8f} |")
    lines+=["","## Confusion matrix","","Rows GT, columns predicted; order p,o,s,sc.","",
            "| GT | p | o | s | sc |","|---|---:|---:|---:|---:|"]
    for c,row in zip(ORDER,test["confusion_matrix"]):
        lines.append("| "+c+" | "+" | ".join(map(str,row))+" |")
    for name,comparison in comparisons.items():
        lines+=["",f"## Saved {name} real-only comparison (EfficientNet minus baseline)","",
                "| Metric | Baseline | EfficientNet-B0 | Delta |","|---|---:|---:|---:|"]
        for key,r in comparison["overall"].items():
            lines.append(f"| {key} | {r['baseline']:.8f} | {r['efficientnet_b0']:.8f} | {r['delta']:+.8f} |")
        for c in ORDER:
            for key,r in comparison["per_class"][c].items():
                lines.append(f"| {c} {key} | {r['baseline']:.8f} | {r['efficientnet_b0']:.8f} | {r['delta']:+.8f} |")
        lines+=["",f"- sc -> p: {comparison['sc_to_p']['baseline']} -> {comparison['sc_to_p']['efficientnet_b0']}.",
                f"- p -> sc: {comparison['p_to_sc']['baseline']} -> {comparison['p_to_sc']['efficientnet_b0']}.",
                f"- p prediction ratio: {comparison['p_prediction_ratio']['baseline']:.8f} -> {comparison['p_prediction_ratio']['efficientnet_b0']:.8f}.",
                f"- Predicted open: {comparison['o_predicted_count']['baseline']} -> {comparison['o_predicted_count']['efficientnet_b0']}; short: {comparison['s_predicted_count']['baseline']} -> {comparison['s_predicted_count']['efficientnet_b0']}."]
    lines+=["","## Rare Test cases",""]
    for r in summary["rare_test_predictions"]:
        lines.append(f"- GT {r['true_class']} -> {r['predicted_class']}; confidence={r['confidence']:.8f}; {r['crop_filename']}.")
    lines+=["","- Train o2/s5 and Test o1/s1 are too sparse for generalization claims. Architecture changes do not establish that data scarcity is solved.",
            "- GT-bbox classification is not whole-image detection; Accuracy/F1 are not directly comparable to detection mAP.","",
            "## Traceability and QA","",
            f"- Best checkpoint: {RUN/'best.pt'}.",
            f"- Last checkpoint: {RUN/'last.pt'}.",
            f"- Training minutes: {summary['training_time_min']:.3f}.",
            f"- Script/config verification: {summary['script_and_config_verified']}.",
            f"- Dataset/fixed split/prior experiments/Git release unchanged: {summary['protected_inputs_unchanged']}.",
            f"- Test evaluations: {summary['test_evaluation_calls']}.",
            f"- NaN: {summary['nan_detected']}; unreadable: {summary['unreadable_samples']}; loader exclusions: {summary['loader_exclusions']}.",
            "- Warnings: "+("; ".join(summary["warnings"]) if summary["warnings"] else "none")+".",
            f"- Error: {summary['error']}.",""]
    (REPORT/"efficientnet_b0_baseline_results.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only",action="store_true")
    mode.add_argument("--train-prepared",action="store_true")
    args=parser.parse_args()
    # Match the recorded CPU thread settings, not a performance/parameter sweep.
    torch.set_num_threads(6)
    if torch.get_num_interop_threads()!=6:
        torch.set_num_interop_threads(6)
    if args.train_prepared:
        summary=json.loads(SUMMARY.read_text(encoding="utf-8"))
        assert summary["status"]=="PREPARED" and not (REPORT/"training.log").exists()
        assert summary["script_sha256"]==common.digest(SCRIPT)
        assert summary["helper_sha256"]==common.digest(Path(common.__file__))
        assert summary["config"]==json.loads((RUN/"training_config.json").read_text(encoding="utf-8"))
        verify_protected(summary["protected_input_sha256"])
    else:
        summary=prepare()
    if args.prepare_only:
        return
    with (REPORT/"training.log").open("x",encoding="utf-8",buffering=1) as logfile:
        with contextlib.redirect_stdout(common.Tee(sys.stdout,logfile)),contextlib.redirect_stderr(common.Tee(sys.stderr,logfile)):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    history,predictions=train(summary)
                    summary["warnings"]=list(dict.fromkeys(str(w.message) for w in caught))
                    write_results(summary,history,predictions)
                    summary["warnings"]=list(dict.fromkeys(str(w.message) for w in caught))
                summary.update(status="COMPLETED",training_success=True,error=None)
                common.json_write(SUMMARY,summary)
                print("EFFICIENTNET_B0_COMPLETED="+json.dumps({k:summary[k] for k in [
                    "training_success","actual_epochs","best_epoch","best_val_macro_f1","total_parameters",
                    "test","baseline_comparisons","best_checkpoint","warnings"]}),flush=True)
            except Exception as error:
                summary.update(status="FAILED",training_success=False,error=repr(error),
                               nan_detected=isinstance(error,FloatingPointError))
                common.json_write(SUMMARY,summary)
                traceback.print_exc()
                raise


if __name__=="__main__":
    main()



