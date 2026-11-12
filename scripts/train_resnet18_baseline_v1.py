"""One ResNet18 real-only GT-bbox classification baseline, without tuning."""
from __future__ import annotations

import argparse
from collections import Counter
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
from torchvision.models import resnet18,ResNet18_Weights

torch,np,nn=common.torch,common.np,common.nn
ROOT=common.ROOT
DATA=common.DATA
SCRIPT=Path(__file__).resolve()
RUN=ROOT/"runs/resnet/resnet18_real_only_v1"
REPORT=ROOT/"data/derived/resnet18_baseline_v1"
SUMMARY=REPORT/"resnet18_baseline_summary.json"
BASE_REPORT=ROOT/"data/derived/mobilenet_baseline_v1"
BASE_RUN=ROOT/"runs/mobilenet/mobilenet_v3_small_real_only_v1"
ORDER=["p","o","s","sc"]
EXPECTED=common.EXPECTED
WEIGHTS=ResNet18_Weights.DEFAULT
WEIGHT=ROOT/"outputs/pretrained"/WEIGHTS.url.rsplit("/",1)[1]


def verify_protected(protected):
    changed=[name for name,sha in protected.items()
             if not (ROOT/name).is_file() or common.digest(ROOT/name)!=sha]
    assert not changed,f"Protected inputs changed: {changed[:5]}"


def prepare():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("Do not overwrite an existing ResNet18 experiment")
    base=json.loads((BASE_REPORT/"mobilenet_baseline_summary.json").read_text(encoding="utf-8"))
    assert base["training_success"] and base["status"]=="COMPLETED"
    base_config=json.loads((BASE_REPORT/"training_config.json").read_text(encoding="utf-8"))
    assert base["config"]==base_config
    assert common.digest(Path(common.__file__))==base["script_sha256"]
    protected=dict(base["protected_input_sha256"])
    for folder in [BASE_REPORT,BASE_RUN]:
        for path in folder.rglob("*"):
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()]=common.digest(path)
    protected[Path(common.__file__).relative_to(ROOT).as_posix()]=common.digest(Path(common.__file__))
    verify_protected(protected)
    rows=common.read_rows()
    assert len(rows)==5953 and len({r["crop_filename"] for r in rows})==5953
    assert len({r["annotation_id"] for r in rows})==5953
    source_splits={}
    group_splits={}
    all_paths=set()
    for index,row in enumerate(rows,1):
        source_splits.setdefault(row["source_image_id"],set()).add(row["split"])
        group_splits.setdefault(row["capture_group"],set()).add(row["split"])
        path=(DATA/row["crop_relative_path"]).resolve()
        assert path.parent==DATA.resolve()/row["split"]/row["class"]
        assert common.digest(path)==row["crop_file_sha256"]
        with common.Image.open(path) as image:
            image.load()
            assert image.mode=="RGB" and image.size==(int(row["crop_width"]),int(row["crop_height"]))
        all_paths.add(path)
        if index%1500==0:
            print(f"RESNET18_CROP_QA={index}/5953",flush=True)
    assert all(len(splits)==1 for splits in source_splits.values())
    assert all(len(splits)==1 for splits in group_splits.values())
    assert all_paths=={p.resolve() for split in EXPECTED for name in ORDER
                      for p in (DATA/split/name).glob("*.png")}
    for split in EXPECTED:
        assert Counter(r["class"] for r in rows if r["split"]==split)==Counter(EXPECTED[split])
    with (BASE_REPORT/"test_predictions.csv").open(encoding="utf-8-sig",newline="") as file:
        base_pred=list(csv.DictReader(file))
    assert [(r["crop_filename"],r["class"]) for r in rows if r["split"]=="test"]==[
        (r["crop_filename"],r["true_class"]) for r in base_pred]
    if not WEIGHT.exists():
        print(f"RESNET18_OFFICIAL_WEIGHT_DOWNLOAD={WEIGHTS.url}",flush=True)
        torch.hub.load_state_dict_from_url(WEIGHTS.url,model_dir=str(WEIGHT.parent),
                                          map_location="cpu",progress=False,check_hash=True,weights_only=True)
    assert WEIGHT.is_file() and common.digest(WEIGHT).startswith("f37072fd")
    protected[WEIGHT.relative_to(ROOT).as_posix()]=common.digest(WEIGHT)
    norm=WEIGHTS.transforms()
    assert list(norm.mean)==base_config["normalization_mean"]
    assert list(norm.std)==base_config["normalization_std"]
    config=json.loads(json.dumps(base_config))
    config.update(
        model="resnet18",requested_weights_enum="ResNet18_Weights.DEFAULT",
        weights_enum=str(WEIGHTS),weights_url=WEIGHTS.url,pretrained_path=str(WEIGHT),
        pretrained_sha256=common.digest(WEIGHT),architecture_dropout="torchvision ResNet18 has no Dropout layer",
        dataset_path=str(DATA),dataset_manifest=str(DATA/"crop_manifest.csv"),
        script_path=str(SCRIPT),script_sha256=common.digest(SCRIPT),
        helper_path=str(Path(common.__file__).resolve()),helper_sha256=common.digest(Path(common.__file__)),
        run_path=str(RUN),result_path=str(REPORT),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
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
    assert all(config[k]==base_config[k] for k in unchanged_keys)
    assert (config["seed"],config["batch_size"],config["max_epochs"],config["early_stopping_patience"])==(42,32,50,10)
    assert (config["optimizer"],config["learning_rate"],config["weight_decay"])==("AdamW",.001,.0001)
    assert config["class_order"]==ORDER and config["class_weight"] is None
    assert not any(config[k] for k in ["augmentation","oversampling","undersampling",
                                      "weighted_sampler","synthetic","sample_duplication"])
    assert common.classification_metrics(np.eye(4,dtype=int))["macro_f1"]==1
    assert common.classification_metrics(np.diag([1,0,1,1]))["macro_f1"]==.75
    transform=common.preprocess(config)
    sample=common.CropDataset(rows,"train",transform)
    assert torch.equal(sample[0][0],sample[0][0]) and sample[0][0].shape==(3,224,224)
    RUN.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    common.json_write(RUN/"training_config.json",config)
    common.json_write(REPORT/"training_config.json",config)
    for path in [SCRIPT,Path(common.__file__).resolve()]:
        shutil.copy2(path,RUN/path.name)
    summary={
        "status":"PREPARED","training_success":False,"error":None,"model":"resnet18","pretrained":True,
        "config":config,"script_path":str(SCRIPT),"script_sha256":common.digest(SCRIPT),
        "helper_sha256":common.digest(Path(common.__file__)),"execution_command":f"{sys.executable} -X utf8 -B -u {SCRIPT} --train-prepared",
        "sample_counts":{s:sum(c.values()) for s,c in EXPECTED.items()},"class_support":EXPECTED,
        "baseline_config_comparison":{"unchanged_training_and_preprocessing_keys":unchanged_keys,
                                      "all_equal":True,"intentional_difference":"ResNet18 architecture and its official pretrained weights"},
        "val_test_unchanged":True,"test_order_matches_mobilenet":True,
        "protected_input_sha256":protected,"dataset_readable":5953,"unreadable_samples":0,
        "loader_exclusions":0,"nan_detected":False,"test_evaluation_calls":0,"warnings":[],
        "actual_epochs":0,"best_epoch":None,"best_val_macro_f1":None,
        "started_utc":datetime.now(timezone.utc).isoformat(),
    }
    common.json_write(SUMMARY,summary)
    print("RESNET18_PREPARED="+json.dumps({"sample_counts":summary["sample_counts"],
          "class_support":EXPECTED,"config":config,"same_training_conditions":True}),flush=True)
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
                print("RESNET18_BATCH="+json.dumps(progress),flush=True)
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
    model=resnet18(weights=None)
    model.load_state_dict(torch.load(WEIGHT,map_location="cpu",weights_only=True),strict=True)
    model.fc=nn.Linear(model.fc.in_features,4)
    assert all(p.requires_grad for p in model.parameters()) and model.fc.out_features==4
    model=model.to(device)
    summary["trainable_parameters"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
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
        print("RESNET18_EPOCH="+json.dumps(progress),flush=True)
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
    config=summary["config"]
    base=json.loads((BASE_REPORT/"mobilenet_baseline_summary.json").read_text(encoding="utf-8"))
    overall=["accuracy","macro_precision","macro_recall","macro_f1","weighted_f1"]
    summary["mobilenet_baseline_comparison"]={
        "overall":{k:{"mobilenet":base["test"][k],"resnet18":test[k],"delta":test[k]-base["test"][k]}
                   for k in overall},
        "per_class":{c:{k:{"mobilenet":base["test"]["per_class"][c][k],"resnet18":test["per_class"][c][k],
                            "delta":test["per_class"][c][k]-base["test"]["per_class"][c][k]}
                         for k in ["precision","recall","f1"]} for c in ORDER},
        "sc_to_p":{"mobilenet":base["test"]["confusion_matrix"][3][0],
                   "resnet18":test["confusion_matrix"][3][0],
                   "delta":test["confusion_matrix"][3][0]-base["test"]["confusion_matrix"][3][0]},
        "p_prediction_ratio":{"mobilenet":base["majority_class_evidence"]["predicted_particle_fraction"],
                              "resnet18":summary["majority_class_evidence"]["predicted_particle_fraction"],
                              "delta":summary["majority_class_evidence"]["predicted_particle_fraction"]-
                                      base["majority_class_evidence"]["predicted_particle_fraction"]},
        "confusion_matrix_delta":(np.array(test["confusion_matrix"])-np.array(base["test"]["confusion_matrix"])).tolist(),
    }
    common.csv_write(REPORT/"classification_report.csv",[
        {"class":name,**test["per_class"][name]} for name in ORDER])
    common.csv_write(REPORT/"confusion_matrix.csv",[
        {"true_class":name,**dict(zip(ORDER,row))} for name,row in zip(ORDER,test["confusion_matrix"])])
    common.csv_write(REPORT/"test_predictions.csv",predictions)
    common.REPORT=REPORT
    common.make_plots(summary,history)
    for name in ["loss_curve.png","accuracy_macro_f1_curve.png"]:
        shutil.copy2(REPORT/name,RUN/name)
    lines=[
        "# ResNet18 real-only GT-bbox classification baseline","",
        f"- Executed script: {SCRIPT}.",
        f"- Script SHA256: {summary['script_sha256']}.",
        f"- Model: ResNet18; requested ResNet18_Weights.DEFAULT; resolved {config['weights_enum']}.",
        f"- Pretrained SHA256: {config['pretrained_sha256']}.",
        f"- torch: {config['environment']['torch']}; torchvision: {config['environment']['torchvision']}.",
        f"- Device: {config['device']}; full-network fine-tuning; 4-class fc; class order [p,o,s,sc].",
        f"- Dataset: {DATA}; crops Train4758 / Val597 / Test598; source split unchanged.",
        "- RGB / direct bilinear224x224 / tensor / ImageNet normalization; no augmentation.",
        "- AdamW lr1e-3 weight_decay1e-4; unweighted CrossEntropyLoss; seed42; batch32; workers0; deterministic=True.",
        "- No resampling, sample duplication, synthetic data, focal loss or label smoothing.",
        "- Requested max50 epochs; patience10; maximize Val macro-F1; strict improvement, earliest tie.",
        f"- Actual epochs: {summary['actual_epochs']}; best epoch: {summary['best_epoch']}; best Val macro-F1: {summary['best_val_macro_f1']:.8f}.",
        "- Macro includes all four labels with zero_division=0. Val o support0 contributes0; maximum four-class Val macro-F1 is0.75.",
        "- Train metrics are online ordinary training-pass metrics; Val/Test use inference mode.",
        "- Test evaluated once, after Val-based model selection.","",
        "## Best-checkpoint results","",
        "| Split | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|---:|---:|"]
    for name,metrics in [("Val",summary["best_val"]),("Test",test)]:
        lines.append("| "+name+" | "+" | ".join(f"{metrics[k]:.8f}" for k in overall)+" |")
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
    lines+=["","## MobileNet real-only comparison (ResNet18 minus MobileNet)","",
            "| Metric | MobileNet | ResNet18 | Delta |","|---|---:|---:|---:|"]
    comp=summary["mobilenet_baseline_comparison"]
    for k,r in comp["overall"].items():
        lines.append(f"| {k} | {r['mobilenet']:.8f} | {r['resnet18']:.8f} | {r['delta']:+.8f} |")
    for c in ORDER:
        for k,r in comp["per_class"][c].items():
            lines.append(f"| {c} {k} | {r['mobilenet']:.8f} | {r['resnet18']:.8f} | {r['delta']:+.8f} |")
    lines+=["",
            f"- sc -> p: {comp['sc_to_p']['mobilenet']} -> {comp['sc_to_p']['resnet18']} (delta{comp['sc_to_p']['delta']:+d}).",
            f"- p prediction ratio: {comp['p_prediction_ratio']['mobilenet']:.8f} -> {comp['p_prediction_ratio']['resnet18']:.8f}.",
            f"- Predicted o count: {test['per_class']['o']['predicted_count']}; predicted s count: {test['per_class']['s']['predicted_count']}.",
            "","## Rare Test cases",""]
    for row in summary["rare_test_predictions"]:
        lines.append(f"- GT {row['true_class']} -> {row['predicted_class']}; confidence={row['confidence']:.8f}; {row['crop_filename']}.")
    lines+=["",
            "- Train o2/s5 and Test o1/s1 are very small samples; o/s metrics describe individual cases, not generalization performance.",
            "- Accuracy alone is insufficient because about91% of Test GT is particle.",
            "- GT-bbox classification metrics are not directly comparable to whole-image detection AP.","",
            "## Reproducibility and QA","",
            f"- Best checkpoint: {RUN/'best.pt'}.",
            f"- Last checkpoint: {RUN/'last.pt'}.",
            f"- Training minutes: {summary['training_time_min']:.3f}.",
            f"- Script/config/checkpoint verification: {summary['script_and_config_verified']}.",
            f"- Protected crop/split/MobileNet files unchanged: {summary['protected_inputs_unchanged']}.",
            f"- Test evaluation calls: {summary['test_evaluation_calls']}.",
            f"- NaN: {summary['nan_detected']}; unreadable samples: {summary['unreadable_samples']}; exclusions: {summary['loader_exclusions']}.",
            "- Warnings: "+("; ".join(summary["warnings"]) if summary["warnings"] else "none")+".",
            f"- Error: {summary['error']}.",""]
    (REPORT/"resnet18_baseline_results.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only",action="store_true")
    mode.add_argument("--train-prepared",action="store_true")
    args=parser.parse_args()
    if args.train_prepared:
        summary=json.loads(SUMMARY.read_text(encoding="utf-8"))
        assert summary["status"]=="PREPARED" and not (REPORT/"training.log").exists()
        assert summary["script_sha256"]==common.digest(SCRIPT)
        assert summary["helper_sha256"]==common.digest(Path(common.__file__))
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
                print("RESNET18_COMPLETED="+json.dumps({k:summary[k] for k in
                      ["training_success","actual_epochs","best_epoch","best_val_macro_f1",
                       "test","mobilenet_baseline_comparison","best_checkpoint","warnings"]}),flush=True)
            except Exception as error:
                summary.update(status="FAILED",training_success=False,error=repr(error),
                               nan_detected=isinstance(error,FloatingPointError))
                common.json_write(SUMMARY,summary)
                traceback.print_exc()
                raise


if __name__=="__main__":
    main()
