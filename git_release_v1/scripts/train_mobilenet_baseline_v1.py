# GitHub release copy.
# Training logic and experiment settings are preserved from the executed experiment.
# Only local absolute paths were replaced with portable repository-relative/CLI paths.
# Helper-module filename references use the six public release filenames.
"""One MobileNetV3-Small real-only GT-bbox classification baseline; no tuning."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import contextlib
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback
import warnings

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler
import torchvision
from torchvision import transforms
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/derived/mobilenet_bbox_crops_v1"
REPORT=ROOT/"data/derived/mobilenet_baseline_v1"
RUN=ROOT/"runs/mobilenet/mobilenet_v3_small_real_only_v1"
SUMMARY=REPORT/"mobilenet_baseline_summary.json"
WEIGHT=ROOT/"outputs/pretrained/mobilenet_v3_small-047dcff4.pth"
CLASS_ORDER=["p","o","s","sc"]
EXPECTED={
    "train":{"p":4352,"o":2,"s":5,"sc":399},
    "val":{"p":546,"o":0,"s":1,"sc":50},
    "test":{"p":546,"o":1,"s":1,"sc":50},
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def json_write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def csv_write(path,rows,fields=None):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows():
    with (DATA/"crop_manifest.csv").open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def classification_metrics(cm):
    cm=np.asarray(cm,dtype=np.int64)
    assert cm.shape==(4,4) and (cm>=0).all()
    support=cm.sum(axis=1)
    predicted=cm.sum(axis=0)
    tp=cm.diagonal().astype(float)
    precision=np.divide(tp,predicted,out=np.zeros(4,dtype=float),where=predicted!=0)
    recall=np.divide(tp,support,out=np.zeros(4,dtype=float),where=support!=0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(4,dtype=float),
                 where=(precision+recall)!=0)
    total=int(support.sum())
    assert total>0
    return {
        "accuracy":float(tp.sum()/total),
        "macro_precision":float(precision.mean()),"macro_recall":float(recall.mean()),
        "macro_f1":float(f1.mean()),"weighted_f1":float((f1*support).sum()/total),
        "per_class":{name:{"support":int(support[i]),"predicted_count":int(predicted[i]),
                     "precision":float(precision[i]),"recall":float(recall[i]),"f1":float(f1[i])}
                     for i,name in enumerate(CLASS_ORDER)},
        "confusion_matrix":cm.tolist(),"samples":total,
    }


def preprocess(config):
    return transforms.Compose([
        transforms.Lambda(lambda im:im.convert("RGB")),
        transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["normalization_mean"],std=config["normalization_std"]),
    ])


class CropDataset(Dataset):
    def __init__(self,rows,split,transform):
        self.rows=[r for r in rows if r["split"]==split]
        self.transform=transform
        self.class_to_idx={name:i for i,name in enumerate(CLASS_ORDER)}
        assert Counter(r["class"] for r in self.rows)==Counter(EXPECTED[split])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(DATA/row["crop_relative_path"]) as im:
            image=self.transform(im)
        return image,self.class_to_idx[row["class"]],index


def preflight():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("Never overwrite an existing MobileNet experiment")
    ds=json.loads((DATA/"dataset_summary.json").read_text(encoding="utf-8"))
    assert ds["success"] and ds["status"]=="COMPLETED"
    assert ds["total_crops"]==5953 and sum(ds["qa_error_counts"].values())==0
    assert WEIGHT.is_file() and digest(WEIGHT).startswith("047dcff4")
    rows=read_rows()
    assert len(rows)==5953 and len({r["crop_filename"] for r in rows})==5953
    assert len({r["annotation_id"] for r in rows})==5953
    with (ROOT/"data/derived/fixed_split_v1/split_manifest.csv").open(encoding="utf-8-sig") as f:
        fixed={r["image_id"]:r for r in csv.DictReader(f)}
    assert Counter(r["split"] for r in fixed.values())==Counter({"train":505,"val":62,"test":62})
    source_splits=defaultdict(set)
    group_splits=defaultdict(set)
    all_paths=set()
    for i,r in enumerate(rows):
        f=fixed[r["source_image_id"]]
        assert r["source_filename"]==f["filename"] and r["split"]==f["split"]
        assert r["capture_group"]==f["group_key"]
        source_splits[r["source_image_id"]].add(r["split"])
        group_splits[r["capture_group"]].add(r["split"])
        p=(DATA/r["crop_relative_path"]).resolve()
        assert p.is_relative_to(DATA.resolve()) and p.parent==DATA.resolve()/r["split"]/r["class"]
        assert p.is_file() and p.stat().st_size>0 and digest(p)==r["crop_file_sha256"]
        with Image.open(p) as im:
            im.load()
            assert im.mode=="RGB" and im.size==(int(r["crop_width"]),int(r["crop_height"]))
        all_paths.add(p)
        if (i+1)%1500==0:
            print(f"CROP_PREFLIGHT={i+1}/5953",flush=True)
    assert all(len(v)==1 for v in source_splits.values())
    assert all(len(v)==1 for v in group_splits.values())
    assert {p.resolve() for split in EXPECTED for name in CLASS_ORDER
            for p in (DATA/split/name).glob("*.png")}==all_paths
    for split in EXPECTED:
        assert Counter(r["class"] for r in rows if r["split"]==split)==Counter(EXPECTED[split])
    protected={p.relative_to(ROOT).as_posix():digest(p) for p in DATA.rglob("*") if p.is_file()}
    for p in (ROOT/"data/derived/fixed_split_v1/split_manifest.csv",WEIGHT):
        protected[p.relative_to(ROOT).as_posix()]=digest(p)
    weight_enum=MobileNet_V3_Small_Weights.IMAGENET1K_V1
    normalization=weight_enum.transforms()
    config={
        "model":"mobilenet_v3_small","weights_enum":str(weight_enum),
        "pretrained":True,"weights_url":weight_enum.url,"pretrained_path":str(WEIGHT),
        "pretrained_sha256":digest(WEIGHT),"class_order":CLASS_ORDER,
        "class_to_idx":{n:i for i,n in enumerate(CLASS_ORDER)},
        "seed":42,"input_size":[224,224],"batch_size":32,"max_epochs":50,
        "early_stopping_patience":10,"best_metric":"val_macro_f1",
        "best_mode":"maximize","best_tie_policy":"keep earliest; strict improvement only",
        "macro_policy":"All fixed4 labels [p,o,s,sc], including support0 classes with zero_division=0; Val open contributes0. No supported-class-only macro.",
        "optimizer":"AdamW","learning_rate":1e-3,"weight_decay":1e-4,
        "optimizer_betas":[0.9,0.999],"optimizer_eps":1e-8,"amsgrad":False,
        "loss":"CrossEntropyLoss","class_weight":None,"label_smoothing":0.0,
        "learning_rate_scheduler":None,"freeze_backbone":False,"num_workers":0,
        "device":"cuda:0" if torch.cuda.is_available() else "cpu",
        "torch_num_threads":torch.get_num_threads(),
        "torch_num_interop_threads":torch.get_num_interop_threads(),
        "deterministic_algorithms":True,"deterministic_warn_only":False,
        "cudnn_benchmark":False,"cudnn_deterministic":True,
        "preprocessing":"RGB -> deterministic bilinear antialiased Resize((224,224)) -> ToTensor -> ImageNet Normalize; direct square resize, no random crop or center crop.",
        "normalization_mean":list(normalization.mean),"normalization_std":list(normalization.std),
        "train_shuffle":True,"train_sampler":"RandomSampler without replacement",
        "evaluation_sampler":"SequentialSampler","drop_last":False,
        "augmentation":False,"oversampling":False,"undersampling":False,
        "weighted_sampler":False,"synthetic":False,"sample_duplication":False,
        "mixed_precision":False,"architecture_dropout":"torchvision MobileNetV3-Small default0.2; seed-controlled model dropout, not image augmentation",
        "train_metric_policy":"Online training-mode predictions during the ordinary optimization pass, not an extra train eval pass.",
        "test_policy":"Best model selected using Val only; exactly one final Test inference pass.",
        "environment":{"python":platform.python_version(),"torch":str(torch.__version__),
                       "torchvision":str(torchvision.__version__),"numpy":np.__version__,
                       "platform":platform.platform(),"cuda_available":torch.cuda.is_available()},
    }
    # Metric boundary checks, no training/test samples or model evaluation involved.
    assert classification_metrics(np.eye(4,dtype=int))["macro_f1"]==1.0
    no_open=np.diag([1,0,1,1])
    assert classification_metrics(no_open)["macro_f1"]==0.75
    transform=preprocess(config)
    sample=CropDataset(rows,"train",transform)
    assert torch.equal(sample[0][0],sample[0][0])
    assert sample[0][0].shape==(3,224,224) and torch.isfinite(sample[0][0]).all()
    RUN.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    json_write(RUN/"training_config.json",config)
    json_write(REPORT/"training_config.json",config)
    shutil.copy2(Path(__file__).resolve(),RUN/Path(__file__).name)
    summary={
        "status":"PREPARED","training_success":False,"error":None,
        "model":config["model"],"pretrained":True,"config":config,
        "started_utc":datetime.now(timezone.utc).isoformat(),
        "sample_counts":{s:sum(v.values()) for s,v in EXPECTED.items()},
        "class_support":EXPECTED,"protected_input_sha256":protected,
        "dataset_readable":5953,"unreadable_samples":0,"loader_exclusions":0,
        "actual_epochs":0,"best_epoch":None,"best_val_macro_f1":None,
        "nan_detected":False,"test_evaluation_calls":0,"warnings":[],
        "script_sha256":digest(Path(__file__).resolve()),
        "classification_scope":"GT-bbox conditioned classification, not whole-image defect detection; metrics not directly comparable to YOLO detection AP.",
    }
    json_write(SUMMARY,summary)
    print("MOBILENET_PREPARED="+json.dumps({"counts":summary["sample_counts"],"config":config}),flush=True)
    return summary


class Tee:
    def __init__(self,stream,log):
        self.stream,self.log=stream,log
    def write(self,value):
        self.stream.write(value);self.log.write(value);self.log.flush()
        return len(value)
    def flush(self):
        self.stream.flush();self.log.flush()
    def isatty(self):
        return False


def pass_epoch(model,loader,loss_fn,device,optimizer=None,epoch=None,retain_predictions=False):
    training=optimizer is not None
    model.train(training)
    cm=np.zeros((4,4),dtype=np.int64)
    loss_sum=0.0
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
                raise FloatingPointError("NaN/Inf logits or loss")
            if training:
                loss.backward()
                optimizer.step()
            pred=logits.detach().argmax(dim=1)
            true_np=target.cpu().numpy()
            pred_np=pred.cpu().numpy()
            cm+=np.bincount(true_np*4+pred_np,minlength=16).reshape(4,4)
            loss_sum+=float(loss.detach().cpu())*len(indices)
            seen.extend(indices.tolist())
            if retain_predictions:
                confidence=logits.detach().softmax(dim=1).max(dim=1).values.cpu().tolist()
                for index,t,p,c in zip(indices.tolist(),true_np.tolist(),pred_np.tolist(),confidence):
                    row=loader.dataset.rows[index]
                    predictions.append({
                        "crop_filename":row["crop_filename"],"source_filename":row["source_filename"],
                        "true_class":CLASS_ORDER[t],"predicted_class":CLASS_ORDER[p],
                        "confidence":float(c),"correct":t==p,"annotation_id":row["annotation_id"],
                        "source_image_id":row["source_image_id"],
                    })
            if training and (step%30==0 or step==len(loader)):
                progress={"status":"TRAINING","epoch":epoch,"batch":step,"batches":len(loader),
                          "mean_train_loss_so_far":loss_sum/len(seen)}
                json_write(RUN/"progress.json",progress)
                print("MOBILENET_BATCH="+json.dumps(progress),flush=True)
    assert len(seen)==len(loader.dataset) and sorted(seen)==list(range(len(loader.dataset))),"Loader duplication/exclusion"
    result=classification_metrics(cm)
    result["loss"]=loss_sum/len(seen)
    return result,predictions


def make_plots(summary,history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs=[row["epoch"] for row in history]
    for name,keys,title in [
        ("loss_curve.png",["loss"],"Cross-entropy loss"),
        ("accuracy_macro_f1_curve.png",["accuracy","macro_f1"],"Accuracy and macro-F1"),
    ]:
        fig,axes=plt.subplots(1,len(keys),figsize=(7*len(keys),4.5),squeeze=False)
        for ax,key in zip(axes[0],keys):
            for split in ["train","val"]:
                ax.plot(epochs,[row[f"{split}_{key}"] for row in history],label=split)
            ax.axvline(summary["best_epoch"],color="gray",ls="--",label="selected epoch")
            ax.set(xlabel="Epoch",ylabel=key,title=title if len(keys)==1 else key)
            ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(REPORT/name,dpi=150);plt.close(fig)
    cm=np.asarray(summary["test"]["confusion_matrix"])
    fig,ax=plt.subplots(figsize=(6,5))
    shown=ax.imshow(cm,cmap="Blues")
    fig.colorbar(shown,ax=ax)
    ax.set(xticks=range(4),yticks=range(4),xticklabels=CLASS_ORDER,
           yticklabels=CLASS_ORDER,xlabel="Predicted class",ylabel="GT class",
           title="Test confusion matrix (GT bbox crops)")
    for y in range(4):
        for x in range(4):
            ax.text(x,y,str(cm[y,x]),ha="center",va="center",
                    color="white" if cm[y,x]>cm.max()/2 else "black")
    fig.tight_layout();fig.savefig(REPORT/"confusion_matrix.png",dpi=150);plt.close(fig)


def write_results(summary,history,predictions):
    test=summary["test"]
    csv_write(REPORT/"classification_report.csv",[
        {"class":name,**test["per_class"][name]} for name in CLASS_ORDER])
    csv_write(REPORT/"confusion_matrix.csv",[
        {"true_class":name,**dict(zip(CLASS_ORDER,row))}
        for name,row in zip(CLASS_ORDER,test["confusion_matrix"])])
    csv_write(REPORT/"test_predictions.csv",predictions)
    make_plots(summary,history)
    config=summary["config"]
    lines=[
        "# MobileNetV3-Small real-only GT-bbox classification baseline","",
        f"- Model: {config['model']}; official pretrained: {config['weights_enum']}.",
        f"- Class order: {', '.join(CLASS_ORDER)}.",
        "- Full-network fine-tuning; unweighted CrossEntropyLoss; no augmentation, resampling or synthetic data.",
        "- Preprocessing: RGB -> direct bilinear 224x224 resize -> tensor -> ImageNet normalization.",
        f"- AdamW lr={config['learning_rate']}, weight_decay={config['weight_decay']}; batch=32; seed=42; workers=0.",
        f"- Device: {config['device']}; torch {config['environment']['torch']}; torchvision {config['environment']['torchvision']}.",
        "- Crop samples Train/Val/Test: 4758 / 597 / 598 (source split is unchanged).",
        f"- Actual epochs: {summary['actual_epochs']}; best epoch: {summary['best_epoch']}.",
        f"- Best Val macro-F1: {summary['best_val_macro_f1']:.8f}.",
        "- Selection: maximize Val macro-F1, strict improvement, earliest tie, patience=10.",
        "- Macro averages include all four classes, zero_division=0; Val open support=0 therefore its F1=0 and maximum four-class Val macro-F1 is 0.75.",
        "- Train metrics are online training-pass metrics (including default model dropout); Val/Test are inference-mode metrics.",
        "- Test evaluated exactly once, after model selection, using best.pt.","",
        "## Best-checkpoint Validation and final Test","",
        "| Split | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split,metrics in [("Val",summary["best_val"]),("Test",test)]:
        lines.append("| "+split+" | "+" | ".join(f"{metrics[k]:.8f}" for k in
                     ["accuracy","macro_precision","macro_recall","macro_f1","weighted_f1"])+" |")
    lines+=["","## Test per-class metrics","",
            "| Class | Support | Predicted count | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|"]
    for name in CLASS_ORDER:
        item=test["per_class"][name]
        lines.append(f"| {name} | {item['support']} | {item['predicted_count']} | "
                     f"{item['precision']:.8f} | {item['recall']:.8f} | {item['f1']:.8f} |")
    lines+=["","## Test confusion matrix","",
            "Rows=GT; columns=prediction; order p, o, s, sc.","",
            "| GT | p | o | s | sc |","|---|---:|---:|---:|---:|"]
    for name,row in zip(CLASS_ORDER,test["confusion_matrix"]):
        lines.append("| "+name+" | "+" | ".join(map(str,row))+" |")
    lines+=["","## Rare-class predictions and majority-class evidence",""]
    for item in summary["rare_test_predictions"]:
        lines.append(f"- GT {item['true_class']} -> {item['predicted_class']}, confidence={item['confidence']:.6f}; {item['crop_filename']}.")
    evidence=summary["majority_class_evidence"]
    lines += [
        f"- GT particle fraction: {evidence['gt_particle_fraction']:.6f}; predicted particle fraction: {evidence['predicted_particle_fraction']:.6f}.",
        f"- Non-particle GT predicted as particle: {evidence['non_particle_to_particle']} / {evidence['non_particle_support']}.",
        f"- Particle-only constant classifier accuracy reference: {evidence['particle_only_accuracy_reference']:.6f} (computed from GT support, not a separate model run).",
        "- Val o=0, s=1; Test o=1, s=1. Rare-class metrics are numeric observations only, not generalization evidence.",
        "- Accuracy alone is not a sufficient performance claim because particle represents about 91% of evaluation crops.",
        "- This is classification conditioned on GT boxes, not whole-image detection; classification metrics must not be equated with YOLO detection AP.","",
        "## Traceability and QA","",
        f"- Best checkpoint: {RUN/'best.pt'}.",
        f"- Training wall time (minutes): {summary['training_time_min']:.3f}.",
        f"- Test evaluation calls: {summary['test_evaluation_calls']}.",
        f"- NaN/Inf detected: {summary['nan_detected']}; unreadable samples: {summary['unreadable_samples']}; loader exclusions: {summary['loader_exclusions']}.",
        f"- Protected crop dataset / fixed split / pretrained weight unchanged: {summary['protected_inputs_unchanged']}.",
        f"- Error: {summary['error']}.",
        "- Captured warnings: "+("; ".join(summary["warnings"]) if summary["warnings"] else "none")+".","",
    ]
    (REPORT/"mobilenet_baseline_results.md").write_text("\n".join(lines),encoding="utf-8")


def train(summary):
    config=summary["config"]
    seed=config["seed"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    device=torch.device(config["device"])
    model=mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(WEIGHT,map_location="cpu",weights_only=True),strict=True)
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,len(CLASS_ORDER))
    assert all(p.requires_grad for p in model.parameters())
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],
                               weight_decay=config["weight_decay"])
    loss_fn=nn.CrossEntropyLoss()
    rows=read_rows()
    transform=preprocess(config)
    train_set=CropDataset(rows,"train",transform)
    val_set=CropDataset(rows,"val",transform)
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train_set,batch_size=32,shuffle=True,num_workers=0,
                            drop_last=False,generator=generator)
    val_loader=DataLoader(val_set,batch_size=32,shuffle=False,num_workers=0,drop_last=False)
    assert isinstance(train_loader.sampler,RandomSampler) and not train_loader.sampler.replacement
    assert isinstance(val_loader.sampler,SequentialSampler)
    started=time.perf_counter()
    best=-math.inf
    stale=0
    history=[]
    summary["status"]="TRAINING"
    json_write(SUMMARY,summary)
    for epoch in range(1,config["max_epochs"]+1):
        epoch_start=time.perf_counter()
        train_metrics,_=pass_epoch(model,train_loader,loss_fn,device,optimizer,epoch)
        val_metrics,_=pass_epoch(model,val_loader,loss_fn,device)
        row={"epoch":epoch,"duration_sec":time.perf_counter()-epoch_start}
        for split,metrics in [("train",train_metrics),("val",val_metrics)]:
            for key in ["loss","accuracy","macro_precision","macro_recall","macro_f1","weighted_f1"]:
                row[f"{split}_{key}"]=metrics[key]
        history.append(row)
        csv_write(REPORT/"training_history.csv",history)
        if val_metrics["macro_f1"]>best:
            best=val_metrics["macro_f1"]
            stale=0
            summary.update(best_epoch=epoch,best_val_macro_f1=best,best_val=val_metrics)
            torch.save({
                "state_dict":model.state_dict(),"epoch":epoch,"class_order":CLASS_ORDER,
                "config":config,"val_metrics":val_metrics,
            },RUN/"best.pt")
        else:
            stale+=1
        summary.update(actual_epochs=epoch,epochs_without_improvement=stale,
                       training_time_min=(time.perf_counter()-started)/60)
        json_write(SUMMARY,summary)
        json_write(RUN/"progress.json",{"status":"EPOCH_COMPLETED",**row,"best_epoch":summary["best_epoch"],
                                      "best_val_macro_f1":best,"stale":stale})
        print("MOBILENET_EPOCH="+json.dumps({**row,"best_epoch":summary["best_epoch"],
                                             "best_val_macro_f1":best,"stale":stale}),flush=True)
        if stale>=config["early_stopping_patience"]:
            summary["stop_reason"]="early_stopping"
            break
    else:
        summary["stop_reason"]="max_epochs"
    summary["training_time_min"]=(time.perf_counter()-started)/60
    checkpoint=torch.load(RUN/"best.pt",map_location="cpu",weights_only=True)
    assert checkpoint["class_order"]==CLASS_ORDER and checkpoint["epoch"]==summary["best_epoch"]
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    model.to(device)
    # The held-out Test loader is constructed only after training/model selection is complete.
    test_set=CropDataset(rows,"test",transform)
    test_loader=DataLoader(test_set,batch_size=32,shuffle=False,num_workers=0,drop_last=False)
    assert summary["test_evaluation_calls"]==0
    summary["status"]="FINAL_TEST_EVALUATION"
    summary["test_evaluation_calls"]=1
    json_write(SUMMARY,summary)
    test,predictions=pass_epoch(model,test_loader,loss_fn,device,retain_predictions=True)
    assert len(predictions)==598 and len({r["crop_filename"] for r in predictions})==598
    assert {c:test["per_class"][c]["support"] for c in CLASS_ORDER}==EXPECTED["test"]
    summary["test"]=test
    summary["rare_test_predictions"]=[r for r in predictions if r["true_class"] in {"o","s"}]
    assert Counter(r["true_class"] for r in summary["rare_test_predictions"])=={"o":1,"s":1}
    cm=np.asarray(test["confusion_matrix"])
    summary["majority_class_evidence"]={
        "gt_particle_fraction":int(cm[0].sum())/598,
        "predicted_particle_fraction":int(cm[:,0].sum())/598,
        "non_particle_to_particle":int(cm[1:,0].sum()),
        "non_particle_support":int(cm[1:].sum()),
        "particle_only_accuracy_reference":546/598,
    }
    changed=[path for path,sha in summary["protected_input_sha256"].items()
             if not Path(path).is_file() or digest(Path(path))!=sha]
    assert not changed,f"Protected inputs changed: {changed[:5]}"
    summary.update(protected_inputs_unchanged=True,protected_changed_count=0,
                   best_checkpoint=str(RUN/"best.pt"),best_checkpoint_sha256=digest(RUN/"best.pt"),
                   completed_utc=datetime.now(timezone.utc).isoformat())
    return history,predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only",action="store_true")
    mode.add_argument("--train-prepared",action="store_true")
    args=parser.parse_args()
    if args.train_prepared:
        summary=json.loads(SUMMARY.read_text(encoding="utf-8"))
        assert summary["status"]=="PREPARED" and not (REPORT/"training.log").exists()
        assert summary["script_sha256"]==digest(Path(__file__).resolve())
        for path,sha in summary["protected_input_sha256"].items():
            assert digest(Path(path))==sha,f"Changed prepared input: {path}"
    else:
        summary=preflight()
    if args.prepare_only:
        return
    with (REPORT/"training.log").open("x",encoding="utf-8",buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout,logfile)),contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    history,predictions=train(summary)
                    summary["warnings"]=list(dict.fromkeys(str(w.message) for w in caught))
                    write_results(summary,history,predictions)
                    summary["warnings"]=list(dict.fromkeys(str(w.message) for w in caught))
                summary.update(status="COMPLETED",training_success=True,error=None)
                json_write(SUMMARY,summary)
                print("MOBILENET_COMPLETED="+json.dumps({
                    k:summary[k] for k in ["training_success","actual_epochs","best_epoch",
                                          "best_val_macro_f1","test","best_checkpoint","warnings"]}),flush=True)
            except Exception as error:
                summary.update(status="FAILED",training_success=False,error=repr(error),
                               nan_detected=isinstance(error,FloatingPointError))
                json_write(SUMMARY,summary)
                traceback.print_exc()
                raise


if __name__=="__main__":
    main()
