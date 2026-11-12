"""Sampler/loader integration checks; no training, real data, GPU or inference."""
from collections import Counter
import ast
import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/train_yolov8_weighted_sampling_v2_colab.py"
spec = importlib.util.spec_from_file_location("e4_sampling_script",SCRIPT)
e4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e4)


def population():
    return [{"filename":f"image_{i:03}.png","classes":[0,2] if i < 105 else [1],
             "rare_positive":i < 105,"weight":2.0 if i < 105 else 1.0} for i in range(604)]


@pytest.mark.parametrize("classes,weight",[
    ([],1.0),([1],1.0),([3],1.0),([1,3],1.0),
    ([0],2.0),([2],2.0),([0,2],2.0),([0,1,2,3],2.0)])
def test_union_weight_not_multiplicative(classes,weight):
    assert e4.image_sampling_weight(classes) == weight


def test_seeded_replacement_sampling_and_relative_exposure():
    rows = population()
    config = e4.sampling_configuration(rows)
    assert config["num_samples_per_epoch"] == 604
    assert config["rare_positive_images"] == 105
    assert config["normal_images"] == 499
    assert config["both_o_and_s_weight"] == 2
    rng_before = torch.random.get_rng_state().clone()
    first = e4.sampler_diagnostic(rows,epochs=100)
    second = e4.sampler_diagnostic(rows,epochs=100)
    assert first == second
    assert torch.equal(rng_before,torch.random.get_rng_state())
    total = first["overall"]
    assert total["sample_count"] == 60400
    assert 1.9 < total["per_image_rare_to_normal_exposure_ratio"] < 2.1
    sampler = e4.make_sampler([r["weight"] for r in rows])
    sampler.set_epoch(0)
    draws = list(sampler)
    assert len(draws) == 604 and len(set(draws)) < 604
    sampler.set_epoch(1)
    assert draws != list(sampler)
    sampler.set_epoch(0)
    assert draws == list(sampler)


def dummy_dataset():
    class Dummy(torch.utils.data.Dataset):
        def __init__(self):
            self.im_files = [r["filename"] for r in population()]
            self.labels = [{"cls":torch.tensor(r["classes"]).reshape(-1,1)}
                           for r in population()]

        def __len__(self):
            return 604

        def __getitem__(self,index):
            return {"im_file":self.im_files[index],"img":torch.zeros(3,2,2,dtype=torch.uint8)}

        @staticmethod
        def collate_fn(batch):
            return {"im_file":[b["im_file"] for b in batch],
                    "img":torch.stack([b["img"] for b in batch])}
    return Dummy()


def test_real_infinite_loader_tracks_consumed_samples_across_epoch_reset(tmp_path):
    from ultralytics.data.build import InfiniteDataLoader
    from ultralytics.models.yolo.detect import DetectionTrainer
    dataset = dummy_dataset()
    baseline = InfiniteDataLoader(dataset,batch_size=8,num_workers=0,
                                  collate_fn=dataset.collate_fn,generator=torch.Generator().manual_seed(42))
    cls = e4.make_weighted_trainer(population())
    trainer = object.__new__(cls)
    trainer.args = SimpleNamespace(workers=0,rect=False,compile=False,multi_scale=0)
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    with patch.object(DetectionTrainer,"get_dataloader",return_value=baseline), \
         patch.object(e4,"loader_check",return_value={"excluded":0}):
        trainer.train_loader = trainer.get_dataloader("train",batch_size=8,rank=-1,mode="train")
    assert len(trainer.train_loader) == 76
    info = {}
    with patch.object(e4,"RUN",tmp_path):
        for epoch in range(2):
            trainer.epoch = epoch
            trainer.train_loader.sampler.set_epoch(epoch)
            trainer.train_loader.reset()
            trainer.sampling_counts.clear()
            sizes = []
            for batch in trainer.train_loader:
                sizes.append(len(batch["im_file"]))
                result = trainer.preprocess_batch(batch)
                assert result["img"].dtype == torch.float32
            assert sizes == [8]*75+[4]
            assert sum(trainer.sampling_counts.values()) == 604
            assert trainer.sampling_counts == Counter(trainer.train_loader.sampler.drawn_indices)
            e4.record_sampling_epoch(trainer,info)
        assert info["sampling"]["overall"]["sample_count"] == 1208
        assert info["sampling"]["completed_epochs"] == 2
        records = list(csv.DictReader((tmp_path/"sampling_image_counts.csv").open()))
        assert len(records) == 1208  # every population member including zero draws, for two epochs
        assert sum(int(r["observed_count"]) for r in records) == 1208
        with pytest.raises(RuntimeError,match="Duplicate epoch"):
            e4.record_sampling_epoch(trainer,info)
    trainer.train_loader.close()


def test_validation_loader_is_unchanged_and_test_loader_forbidden():
    from ultralytics.models.yolo.detect import DetectionTrainer
    cls = e4.make_weighted_trainer(population())
    trainer = object.__new__(cls)
    sentinel = object()
    with patch.object(DetectionTrainer,"get_dataloader",return_value=sentinel) as parent:
        assert trainer.get_dataloader("val",rank=-1,mode="val") is sentinel
        parent.assert_called_once()
        with pytest.raises(RuntimeError,match="forbids a Test loader"):
            trainer.get_dataloader("test",rank=-1,mode="test")


def test_no_model_or_loss_override_and_only_val_inference():
    cls = e4.make_weighted_trainer(population())
    assert set(k for k in cls.__dict__ if not k.startswith("__")) == {"get_dataloader","preprocess_batch"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and
             isinstance(n.func,ast.Attribute) and n.func.attr == "val"]
    assert len(calls) == 1  # reused for E4 and E1, both Val67
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                   n.func.attr == "predict" for n in ast.walk(tree))
    evaluator = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "evaluate_best_validation")
    settings = [n for n in ast.walk(evaluator) if isinstance(n,ast.Call) and
                isinstance(n.func,ast.Name) and n.func.id == "dict"]
    assert any(any(k.arg == "split" and isinstance(k.value,ast.Constant) and
                   k.value.value == "val" for k in call.keywords) for call in settings)


def test_sampling_population_is_read_from_labels_not_annotation_frequency(tmp_path):
    import json
    folder = tmp_path/"dataset"
    labels = folder/"train/labels"
    labels.mkdir(parents=True)
    with (folder/"image_manifest.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=["filename","split"])
        writer.writeheader()
        for i in range(604):
            filename = f"image_{i:03}.png"
            writer.writerow({"filename":filename,"split":"train"})
            text = "0 0.5 0.5 0.1 0.1\n2 0.5 0.5 0.1 0.1\n" if i == 0 else (
                "1 0.5 0.5 0.1 0.1\n"*10 if i < 603 else "")
            (labels/Path(filename).with_suffix(".txt")).write_text(text,encoding="ascii")
    rows = e4.read_sampling_population(folder)
    assert rows[0]["weight"] == 2
    assert all(r["weight"] == 1 for r in rows[1:])
    assert e4.sampling_configuration(rows)["both_positive_images"] == 1
