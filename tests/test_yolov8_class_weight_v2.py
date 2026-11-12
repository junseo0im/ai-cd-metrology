"""E5 tests use real Ultralytics loss tensors; no training or image inference."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/train_yolov8_class_weight_v2_colab.py"
spec = importlib.util.spec_from_file_location("e5_class_weight_script",SCRIPT)
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)


@pytest.fixture(scope="module")
def loss_check():
    return e5.loss_sanity_check()


@pytest.mark.parametrize("variant,weights,folder",[
    ("E5-A",[1.5,1.0,1.5,1.0],"e5a_weight_1p5"),
    ("E5-B",[2.0,1.0,2.0,1.0],"e5b_weight_2p0"),
])
def test_only_exact_requested_experiments(variant,weights,folder):
    assert set(e5.EXPERIMENTS) == {"E5-A","E5-B"}
    assert e5.EXPERIMENTS[variant] == {"weights":weights,"run_name":folder}
    with pytest.raises(RuntimeError,match="Only E5-A and E5-B"):
        e5.make_class_weight_trainer("OTHER")


@pytest.mark.parametrize("variant",["E5-A","E5-B"])
@pytest.mark.parametrize("class_id",[0,1,2,3])
def test_classwise_contributions_and_no_double_weighting(loss_check,variant,class_id):
    result = loss_check["variants"][variant]
    expected = e5.EXPERIMENTS[variant]["weights"][class_id]
    assert result["per_class_bce_ratio"][class_id] == pytest.approx(expected,rel=1e-6)
    assert result["effective_model"][class_id] == expected
    assert result["effective_criterion"][class_id] == expected
    assert result["no_double_weighting"]
    if class_id in {0,2}:
        assert result["per_class_bce_ratio"][class_id] > 1
    else:
        assert result["per_class_bce_ratio"][class_id] == pytest.approx(1,abs=1e-6)


@pytest.mark.parametrize("variant",["E5-A","E5-B"])
def test_nonzero_box_dfl_and_gradients_unchanged(loss_check,variant):
    result = loss_check["variants"][variant]
    assert result["box_loss_unchanged"] and result["box_gradient_unchanged"]
    assert result["dfl_loss_unchanged"] and result["dfl_gradient_unchanged"]
    assert result["assignment_unchanged"]


def make_stub_trainer(variant):
    from ultralytics.utils.loss import v8DetectionLoss
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))
            self.model = [SimpleNamespace(nc=4,reg_max=16,stride=torch.tensor([8.]))]
            self.args = SimpleNamespace(box=7.5,cls=.5,dfl=1.5)
    cls = e5.make_class_weight_trainer(variant)
    trainer = object.__new__(cls)
    trainer.model = Model()
    trainer.args = SimpleNamespace(cls_pw=0.0)
    trainer.data = {"nc":4,"names":e5.NAMES.copy()}
    return trainer


@pytest.mark.parametrize("variant",["E5-A","E5-B"])
def test_setter_replaces_without_normalization_or_parameter_change(variant):
    from ultralytics.utils.loss import v8DetectionLoss
    trainer = make_stub_trainer(variant)
    parameters = list(trainer.model.named_parameters())
    trainer.model.class_weights = torch.tensor([20.,20.,20.,20.])  # must replace, never multiply
    trainer.set_class_weights()
    trainer.set_class_weights()
    expected = torch.tensor(e5.EXPERIMENTS[variant]["weights"])
    assert torch.equal(trainer.model.class_weights,expected)
    assert len(list(trainer.model.named_parameters())) == len(parameters)
    assert all(a is b for (_,a),(_,b) in zip(parameters,trainer.model.named_parameters()))
    assert set(dict(trainer.model.named_buffers())) == set()
    trainer.model.criterion = v8DetectionLoss(trainer.model)
    actual = e5.actual_weight_state(trainer.model,variant,require_criterion=True)
    assert actual["effective_criterion"] == expected.tolist()
    assert type(trainer.model.criterion.bce) is torch.nn.BCEWithLogitsLoss
    assert trainer.model.criterion.bce.weight is None
    assert trainer.model.criterion.bce.pos_weight is None
    assert trainer.model.criterion.hyp.cls == .5  # global cls gain unchanged
    assert trainer.model.criterion.hyp.box == 7.5
    assert trainer.model.criterion.hyp.dfl == 1.5
    with pytest.raises(RuntimeError,match="before lazy loss initialization"):
        trainer.set_class_weights()  # do not silently change an already-running criterion


def test_automatic_frequency_weighting_and_wrong_mapping_rejected():
    trainer = make_stub_trainer("E5-A")
    trainer.args.cls_pw = .5
    with pytest.raises(RuntimeError,match="frequency weighting"):
        trainer.set_class_weights()
    trainer.args.cls_pw = 0
    trainer.data["names"] = {0:"p",1:"o",2:"s",3:"sc"}
    with pytest.raises(RuntimeError,match="class order"):
        trainer.set_class_weights()


def test_native_sampler_model_and_loss_methods_are_inherited():
    from ultralytics.models.yolo.detect import DetectionTrainer
    cls = e5.make_class_weight_trainer("E5-A")
    assert set(k for k in cls.__dict__ if not k.startswith("__")) == {"set_class_weights"}
    for method in ["get_dataloader","get_model","build_dataset","preprocess_batch"]:
        assert getattr(cls,method) is getattr(DetectionTrainer,method)
    assert "WeightedRandomSampler" not in SCRIPT.read_text(encoding="utf-8")


def test_audited_native_loss_is_retained():
    info = e5.native_loss_structure()
    assert info["source_sha256"] == e5.LOSS_SOURCE_SHA256
    assert info["ultralytics"] == "8.4.174"
    assert "once" in info["application"]


def test_val_only_evaluation_and_no_predict_calls():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    inference = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and
                 isinstance(n.func,ast.Attribute) and n.func.attr in {"val","predict"}]
    assert len(inference) == 1 and inference[0].func.attr == "val"
    evaluate = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "evaluate_best_validation")
    settings = [n for n in ast.walk(evaluate) if isinstance(n,ast.Call) and
                isinstance(n.func,ast.Name) and n.func.id == "dict"]
    assert any(any(k.arg == "split" and isinstance(k.value,ast.Constant) and k.value.value == "val"
                   for k in call.keywords) for call in settings)


def test_e1_requested_settings_unchanged_except_paths_and_gpu(monkeypatch,tmp_path):
    source = dict(data="old/data.yaml",device="cpu",project="old",name="E1",
                  epochs=100,imgsz=640,batch=8,seed=42,patience=20,workers=0,
                  deterministic=True,pretrained=True,amp=False,auto_augment=None,
                  augment=False,close_mosaic=0)
    source.update({k:0 for k in e5.ZERO_AUGMENTATIONS})
    reference = {"requested_train_args":source.copy()}
    monkeypatch.setattr(e5,"RUN",tmp_path/"e5a_weight_1p5")
    current = e5.training_settings(reference)
    allowed = {"data","device","project","name","optimizer"}
    assert {k for k in source.keys()|current.keys() if source.get(k) != current.get(k)} <= allowed
    assert source == reference["requested_train_args"]
    assert current["optimizer"] == "auto"
    assert current["device"] == 0
    assert current["epochs"] == 100 and current["batch"] == 8 and current["amp"] is False
    assert all(current[k] == 0 for k in e5.ZERO_AUGMENTATIONS)
