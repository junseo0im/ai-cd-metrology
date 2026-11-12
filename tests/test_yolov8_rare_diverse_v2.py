"""E6 helper geometry/determinism tests, without real data/model/training."""
from collections import Counter
import ast
import importlib.util
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
build = load("e6_builder",ROOT/"scripts/build_yolov8_rare_diverse_v2.py")
train = load("e6_train",ROOT/"scripts/train_yolov8_rare_diverse_v2_colab.py")


def test_bbox_translation_four_corner_expected_coordinates():
    row = (1,0,.5,.5,.2,.4)
    encoded,raw,clipped,did_clip = build.transform_bbox(row,100,100,build.matrix_for(100,100,0,10,-5))
    assert encoded == pytest.approx((0,.6,.45,.2,.4))
    assert raw == pytest.approx([50,25,70,65])
    assert clipped == pytest.approx(raw)
    assert not did_clip


def test_rotation_aabb_preserves_class_and_swaps_extents():
    encoded,_,_,_ = build.transform_bbox((1,2,.5,.5,.2,.4),100,100,build.matrix_for(100,100,90,0,0))
    assert encoded == pytest.approx((2,.5,.5,.4,.2))


def test_bbox_clipping_and_off_canvas_rejection():
    row = (1,1,.9,.5,.2,.2)
    encoded,_,clipped,did_clip = build.transform_bbox(row,100,100,build.matrix_for(100,100,0,5,0))
    assert did_clip and encoded == pytest.approx((1,.925,.5,.15,.2))
    assert clipped == pytest.approx([85,40,100,60])
    encoded,_,_,_ = build.transform_bbox(row,100,100,build.matrix_for(100,100,0,25,0))
    assert encoded is None


@pytest.mark.parametrize("variant",["A","B","C","D"])
def test_families_are_distinct_and_stay_in_fixed_ranges(variant):
    rng = random.Random(42)
    for _ in range(100):
        p = build.sample_parameters(rng,variant,1000,500)
        assert p["flip"] is False
        if variant == "A":
            assert .90 <= p["brightness_factor"] <= 1.10 and .92 <= p["contrast_factor"] <= 1.08
        elif variant == "B":
            assert -5 <= p["angle_deg"] <= 5 and p["angle_deg"] != 0
        elif variant == "C":
            assert -30 <= p["translate_x_px"] <= 30 and -15 <= p["translate_y_px"] <= 15
            assert (p["translate_x_px"],p["translate_y_px"]) != (0,0)
        else:
            assert p["blur_kernel_size"] == 3 and .3 <= p["blur_sigma_px"] <= .7
            assert .95 <= p["brightness_factor"] <= 1.05 and .97 <= p["contrast_factor"] <= 1.03
        if variant != "B":
            assert p["angle_deg"] == 0
        if variant != "C":
            assert p["translate_x_px"] == p["translate_y_px"] == 0
        if variant != "D":
            assert p["blur_kernel_size"] == p["blur_sigma_px"] == 0
        if variant in {"B","C"}:
            assert p["brightness_factor"] == p["contrast_factor"] == 1


@pytest.fixture
def parents(tmp_path,monkeypatch):
    folder = tmp_path/"E1"
    (folder/"train/labels").mkdir(parents=True)
    rows = {}
    for i in range(35):
        filename = f"parent_{i}.png"
        (folder/"train/labels"/f"parent_{i}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1\n",encoding="ascii")
        rows[i] = dict(filename=filename,width="100",height="100",group_key=f"group{i}",
                       o_count="1",s_count="1")
    monkeypatch.setattr(build,"SOURCE",folder)
    return rows


def test_plan_has_exact_four_variants_and_deterministic_replay(parents):
    first = build.build_plan(parents,list(parents))
    second = build.build_plan(parents,list(reversed(parents)))
    assert first[0] == second[0]  # sorted IDs, independent of caller iteration order
    assert len(first[0]) == len(first[1]) == 140
    assert Counter(r["variant"] for r in first[0]) == {"A":35,"B":35,"C":35,"D":35}
    assert Counter(r["parent_image_id"] for r in first[0]) == {i:4 for i in parents}
    for plan_a,plan_b in zip(first[1],second[1]):
        assert np.array_equal(plan_a["matrix"],plan_b["matrix"])
        assert plan_a["rows"] == plan_b["rows"]
        assert len(plan_a["rows"]) == 2
        assert [r[0] for r in plan_a["rows"]] == [0,2]


def test_invalid_transform_is_retried_without_deleting_a_bbox(parents,monkeypatch):
    original = build.transform_bbox
    calls = []
    def invalid_once(*args,**kwargs):
        calls.append(1)
        if len(calls) == 1:
            return None,[0,0,1,0],[0,0,1,0],True
        return original(*args,**kwargs)
    monkeypatch.setattr(build,"transform_bbox",invalid_once)
    manifest,plans,rejected,_ = build.build_plan(parents,list(parents))
    assert len(manifest) == len(plans) == 140
    assert rejected[0]["variant"] == "B" and rejected[0]["attempt"] == 1
    assert next(r for r in manifest if r["parent_image_id"] == 0 and r["variant"] == "B")["retry_count"] == 1
    assert all(len(p["rows"]) == 2 for p in plans)


def test_retry_exhaustion_stops_generation(parents,monkeypatch):
    monkeypatch.setattr(build,"transform_bbox",lambda *a,**k:(None,[0]*4,[0]*4,True))
    with pytest.raises(ValueError,match="2 invalid transforms; no bbox removed"):
        build.build_plan(parents,list(parents),max_attempts=2)


@pytest.mark.parametrize("variant",["A","D"])
def test_photometric_labels_are_byte_identical(parents,variant):
    manifest,plans,_,_ = build.build_plan(parents,list(parents))
    plan = next(p for p in plans if p["info"]["variant"] == variant)
    source = build.SOURCE/"train/labels"/(Path(plan["info"]["parent_filename"]).stem+".txt")
    assert build.encoded_label(plan) == source.read_bytes()


def test_pixel_variants_keep_native_canvas_and_source_bytes(tmp_path,monkeypatch):
    source = tmp_path/"source"
    (source/"train/images").mkdir(parents=True)
    y,x = np.indices((32,48))
    rgb = np.stack([x*5,y*7,(x+y)*3],axis=-1).astype(np.uint8)
    path = source/"train/images/parent.png"
    Image.fromarray(rgb).save(path)
    before = path.read_bytes()
    monkeypatch.setattr(build,"SOURCE",source)
    for variant in build.PARAMETERS:
        p = build.sample_parameters(random.Random(42),variant,48,32)
        p.update(parent_filename=path.name,width=48,height=32)
        matrix = build.matrix_for(48,32,p["angle_deg"],p["translate_x_px"],p["translate_y_px"])
        actual = build.augmented_pixels(p,matrix)
        assert actual.shape == rgb.shape and actual.dtype == np.uint8
        assert not np.array_equal(actual,rgb)
    assert path.read_bytes() == before


def test_native_training_without_weighting_and_only_val_inference():
    path = ROOT/"scripts/train_yolov8_rare_diverse_v2_colab.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert not any(isinstance(n,ast.ClassDef) for n in tree.body)
    assert "WeightedRandomSampler" not in text
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not any(n.func.attr == "predict" for n in calls)
    assert len([n for n in calls if n.func.attr == "val"]) == 1
    fitting = next(n for n in calls if n.func.attr == "train")
    assert not any(k.arg == "trainer" for k in fitting.keywords)
    evaluator = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "evaluate_best_validation")
    assert any(isinstance(n,ast.keyword) and n.arg == "split" and isinstance(n.value,ast.Constant) and
               n.value.value == "val" for n in ast.walk(evaluator))


def test_saved_training_policy_matches_e1_except_paths(monkeypatch,tmp_path):
    source = dict(data="E1.yaml",device=0,project="old",name="E1",epochs=100,imgsz=640,batch=8,
                  seed=42,patience=20,workers=0,deterministic=True,pretrained=True,amp=False,
                  auto_augment=None,augment=False,close_mosaic=0,optimizer="auto")
    source.update({k:0 for k in train.ZERO_AUGMENTATIONS})
    reference = {"requested_train_args":source.copy()}
    monkeypatch.setattr(train,"RUN",tmp_path/"E6")
    actual = train.training_settings(reference)
    assert {k for k in source.keys()|actual.keys() if source.get(k) != actual.get(k)} <= {"data","project","name"}
    assert source == reference["requested_train_args"]
