"""Create immutable group-level assignments only; never copy or alter GT/images.

Run from the repository: .venv/Scripts/python.exe -B scripts/create_fixed_split_v1.py
Use --verify-only to regenerate in memory and compare all three frozen outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/derived/gt_629_audit_v1"
OUTPUT = ROOT / "data/derived/fixed_split_v1"
SEED = 42
SPLITS = ("train", "val", "test")
RATIOS = (0.8, 0.1, 0.1)
CLASSES = ("p", "o", "s", "sc")
GROUP_PATTERN = re.compile(r"^(w[0-9]+)[_-]([0-9]+)(?=[_.-]|$)", re.I)
FIELDS = (
    "image_id", "filename", "original_filename", "group_key", "split",
    "wafer", "exposure", "position", "p_count", "o_count", "s_count",
    "sc_count", "total_annotations", "zero_annotation", "rare_class",
    "assignment_reason", "sha256", "grouping_status",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_source():
    """Cross-check audited identities/counts/hashes against the unmodified COCO ZIP."""
    audit_summary_path = AUDIT / "dataset_summary.json"
    manifest_path = AUDIT / "image_manifest.csv"
    audit = json.loads(audit_summary_path.read_text(encoding="utf-8-sig"))
    protected = {audit_summary_path: digest(audit_summary_path),
                 manifest_path: digest(manifest_path)}
    for relative, expected in audit["source_sha256"].items():
        path = ROOT / relative
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Source archive differs from audited source: {relative}")
        protected[path] = actual
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    coco_path = ROOT / audit["sources"]["coco"]
    with zipfile.ZipFile(coco_path) as archive:
        json_members = [n for n in archive.namelist() if n.endswith("_annotations.coco.json")]
        if len(json_members) != 1:
            raise ValueError("Expected exactly one authoritative COCO JSON")
        coco_bytes = archive.read(json_members[0])
        coco = json.loads(coco_bytes)
        categories = {c["id"]: c["name"] for c in coco["categories"]}
        counts = defaultdict(Counter)
        for ann in coco["annotations"]:
            name = categories[ann["category_id"]]
            if name not in CLASSES:
                raise ValueError(f"Unexpected annotated category: {name}")
            counts[str(ann["image_id"])][name] += 1
        images = {str(im["id"]): im for im in coco["images"]}
        if len(images) != 629 or len(rows) != 629 or len(coco["annotations"]) != 5953:
            raise ValueError("Source dataset size changed; do not silently regenerate v1")
        if len({r["image_id"] for r in rows}) != 629:
            raise ValueError("Repeated audit image_id")
        if {r["image_id"] for r in rows} != set(images):
            raise ValueError("COCO/audit identity mismatch")
        for row in rows:
            im = images[row["image_id"]]
            if row["filename"] != im["file_name"]:
                raise ValueError("COCO/audit filename mismatch")
            if row["original_filename"] != im["extra"]["name"]:
                raise ValueError("COCO/audit original filename mismatch")
            for name in CLASSES:
                row[name + "_count"] = int(row[name + "_count"])
                if row[name + "_count"] != counts[row["image_id"]][name]:
                    raise ValueError("COCO/audit annotation count mismatch")
            row["total_annotations"] = sum(counts[row["image_id"]].values())
            zero = row["total_annotations"] == 0
            if str(zero) != row["zero_annotation"]:
                raise ValueError("COCO/audit zero-label mismatch")
            row["zero_annotation"] = zero
            actual = hashlib.sha256(archive.read(row["zip_member"])).hexdigest()
            if actual != row["sha256"]:
                raise ValueError("Image bytes differ from audited SHA-256")
            match = GROUP_PATTERN.match(row["original_filename"])
            if match:
                wafer, number = match.groups()
                if wafer.lower() != row["wafer"].lower():
                    raise ValueError("Parsed wafer conflicts with audit metadata")
                # Numeric capture/die token, not a physical-coordinate or exposure inference.
                row["group_key"] = f"{wafer.lower()}_{int(number)}"
                row["grouping_status"] = "PARSED_WAFER_CAPTURE_NUMBER"
            else:
                row["group_key"] = "UNRESOLVED_SHA256_" + actual
                row["grouping_status"] = "UNRESOLVED_MANUAL_REVIEW"
            row["rare_class"] = "|".join(c for c in ("o", "s") if row[c + "_count"])
    if len({r["sha256"] for r in rows}) != 629:
        raise ValueError("Exact image duplicates require explicit group reconciliation")
    if sum(r["zero_annotation"] for r in rows) != 6:
        raise ValueError("Human-confirmed negative set changed")
    assert {c: sum(r[c + "_count"] for r in rows) for c in CLASSES} == {
        "p": 5444, "o": 3, "s": 7, "sc": 499}
    return sorted(rows, key=lambda r: int(r["image_id"])), protected, {
        "coco_archive": audit["sources"]["coco"],
        "coco_json_member": json_members[0],
        "coco_json_sha256": hashlib.sha256(coco_bytes).hexdigest(),
        "audit_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "input_sha256": {p.relative_to(ROOT).as_posix(): h
                         for p, h in protected.items()},
    }


def make_vectors(rows, groups):
    """Soft balance: image count, p/sc instances and positive images, wafer/position."""
    wafers = sorted({r["wafer"] for r in rows})
    positions = sorted({r["position"] or "unknown" for r in rows})
    weights = [10, 1, 1, 1, 1] + [2 / len(wafers)] * len(wafers)
    weights += [1 / len(positions)] * len(positions)

    def vector(rs):
        return [len(rs), sum(r["p_count"] for r in rs),
                sum(r["sc_count"] for r in rs),
                sum(r["p_count"] > 0 for r in rs),
                sum(r["sc_count"] > 0 for r in rs)] + [
                    sum(r["wafer"] == w for r in rs) for w in wafers
                ] + [sum((r["position"] or "unknown") == p for r in rs)
                     for p in positions]

    vectors = {g: vector(rs) for g, rs in groups.items()}
    total = vector(rows)
    targets = [[ratio * value for value in total] for ratio in RATIOS]

    def loss(values):
        return sum(w * ((v - t) / max(t, 1)) ** 2
                   for value, target in zip(values, targets)
                   for v, t, w in zip(value, target, weights))

    return vectors, loss, len(total)


def assign_groups(rows):
    """Exhaustive feasible rare/negative anchors, then seeded group-level balancing.

    Anchor requirements: open images 2/0/1; short-positive images 4/1/1;
    human-confirmed negatives 4/1/1. Infeasibility stops instead of splitting groups.
    Free groups use largest-first seeded greedy assignment and strict-improvement
    moves/swaps. Exposure is deliberately not part of the assignment objective.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_key"]].append(row)
    vectors, loss, size = make_vectors(rows, groups)
    rng = random.Random(SEED)
    rare_groups = sorted(g for g, rs in groups.items()
                         if any(r["o_count"] or r["s_count"] for r in rs))
    negative_only_groups = sorted(g for g, rs in groups.items()
                                  if g not in rare_groups and
                                  any(r["zero_annotation"] for r in rs))
    rare_stats = {g: (sum(r["o_count"] > 0 for r in groups[g]),
                      sum(r["s_count"] > 0 for r in groups[g]))
                  for g in rare_groups}
    zero_stats = {g: sum(r["zero_annotation"] for r in rs)
                  for g, rs in groups.items()}
    best_key = None
    feasible = 0
    for allocation in itertools.product(range(3), repeat=len(rare_groups)):
        open_images = [0, 0, 0]
        short_images = [0, 0, 0]
        for g, split in zip(rare_groups, allocation):
            open_images[split] += rare_stats[g][0]
            short_images[split] += rare_stats[g][1]
        if open_images != [2, 0, 1] or short_images != [4, 1, 1]:
            continue
        base = dict(zip(rare_groups, allocation))
        for negative_allocation in itertools.product(range(3), repeat=len(negative_only_groups)):
            anchors = base | dict(zip(negative_only_groups, negative_allocation))
            negatives = [0, 0, 0]
            values = [[0] * size for _ in SPLITS]
            for g, split in anchors.items():
                negatives[split] += zero_stats[g]
                values[split] = [a + b for a, b in zip(values[split], vectors[g])]
            if negatives != [4, 1, 1]:
                continue
            feasible += 1
            key = (loss(values), rng.random())
            if best_key is None or key < best_key:
                best_key, best_anchors = key, anchors
    if best_key is None:
        raise ValueError("Rare/negative targets infeasible with group integrity; manual review required")
    assigned = best_anchors.copy()
    locked = set(assigned)
    values = [[0] * size for _ in SPLITS]
    for g, split in assigned.items():
        values[split] = [a + b for a, b in zip(values[split], vectors[g])]
    free = sorted(set(groups) - locked)
    rng.shuffle(free)
    free.sort(key=lambda g: -len(groups[g]))
    for g in free:
        options = []
        for split in range(3):
            trial = [v.copy() for v in values]
            trial[split] = [a + b for a, b in zip(trial[split], vectors[g])]
            options.append((loss(trial), rng.random(), split, trial))
        _, _, chosen, values = min(options)
        assigned[g] = chosen
    # Bounded, deterministic balancing only; locked rare/negative groups never move.
    passes = 0
    for _ in range(200):
        current = loss(values)
        best = (current, None, None)
        for g in free:
            old = assigned[g]
            for new in range(3):
                if old == new:
                    continue
                trial = [v.copy() for v in values]
                trial[old] = [a - b for a, b in zip(trial[old], vectors[g])]
                trial[new] = [a + b for a, b in zip(trial[new], vectors[g])]
                score = loss(trial)
                if score < best[0] - 1e-12:
                    best = score, (g, new, None), trial
        for i, g in enumerate(free):
            for h in free[i + 1:]:
                a, b = assigned[g], assigned[h]
                if a == b:
                    continue
                trial = [v.copy() for v in values]
                trial[a] = [x - y + z for x, y, z in zip(values[a], vectors[g], vectors[h])]
                trial[b] = [x - y + z for x, y, z in zip(values[b], vectors[h], vectors[g])]
                score = loss(trial)
                if score < best[0] - 1e-12:
                    best = score, (g, b, h), trial
        if best[1] is None:
            break
        g, new, h = best[1]
        if h is not None:
            assigned[h] = assigned[g]
        assigned[g] = new
        values = best[2]
        passes += 1
    for row in rows:
        g = row["group_key"]
        row["split"] = SPLITS[assigned[g]]
        if g in locked:
            row["assignment_reason"] = "group_integrity; rare/negative feasible anchor"
        else:
            row["assignment_reason"] = "group_integrity; seeded p/sc+wafer+position balance"
        if row["zero_annotation"]:
            row["assignment_reason"] += "; human-confirmed normal negative"
        if row["grouping_status"].startswith("UNRESOLVED"):
            row["assignment_reason"] += "; grouping unknown: manual review required"
    return {"feasible_anchor_allocations": feasible, "balance_improvement_passes": passes,
            "balance_objective": loss(values), "locked_groups": sorted(locked),
            "soft_balance_weights": {"image_count": 10, "p_instances": 1,
                                    "sc_instances": 1, "p_positive_images": 1,
                                    "sc_positive_images": 1, "wafer_total": 2,
                                    "position_total": 1}}


def summarize(rows, provenance, algorithm):
    result = {
        "version": "fixed_split_v1", "seed": SEED,
        "target_ratios": dict(zip(SPLITS, RATIOS)),
        "grouping_rule": "Original filename anchored wafer+numeric capture/die token; c/lt/rb and other suffixes, extensions, exposure variants stay together.",
        "unknown_group_policy": "Unique SHA-256 group; explicitly unresolved; never infer missing wafer/die.",
        "unresolved_grouping": [{k: r[k] for k in ("image_id", "original_filename", "group_key")}
                               for r in rows if r["grouping_status"].startswith("UNRESOLVED")],
        "source": provenance, "algorithm": algorithm,
        "negative_policy": "All six zero-label images human-confirmed normal negatives by user; audit historical uncertainty is not rewritten.",
        "benchmark_policy": "Freeze Val/Test for Baseline/Oversampling/Augmentation/Synthetic; train-only future transformations. Assignment metadata only, not a training-ready export.",
        "exposure_policy": "Observed metadata only, not used for assignment; missing exposure remains blank.",
        "source_qa_policy": "Existing source/audit annotation issues are preserved; this split neither repairs nor approves annotations for training.",
        "splits": {},
    }
    for split in SPLITS:
        rs = [r for r in rows if r["split"] == split]
        result["splits"][split] = {
            "image_count": len(rs), "ratio": len(rs) / len(rows),
            "group_count": len({r["group_key"] for r in rs}),
            "annotation_counts": {c: sum(r[c + "_count"] for r in rs) for c in CLASSES},
            "positive_image_counts": {c: sum(r[c + "_count"] > 0 for r in rs) for c in CLASSES},
            "wafer_distribution": dict(sorted(Counter(r["wafer"] or "unknown" for r in rs).items())),
            "position_distribution": dict(sorted(Counter(r["position"] or "unknown" for r in rs).items())),
            "available_exposure_distribution": dict(sorted(Counter(r["exposure"] for r in rs if r["exposure"]).items())),
            "exposure_available_images": sum(bool(r["exposure"]) for r in rs),
            "exposure_missing_images": sum(not r["exposure"] for r in rs),
            "zero_label_images": sum(r["zero_annotation"] for r in rs),
            "zero_label_image_ids": [r["image_id"] for r in rs if r["zero_annotation"]],
            "rare_class": {c: {
                "annotations": sum(r[c + "_count"] for r in rs),
                "images": sum(r[c + "_count"] > 0 for r in rs),
                "groups": sorted({r["group_key"] for r in rs if r[c + "_count"]}),
                "image_ids": [r["image_id"] for r in rs if r[c + "_count"]],
                "original_filenames": [r["original_filename"] for r in rs if r[c + "_count"]],
            } for c in ("o", "s")},
        }
    result["exposure_bias_check"] = {
        "available_total": sum(bool(r["exposure"]) for r in rows),
        "missing_total": sum(not r["exposure"] for r in rows),
        "available_share_by_split": {
            s: result["splits"][s]["exposure_available_images"] /
            max(result["splits"][s]["image_count"], 1) for s in SPLITS},
        "interpretation": "Descriptive bias check only; 413 unknown exposures are not inferred. No exposure-based assignment or regrouping.",
    }
    return result


def validate(rows):
    assert len(rows) == len({r["image_id"] for r in rows}) == 629
    assert len({r["filename"] for r in rows}) == 629
    assert len({r["sha256"] for r in rows}) == 629
    groups = defaultdict(set)
    for r in rows:
        assert r["split"] in SPLITS
        groups[r["group_key"]].add(r["split"])
    assert all(len(s) == 1 for s in groups.values())
    assert [sum(r["zero_annotation"] for r in rows if r["split"] == s)
            for s in SPLITS] == [4, 1, 1]
    assert [sum(r["o_count"] > 0 for r in rows if r["split"] == s)
            for s in SPLITS] == [2, 0, 1]
    assert [sum(r["s_count"] > 0 for r in rows if r["split"] == s)
            for s in SPLITS] == [4, 1, 1]
    return {"images_exactly_once": 629, "missing_images": 0,
            "cross_split_image_overlap": 0, "cross_split_sha256_overlap": 0,
            "cross_split_group_leakage": 0, "zero_label_images_included": 6,
            "rare_allocation_targets_met": True}


def artifacts(rows, summary):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    manifest_bytes = buffer.getvalue().encode("utf-8-sig")
    summary["split_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    counts = summary["splits"]
    lines = ["# Fixed benchmark split v1", "",
             "Assignment-only common benchmark for Baseline, Oversampling, Augmentation and Synthetic experiments.",
             "Val/Test assignments must remain unchanged. No images/labels were copied, converted or modified.", "",
             "Source: " + summary["source"]["coco_archive"],
             "Audited metadata: data/derived/gt_629_audit_v1/image_manifest.csv.",
             "COCO is authoritative; original category IDs/names and annotations are untouched.", "",
             "Grouping: original filename wafer + numeric capture/die number, e.g. w4_38.",
             "All positions/suffixes/exposure variants of that group share one split.",
             f"Unresolved grouping images: {len(summary['unresolved_grouping'])}; recorded explicitly in split_summary.json.",
             "", "Rare groups allocated before other groups: open positive images 2/0/1;",
             "short positive images 4/1/1 (Train/Val/Test). Group integrity takes precedence.",
             "All six zero-label images are user-confirmed normal negatives; allocation 4/1/1.", "",
             "| Split | Images | Ratio | Open instances/images/groups | Short instances/images/groups | Negatives |",
             "|---|---:|---:|---|---|---:|"]
    for s in SPLITS:
        d = counts[s]
        rare = []
        for c in ("o", "s"):
            v = d["rare_class"][c]
            rare.append(f"{v['annotations']}/{v['images']}/{len(v['groups'])}")
        lines.append(f"| {s} | {d['image_count']} | {d['ratio']:.2%} | {rare[0]} | {rare[1]} | {d['zero_label_images']} |")
    lines += ["", f"Seed: {SEED}. Sorted inputs, exhaustive feasible rare/negative anchors, seeded",
              "largest-group-first greedy assignment, deterministic strict-improvement group moves/swaps.",
              "Soft balance uses image count, p/sc instance and positive-image counts, wafer and position.",
              "Exposure is not an assignment criterion: only observed values are summarized; unknown stays blank.",
              "Actual exposure coverage by split and distributions are in split_summary.json.",
              "", "Reproduction from the repository root (existing outputs are never overwritten):",
              ".venv/Scripts/python.exe -B scripts/create_fixed_split_v1.py",
              "Verification regenerates all outputs in memory and requires byte-identical files:",
              ".venv/Scripts/python.exe -B scripts/create_fixed_split_v1.py --verify-only",
              "Input archive/manifest hashes and output manifest SHA-256 are recorded in the summary.",
              "", "Existing QA findings remain in gt_629_audit_v1 and are not corrected here.",
              "A frozen assignment does not certify annotation quality or make a training-ready export.",
              "No training, augmentation, oversampling, synthetic generation or annotation conversion was run.", ""]
    return {"split_manifest.csv": manifest_bytes,
            "split_summary.json": (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            "README.md": "\n".join(lines).encode("utf-8")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and OUTPUT.exists():
        raise FileExistsError("Frozen output already exists; use --verify-only, never overwrite v1")
    rows, protected, provenance = load_source()
    algorithm = assign_groups(rows)
    summary = summarize(rows, provenance, algorithm)
    summary["validation"] = validate(rows)
    if any(digest(p) != original for p, original in protected.items()):
        raise ValueError("Protected source changed during assignment")
    summary["validation"]["source_archives_and_audit_inputs_unchanged"] = True
    outputs = artifacts(rows, summary)
    # Independent serialization/readback before any write.
    decoded = list(csv.DictReader(io.StringIO(outputs["split_manifest.csv"].decode("utf-8-sig"))))
    assert len(decoded) == 629 and len({r["image_id"] for r in decoded}) == 629
    assert {r["image_id"]: r["split"] for r in decoded} == {r["image_id"]: r["split"] for r in rows}
    if args.verify_only:
        for name, content in outputs.items():
            if (OUTPUT / name).read_bytes() != content:
                raise ValueError(f"Frozen output not reproducible: {name}")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        for name, content in outputs.items():
            with (OUTPUT / name).open("xb") as stream:
                stream.write(content)
        for name, content in outputs.items():
            assert (OUTPUT / name).read_bytes() == content
    print(json.dumps({"mode": "verified" if args.verify_only else "created",
                      "splits": summary["splits"], "validation": summary["validation"],
                      "unresolved_grouping": summary["unresolved_grouping"],
                      "manifest_sha256": summary["split_manifest_sha256"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

