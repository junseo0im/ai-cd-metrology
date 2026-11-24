"""I&A 결과 리포트 (1차): one report data set for both the preview and the Word (.docx) export.

Read-only. Everything comes from the existing I&A path: per-image Wafer/process (collect_images +
metadata_frame), latest stored analysis per image (latest_analysis / stored_results), the I&A
eligibility and averages (image_mean_um / summarize_um), the Wafer Map Die colour (die_snapshot +
map_risk) and the verified stored overlays (select_run). Nothing is recomputed by the engines,
judged anew or written back.
"""
from collections import Counter
from datetime import datetime
from io import BytesIO
import hashlib
import json
from statistics import mean, stdev

import pandas as pd

from dashboard.process_analysis import DESIGN_LINE_UM, image_mean_um, metadata_frame, summarize_um
from dashboard.units import EXPOSURE_DISPLAY_UNIT, stored_unit_text

ALL_CONDITIONS = "__all__"
SECTIONS = ("defects", "wafer_map", "images")
OUTCOME_LABELS = {"COMPLETE": "분석 완료", "QUALITY_REVIEW_REQUIRED": "검토 필요 · 품질",
                  "PARTIAL_MEASUREMENT": "부분 계측", "POLICY_REVIEW_REQUIRED": "PD 정책 검토",
                  "MODEL_PD_EXCLUDED": "PD 검출 · 계측 제외", "FAILED": "실패"}


# ---- scope ---------------------------------------------------------------------------------------
def condition_key(row):
    peb = row["peb_condition"]
    return f"{row['exposure_condition']} · " + (peb if peb.startswith("PEB") else "PEB " + peb)


def target_images(provider, kind, target_id):
    from dashboard.components.analysis_management import collect_images
    frame = collect_images(provider, batch_id=target_id) if kind == "Batch" else collect_images(provider, wafer_id=target_id)
    frame = metadata_frame(frame)
    frame["condition"] = [condition_key(r) for r in frame.to_dict("records")]
    return frame


def scope(provider, kind, target_id, condition=ALL_CONDITIONS):
    """Images of ONE Wafer or ONE Batch, optionally ONE (노광, PEB) condition; grouped Wafer x condition."""
    from dashboard.components.process_analysis import stored_results
    images = target_images(provider, kind, target_id)
    if condition != ALL_CONDITIONS:
        images = images[images["condition"] == condition]
    analysis = provider.latest_analysis()
    ids = set(images["image_id"])
    analysis = {k: v for k, v in analysis.items() if k in ids}
    results = {k: v for k, v in stored_results(provider).items() if k in ids}
    groups = [(wafer, cond, frame) for (wafer, cond), frame in images.groupby(["wafer_id", "condition"], sort=True)]
    return {"kind": kind, "target_id": target_id, "condition": condition, "images": images,
            "analysis": analysis, "results": results, "groups": groups}


def signature(scope_data, options):
    """Identity of exactly what a generated document contains (scope + inputs + checked sections)."""
    payload = json.dumps([scope_data["kind"], scope_data["target_id"], scope_data["condition"],
                          sorted(scope_data["images"]["image_id"]), options], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---- report data ---------------------------------------------------------------------------------
def _image_state(row, results):
    return image_mean_um(results.get(row["image_id"]), magnification=row.get("magnification"))


def _stats(rows, data):
    """I&A averages (eligible READY images only) + stored measurement counts of those images."""
    states = [(_image_state(r, data["results"]), r) for r in rows]
    ready = [(s, r) for s, r in states if s["status"] == "READY"]
    line = summarize_um([s["mean_um"] for s, _ in ready])
    gaps = []
    for _, r in ready:
        widths = (data["results"].get(r["image_id"]) or {}).get("valid_gap_widths_um") or []
        if widths:
            gaps.append(mean(widths))
    items = [data["analysis"].get(r["image_id"]) or {} for _, r in ready]
    # TASK18-B-16: every valid individual Line Width of those images (all Dies of the group).
    widths = [w for _, r in ready for w in (data["results"].get(r["image_id"]) or {}).get("valid_line_widths_um") or []]
    return {"lw_mean": mean(widths) if widths else None, "lw_std": stdev(widths) if len(widths) >= 2 else None,
            "lw_n": len(widths), "registered": len(rows), "analysed": sum(r["image_id"] in data["analysis"] for r in rows),
            "measured": len(ready), "line_mean": line["mean_um"], "line_std": line["std_um"],
            "gap_mean": mean(gaps) if gaps else None, "gap_std": stdev(gaps) if len(gaps) >= 2 else None,
            "line_count": sum(int(i.get("line_measurement_count") or 0) for i in items),
            "gap_count": sum(int(i.get("gap_measurement_count") or 0) for i in items),
            "signed": line["signed_error_pct"], "absolute": line["absolute_error_pct"],
            "excluded": Counter(s["reason"] for s, _ in states if s["status"] != "READY")}


def _process(frame, data):
    rows = frame.to_dict("records")
    first = rows[0]
    items = [data["analysis"][r["image_id"]] for r in rows if r["image_id"] in data["analysis"]]
    calibrations = sorted({(i.get("calibration") or {}).get("calibration_id") or i.get("calibration_id") or "미기록" for i in items})
    times = sorted(i["started_at"] for i in items if i.get("started_at"))
    units = sorted({stored_unit_text(r.get("exposure_unit")) for r in rows})
    peb = first.get("peb_temperature_c")
    return {"exposure": first["exposure_condition"], "exposure_unit": ", ".join(units),
            "peb": "미입력" if peb is None or peb != peb else f"{peb:g}", "peb_unit": "s (peb_time_s)",
            "calibration": ", ".join(calibrations) if items else "분석 결과 없음",
            "analysed_at": (times[0][:16].replace("T", " ") + (" ~ " + times[-1][:16].replace("T", " ") if times[-1] != times[0] else "")
                            + " (UTC)") if times else "분석 결과 없음",
            "batches": sorted(set(frame["batch_id"]))}


def _defects(frame, data):
    counts = Counter(); images = 0; heuristic = 0
    for r in frame.to_dict("records"):
        item = data["analysis"].get(r["image_id"])
        if not item:
            continue
        model = [d for d in item.get("defect_details") or [] if d.get("source") == "MODEL_POSTPROCESSED"]
        heuristic += sum(d.get("source") == "HEURISTIC" for d in item.get("defect_details") or [])
        counts.update(d["class_code"] for d in model)
        images += bool(model)
    return {"counts": dict(counts), "images_with_defects": images, "heuristic_pd": heuristic,
            "analysed": sum(r["image_id"] in data["analysis"] for r in frame.to_dict("records"))}


def _dies(wafer, frame, data, provider):
    """Wafer Map Die states (same die_snapshot + map_risk as I&A) for ONE Wafer x condition group."""
    from dashboard.components.inspection_wafer import die_snapshot, duplicate_exclusions, apply_judgment_view
    records = frame.to_dict("records")
    dies = die_snapshot(wafer, provider.wafer_sites(wafer), records, data["analysis"],
                        exclude_from_judgment=duplicate_exclusions(records, data["analysis"]))
    return apply_judgment_view(dies, "AI 잠정 위험도")


def build(provider, scope_data, options):
    """The single data set used by the preview and the Word export."""
    data = scope_data
    report = {"title": options["title"], "target": f"{data['kind']} {data['target_id']}",
              "condition": "전체 조건 (Wafer·조건별 구분)" if data["condition"] == ALL_CONDITIONS else data["condition"],
              "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "author": options.get("author") or None, "purpose": options.get("purpose") or None,
              "notes": options.get("notes") or None, "sections": [s for s in SECTIONS if options.get(s)],
              "groups": [], "kind": data["kind"], "empty": data["images"].empty}
    all_rows = data["images"].to_dict("records")
    outcomes = Counter(OUTCOME_LABELS.get((data["analysis"].get(r["image_id"]) or {}).get("outcome"), "분석 결과 없음")
                       if r["image_id"] in data["analysis"] else "분석 결과 없음" for r in all_rows)
    adopted = sum((data["analysis"].get(r["image_id"]) or {}).get("is_adopted") == 1 for r in all_rows)
    report["scope"] = {**_stats(all_rows, data), "outcomes": dict(outcomes), "adopted": adopted,
                       "wafers": sorted(set(data["images"]["wafer_id"])) if all_rows else []}
    for wafer, cond, frame in data["groups"]:
        group = {"wafer": wafer, "condition": cond, "label": f"{wafer} · {cond}", "process": _process(frame, data),
                 "stats": _stats(frame.to_dict("records"), data)}
        if data["kind"] == "Wafer":   # Die-level comparison.
            group["rows"] = [{"Die": die, **_stats(f.to_dict("records"), data)}
                             for die, f in frame.groupby("die_code", sort=True)]
        if "defects" in report["sections"]:
            group["defects"] = _defects(frame, data)
        if "wafer_map" in report["sections"]:
            group["dies"] = _dies(wafer, frame, data, provider)
        report["groups"].append(group)
    if data["kind"] == "Batch":       # Condition-level comparison (Wafer x condition rows).
        report["rows"] = [{"그룹": g["label"], **g["stats"]} for g in report["groups"]]
    report["charts"] = charts(report)
    if "images" in report["sections"] and options.get("image_id") in set(data["images"]["image_id"]):
        report["image"] = representative(provider, options["image_id"])
    return report


def representative(provider, image_id):
    """Original + stored overlays of ONE image (verified by the existing I&A reader)."""
    from dashboard.components.ia_analysis_result import load_image_analysis
    record = provider.image_record(image_id)
    view = load_image_analysis(provider, record) if record else None   # latest run, no widget
    overlays = {stage: (view["overlays"].get(stage) or {}).get("path") for stage in ("metrology", "defect")} if view and view.get("state") == "OK" else {}
    return {"image_id": image_id, "filename": record["original_filename"] if record else image_id,
            "location": f"{record['wafer_id']} / Die {record['die_code']} / {record['position_code']}" if record else "-",
            "original": record["storage_path"] if record else None,
            "metrology": overlays.get("metrology"), "defect": overlays.get("defect")}


# ---- charts (matplotlib PNG; inserted into Word as pictures) --------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _png(fig):
    buffer = BytesIO(); fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buffer.getvalue()


def charts(report):
    plt = _plt(); out = []
    if report["kind"] == "Wafer":
        for g in report["groups"]:
            rows = [r for r in g.get("rows", []) if r["line_mean"] is not None]
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(8, 3.2))
            x = range(len(rows))
            ax.bar(x, [r["line_mean"] for r in rows], yerr=[r["line_std"] or 0 for r in rows], color="#568ee9",
                   capsize=3, label="평균 Line Width")
            ax.plot(list(x), [r["gap_mean"] if r["gap_mean"] is not None else float("nan") for r in rows], "o",
                    color="#e0883e", label="평균 Gap")
            ax.axhline(DESIGN_LINE_UM, color="#888", ls="--", lw=1, label=f"설계 {DESIGN_LINE_UM:g} µm")
            ax.set_xticks(list(x)); ax.set_xticklabels([r["Die"] for r in rows])
            ax.set_xlabel("Die"); ax.set_ylabel("µm"); ax.set_title(f"Die별 평균 선폭 · {g['label']}")
            ax.legend(fontsize=8, loc="lower right")
            out.append({"title": f"Die별 평균 Line Width / Gap · {g['label']}", "png": _png(fig)})
    else:
        rows = [r for r in report.get("rows", []) if r["line_mean"] is not None]
        if rows:
            fig, ax = plt.subplots(figsize=(8, 3.4))
            x = range(len(rows))
            ax.bar(x, [r["line_mean"] for r in rows], yerr=[r["line_std"] or 0 for r in rows], color="#568ee9",
                   capsize=3, label="평균 Line Width")
            ax.plot(list(x), [r["gap_mean"] if r["gap_mean"] is not None else float("nan") for r in rows], "o",
                    color="#e0883e", label="평균 Gap")
            ax.axhline(DESIGN_LINE_UM, color="#888", ls="--", lw=1, label=f"설계 {DESIGN_LINE_UM:g} µm")
            ax.set_xticks(list(x)); ax.set_xticklabels([r["그룹"] for r in rows], fontsize=8)
            ax.set_ylabel("µm"); ax.set_title("공정 조건별 평균 선폭 (Wafer · 노광 · PEB)")
            ax.legend(fontsize=8, loc="lower right")
            out.append({"title": "공정 조건별 평균 Line Width / Gap", "png": _png(fig)})
    if "wafer_map" in report["sections"]:
        for g in report["groups"]:
            if g.get("dies"):
                out.append({"title": f"Wafer Map (AI 잠정 위험도) · {g['label']}", "png": wafer_map_png(g["dies"]),
                            "wafer_map": True})
    return out


def wafer_map_png(dies):
    from dashboard.components.inspection_wafer import CRITICALITY_STYLES
    from matplotlib.patches import Circle, Rectangle
    plt = _plt(); colors = {k.value: c for k, (_, c) in CRITICALITY_STYLES.items()}
    names = {k.value: n for k, (n, _) in CRITICALITY_STYLES.items()}
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(Circle((0, 3.5), 4.65, fill=True, facecolor="#eef2f8", edgecolor="#8498b6", lw=1.5))
    for d in dies:
        cx = (d["col"] - 4.5) * 1.05; cy = 3.5 + (2 - d["row"]) * 1.5
        ax.add_patch(Rectangle((cx - .4, cy - .42), .8, .84, facecolor=colors.get(d["criticality"], "#b0bac7"),
                               edgecolor="#71839a", lw=.8))
        ax.text(cx, cy, d["die_code"], ha="center", va="center", fontsize=8, color="#172b46")
    used = sorted({d["criticality"] for d in dies}, key=list(colors).index)
    for i, name in enumerate(used):
        ax.add_patch(Rectangle((-4.6 + i * 1.6, -1.6), .3, .3, facecolor=colors[name], edgecolor="#71839a"))
        ax.text(-4.2 + i * 1.6, -1.45, names[name], fontsize=7, va="center")
    ax.set_xlim(-4.9, 4.9); ax.set_ylim(-1.9, 8.3); ax.set_aspect("equal"); ax.axis("off")
    return _png(fig)


# ---- Word -----------------------------------------------------------------------------------------
def _num(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def _pct(value):
    return "-" if value is None else f"{value:+.2f}%"


def measurement_rows(stats_rows, key):
    return [[r[key], str(r["registered"]), str(r["measured"]), _num(r["line_mean"]), str(r["line_count"]),
             _num(r["gap_mean"]), str(r["gap_count"]), _pct(r["signed"]),
             "-" if r["absolute"] is None else f"{r['absolute']:.2f}%"] for r in stats_rows]


MEASUREMENT_HEAD = ["항목", "등록", "계측 포함", "Line 평균 (µm)", "Line 측정 수", "Gap 평균 (µm)", "Gap 측정 수",
                    "Signed Error", "Absolute Error"]


def docx_bytes(report):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt
    doc = Document()
    style = doc.styles["Normal"]; style.font.name = "Malgun Gothic"; style.font.size = Pt(10)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic")
    for name in ("Title", "Heading 1", "Heading 2"):
        doc.styles[name].font.name = "Malgun Gothic"
        doc.styles[name].element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic")

    def table(head, rows):
        t = doc.add_table(rows=1, cols=len(head)); t.style = "Table Grid"
        for cell, text in zip(t.rows[0].cells, head):
            cell.text = str(text); cell.paragraphs[0].runs[0].bold = True
        for row in rows:
            for cell, text in zip(t.add_row().cells, row):
                cell.text = str(text)
        return t

    def picture(png, width=16):
        doc.add_picture(BytesIO(png), width=Cm(width)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading(report["title"], level=0)
    doc.add_heading("1. 실험 개요", level=1)
    table(["항목", "내용"], overview_rows(report))
    if report["empty"]:
        doc.add_paragraph("선택 범위에 등록 이미지가 없습니다.")
        return _save(doc)
    doc.add_heading("2. 공정·분석 조건", level=1)
    table(PROCESS_HEAD, process_rows(report))
    doc.add_heading("3. 분석 범위", level=1)
    s = report["scope"]
    table(["항목", "값"], scope_rows(report))
    if s["excluded"]:
        doc.add_paragraph("계측 제외 사유: " + " · ".join(f"{k} {v}장" for k, v in s["excluded"].most_common()))
    doc.add_heading("4. 계측 결과", level=1)
    doc.add_heading("웨이퍼 전체 Line Width 통계", level=2)
    table(WAFER_STATS_HEAD, wafer_stats_rows(report))
    doc.add_paragraph(WAFER_STATS_NOTE)
    doc.add_paragraph(f"설계 Line 폭 {DESIGN_LINE_UM:g} µm 기준 · 평균은 계측 포함(유효) 이미지 평균의 평균, "
                      "측정 수는 그 이미지들의 저장된 Line/Gap 측정 개수 합 · 그래프 오차막대는 이미지 평균 간 표본 표준편차 · "
                      "AI 잠정 결과(공식 채택 아님)")
    if report["kind"] == "Wafer":
        for g in report["groups"]:
            doc.add_heading(f"{g['label']} · Die별", level=2)
            table(MEASUREMENT_HEAD, measurement_rows(g["rows"], "Die"))
    else:
        table(MEASUREMENT_HEAD, measurement_rows(report["rows"], "그룹"))
    for chart in [c for c in report["charts"] if not c.get("wafer_map")]:
        doc.add_paragraph(chart["title"]); picture(chart["png"])
    number = 5
    if "defects" in report["sections"]:
        doc.add_heading(f"{number}. 결함 검출 결과", level=1); number += 1
        table(["Wafer · 조건", "분석 이미지", "결함 검출 이미지", "s", "o", "sc", "p", "pd", "규칙 기반 PD"],
              [[g["label"], str(g["defects"]["analysed"]), str(g["defects"]["images_with_defects"]),
                *[str(g["defects"]["counts"].get(c, 0)) for c in ("s", "o", "sc", "p", "pd")], str(g["defects"]["heuristic_pd"])]
               for g in report["groups"]])
        doc.add_paragraph("모델 후처리 검출 건수 · 결함 0건은 품질 PASS가 아닙니다.")
    if "wafer_map" in report["sections"]:
        doc.add_heading(f"{number}. Wafer Map", level=1); number += 1
        doc.add_paragraph("AI 잠정 위험도 기준 v0.1 (공식 판정 아님) · 회색: 평가 미완료/미평가")
        for chart in [c for c in report["charts"] if c.get("wafer_map")]:
            doc.add_paragraph(chart["title"]); picture(chart["png"], width=11)
    if "images" in report["sections"] and report.get("image"):
        image = report["image"]
        doc.add_heading(f"{number}. 대표 원본 · Overlay", level=1)
        doc.add_paragraph(f"{image['filename']} · {image['location']} · {image['image_id']}")
        for label, key in (("원본", "original"), ("Metrology Overlay", "metrology"), ("Defect Overlay", "defect")):
            if image.get(key):
                doc.add_paragraph(label)
                doc.add_picture(image[key], width=Cm(15)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                doc.add_paragraph(f"{label}: 저장된 이미지 없음")
    return _save(doc)


PROCESS_HEAD = ["Wafer · 조건", "Batch", "노광 (" + EXPOSURE_DISPLAY_UNIT + ")", "노광 단위 (저장)", "PEB", "PEB 단위",
                "Calibration", "분석 실행 시점"]


def process_rows(report):
    return [[g["label"], ", ".join(b[-8:] for b in g["process"]["batches"]), g["process"]["exposure"],
             g["process"]["exposure_unit"], g["process"]["peb"], g["process"]["peb_unit"], g["process"]["calibration"],
             g["process"]["analysed_at"]] for g in report["groups"]]


WAFER_STATS_HEAD = ["Wafer · 조건", "Line Width 평균 (µm)", "Line Width 표준편차 (µm)", "측정 수 N"]
WAFER_STATS_NOTE = ("웨이퍼 전체 = 선택 조건의 모든 Die에서 계측 포함 이미지의 유효 개별 Line Width 값 · "
                    "표준편차는 표본 표준편차(n−1, 프로젝트 기존 정의) · N < 2이면 —")


def wafer_stats_rows(report):
    """One row per Wafer x condition group (single source for the preview and Word)."""
    return [[g["label"], _num(g["stats"]["lw_mean"]), "—" if g["stats"]["lw_std"] is None else f"{g['stats']['lw_std']:.3f}",
             str(g["stats"]["lw_n"])] for g in report["groups"]]


def overview_rows(report):
    rows = [["대상 ID", report["target"]], ["조건 범위", report["condition"]], ["생성일", report["created"]]]
    for label, key in (("작성자", "author"), ("실험 목적", "purpose"), ("특이사항", "notes")):
        if report.get(key):
            rows.append([label, report[key]])
    return rows


def scope_rows(report):
    s = report["scope"]
    return [["등록 이미지", f"{s['registered']}장"], ["분석 결과 이미지", f"{s['analysed']}장"],
            ["계측 포함 이미지 (유효 µm)", f"{s['measured']}장"],
            ["분석 상태 (저장값)", " · ".join(f"{k} {v}" for k, v in sorted(s["outcomes"].items()))],
            ["공식 채택", f"{s['adopted']}장 (is_adopted=1)"]]


def filename(report):
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in report["target"].replace(" ", "_"))
    return f"IDA_report_{safe}_{report['created'][:10]}.docx"


def plain_text(report):
    """Fallback body (paste into Word) when .docx generation is blocked."""
    frames = preview_frames(report); lines = [report["title"], ""]
    for name, frame in (("1. 실험 개요", frames["overview"]), ("2. 공정·분석 조건", frames["process"]),
                        ("3. 분석 범위", frames["scope"])):
        lines += [name, frame.to_csv(sep=chr(9), index=False), ""]
    lines.append("4. 계측 결과")
    lines += ["웨이퍼 전체 Line Width 통계", frames["wafer_stats"].to_csv(sep=chr(9), index=False), WAFER_STATS_NOTE]
    for label, frame in frames["measure"].items():
        lines += [label, frame.to_csv(sep=chr(9), index=False)]
    if "defects" in frames:
        lines += ["", "결함 검출 결과", frames["defects"].to_csv(sep=chr(9), index=False)]
    return chr(10).join(lines)


def _save(doc):
    buffer = BytesIO(); doc.save(buffer)
    return buffer.getvalue()


def preview_frames(report):
    """Tables shown in the preview (same values as the Word tables)."""
    frames = {"overview": pd.DataFrame(overview_rows(report), columns=["항목", "내용"]),
              "scope": pd.DataFrame(scope_rows(report), columns=["항목", "값"])}
    frames["process"] = pd.DataFrame(process_rows(report), columns=PROCESS_HEAD)
    frames["wafer_stats"] = pd.DataFrame(wafer_stats_rows(report), columns=WAFER_STATS_HEAD)
    if report["kind"] == "Wafer":
        frames["measure"] = {g["label"]: pd.DataFrame(measurement_rows(g["rows"], "Die"), columns=MEASUREMENT_HEAD)
                             for g in report["groups"]}
    else:
        frames["measure"] = {"공정 조건별": pd.DataFrame(measurement_rows(report.get("rows", []), "그룹"), columns=MEASUREMENT_HEAD)}
    if "defects" in report["sections"]:
        frames["defects"] = pd.DataFrame([{"Wafer · 조건": g["label"], "분석 이미지": g["defects"]["analysed"],
                                           "결함 검출 이미지": g["defects"]["images_with_defects"],
                                           **{c: g["defects"]["counts"].get(c, 0) for c in ("s", "o", "sc", "p", "pd")},
                                           "규칙 기반 PD": g["defects"]["heuristic_pd"]} for g in report["groups"]])
    return frames
