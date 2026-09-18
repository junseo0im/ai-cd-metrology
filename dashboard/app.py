"""Streamlit entry point for the AI CD Metrology dashboard MVP."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.adapters import (  # noqa: E402
    DashboardAnalysis,
    build_evaluation_record,
    build_method2,
    diagnostic_summary,
    load_dashboard_image,
    local_band_records,
    method_comparison_records,
    run_method2_analysis,
)
from dashboard.data_loader import (  # noqa: E402
    DATASET_SCOPES,
    catalog_for_scope,
    curated_catalog,
    filter_catalog,
    load_dataset_inventory,
    overview_snapshot,
)
from dashboard.visualization import analysis_overlay, image_to_rgb  # noqa: E402


@st.cache_data(show_spinner=False)
def _inventory() -> pd.DataFrame:
    return load_dataset_inventory(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def _method2():
    return build_method2(PROJECT_ROOT)


def _analysis_cache() -> dict[str, DashboardAnalysis]:
    if "method2_analysis_cache" not in st.session_state:
        st.session_state.method2_analysis_cache = {}
    return st.session_state.method2_analysis_cache


def _status_message(status: str, detail: str | None = None) -> None:
    text = status if not detail else f"{status}: {detail}"
    if status == "VALID":
        st.success(text)
    elif status == "WARNING":
        st.warning(text)
    elif status == "FAIL":
        st.error(text)
    else:
        st.info(text)


def _display_value(value: object, *, pending: str = "Pending") -> object:
    if value is None or value == "":
        return pending
    return value


def _dataset_display_label(row: pd.Series) -> str:
    """Expand a known curated filename abbreviation for display only."""

    label = str(row["defect_label"])
    source = str(row["source"])
    if label.lower() == "p" and source.endswith("::FAIL/particle"):
        return "particle"
    return label


def _catalog_selector(catalog: pd.DataFrame, prefix: str) -> pd.Series | None:
    if catalog.empty:
        st.warning("No curated images are available.")
        return None
    source = st.selectbox("Source", sorted(catalog["source"].unique()), key=f"{prefix}_source")
    filtered = filter_catalog(catalog, source=source)
    wafer = st.selectbox("Wafer", sorted(filtered["wafer_id"].unique()), key=f"{prefix}_wafer")
    filtered = filter_catalog(filtered, wafer_id=wafer)
    die = st.selectbox("Die", sorted(filtered["die_id"].unique()), key=f"{prefix}_die")
    filtered = filter_catalog(filtered, die_id=die)
    position = st.selectbox(
        "Pattern position",
        sorted(filtered["pattern_position"].unique()),
        key=f"{prefix}_position",
    )
    filtered = filter_catalog(filtered, pattern_position=position)
    image_id = st.selectbox(
        "Image ID", sorted(filtered["image_id"].unique()), key=f"{prefix}_image"
    )
    st.session_state.selected_image_id = image_id
    return filtered.loc[filtered["image_id"].eq(image_id)].iloc[0]


def _render_overview(inventory: pd.DataFrame) -> None:
    snapshot = overview_snapshot(inventory)
    st.subheader("Dataset")
    columns = st.columns(4)
    columns[0].metric("Total images", snapshot["total_image_count"])
    columns[1].metric("Curated images", snapshot["curated_image_count"])
    columns[2].metric("Known curated wafers", snapshot["wafer_count"])
    columns[3].metric("Official calibration", "PENDING")

    left, middle, right = st.columns(3)
    with left:
        st.caption("Source count")
        st.dataframe(
            pd.DataFrame(
                snapshot["source_counts"].items(), columns=["source_root", "count"]
            ),
            hide_index=True,
            width="stretch",
        )
    with middle:
        st.caption("Curated label status")
        st.dataframe(
            pd.DataFrame(
                snapshot["status_counts"].items(), columns=["status", "count"]
            ),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.caption("Curated pattern position")
        st.dataframe(
            pd.DataFrame(
                snapshot["pattern_position_counts"].items(),
                columns=["pattern_position", "count"],
            ),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Analysis flow")
    flow = pd.DataFrame(
        [
            ("Input Image", "READY"),
            ("Metadata", "READY"),
            ("Method 2", "READY / FROZEN"),
            ("Calibration", "PENDING"),
            ("Manual Reference", "PENDING"),
            ("Defect", "NOT AVAILABLE"),
            ("Unified Result", "PARTIAL"),
            ("Wafer aggregation", "FUTURE"),
        ],
        columns=["Stage", "Status"],
    )
    st.dataframe(flow, hide_index=True, width="stretch")

    st.subheader("System status")
    st.dataframe(
        pd.DataFrame(
            [
                ("Method 1", "NOT AVAILABLE"),
                ("Method 2", "READY / FROZEN"),
                ("Method 3", "NOT AVAILABLE"),
                ("Calibration", "PENDING"),
                ("Manual reference", "PENDING"),
                ("Defect detection", "NOT AVAILABLE"),
                ("Defect classification", "NOT AVAILABLE"),
                ("Wafer aggregation", "SKELETON"),
                ("What-if simulator", "SKELETON"),
            ],
            columns=["Component", "Availability"],
        ),
        hide_index=True,
        width="stretch",
    )

    research, operational = st.columns(2)
    with research:
        st.markdown("**Research view**")
        st.write("CD px: available after selected-image analysis")
        st.write("Manual reference / accuracy: pending")
        st.write("Method comparison: Method 2 only")
        st.write("Runtime: available after selected-image analysis")
    with operational:
        st.markdown("**Operational view**")
        st.write("Wafer / die / pattern identity: available where metadata exists")
        st.write("Status: available after selected-image analysis")
        st.write("review_required / critical_flag: not defined")
        st.write("acquisition / inspection traceability: pending")


def _result_record_table(record: dict[str, object]) -> pd.DataFrame:
    groups = {
        "Identity": ["image_id", "wafer_id", "die_id", "pattern_position", "source"],
        "Method": ["method_id", "method_version"],
        "Measurement": ["outer_px", "inner_px", "gap_px", "outer_um", "inner_um", "gap_um"],
        "Reference / Accuracy": [
            "manual_outer_um", "manual_inner_um", "manual_gap_um",
            "outer_absolute_error_um", "inner_absolute_error_um", "gap_absolute_error_um",
        ],
        "Reliability / Traceability": [
            "measurement_status", "failure_reason", "calibration_id", "runtime_ms",
        ],
        "Operational extension": [
            "review_required", "critical_flag", "inspection_timestamp",
            "algorithm_version", "acquisition_id",
        ],
    }
    rows = []
    for group, fields in groups.items():
        for field in fields:
            value = record.get(field)
            if field.endswith("_um") and value is None:
                value = "Calibration / Reference pending"
            elif value is None:
                value = "Pending / Not defined"
            rows.append({"Group": group, "Field": field, "Value": str(value)})
    return pd.DataFrame(rows)


def _render_image_analysis(inventory: pd.DataFrame) -> None:
    st.subheader("Selected image analysis")
    scope = st.radio(
        "Dataset scope",
        DATASET_SCOPES,
        horizontal=True,
        key="ia_dataset_scope",
    )
    catalog = catalog_for_scope(inventory, scope)
    selector_columns = st.columns(5)
    with selector_columns[0]:
        source_options = sorted(catalog["source"].unique())
        preferred_source = "CURATED_IMAGES::PASS_normal"
        source = st.selectbox(
            "Source",
            source_options,
            index=source_options.index(preferred_source) if preferred_source in source_options else 0,
            key="ia_source",
        )
    filtered = filter_catalog(catalog, source=source)
    with selector_columns[1]:
        wafer = st.selectbox("Wafer", sorted(filtered["wafer_id"].unique()), key="ia_wafer")
    filtered = filter_catalog(filtered, wafer_id=wafer)
    with selector_columns[2]:
        die = st.selectbox("Die", sorted(filtered["die_id"].unique()), key="ia_die")
    filtered = filter_catalog(filtered, die_id=die)
    with selector_columns[3]:
        position = st.selectbox(
            "Pattern position", sorted(filtered["pattern_position"].unique()), key="ia_position"
        )
    filtered = filter_catalog(filtered, pattern_position=position)
    with selector_columns[4]:
        image_id = st.selectbox("Image ID", sorted(filtered["image_id"].unique()), key="ia_image")
    row = filtered.loc[filtered["image_id"].eq(image_id)].iloc[0]
    dataset_label = _dataset_display_label(row)
    st.session_state.selected_image_id = image_id

    if scope == "Full inventory" and not bool(row["is_curated"]):
        st.caption(
            "Non-curated / evaluation data. Running analysis exposes this image result; "
            "do not use it for subsequent Method 2 tuning if it is reserved for untouched validation."
        )

    try:
        image = load_dashboard_image(row)
    except (FileNotFoundError, ValueError) as exc:
        image = None
        image_error = str(exc)
    else:
        image_error = None

    can_run = image is not None and row["pattern_position"] != "unknown"
    analysis = _analysis_cache().get(str(row["file_path"]))
    original, summary = st.columns([3, 2], gap="large")
    with original:
        st.markdown("### Original Image")
        if image is None:
            st.error(f"Image loading failed: {image_error}")
        else:
            st.image(image_to_rgb(image), width="stretch")

    with summary:
        st.markdown("### Analysis Summary")
        st.write(f"**Image ID:** {row['image_id']}")
        st.markdown("**Dataset / Inspection**")
        st.write(f"**{row['status_label']} / {dataset_label}**")
        st.caption("Defect status and CD measurement validity are independent.")
        st.divider()
        st.markdown("**Method 2 Measurement Validity**")
        if analysis is None:
            st.info("NOT RUN")
            st.caption("No measurement result is shown until this selected image is analyzed.")
        else:
            result = analysis.result
            _status_message(result.status.value, result.failure_reason)
            st.metric(
                "Outer Width",
                "Unavailable" if result.outer_width_px is None else f"{result.outer_width_px:g} px",
            )
            st.metric(
                "Inner Width",
                "Unavailable" if result.inner_width_px is None else f"{result.inner_width_px:g} px",
            )
            st.metric(
                "Gap",
                "Unavailable" if result.gap_px is None else f"{result.gap_px:g} px",
            )
            st.write("µm: **Calibration pending**")
            st.write(
                f"Runtime: {result.runtime_ms:.2f} ms"
                if result.runtime_ms is not None
                else "Runtime: unavailable"
            )
            st.write(f"Calibration ID: {result.calibration_id or 'Pending'}")

        if st.button("Run Method 2 Analysis", disabled=not can_run):
            try:
                with st.spinner("Running selected-image measurement..."):
                    analysis = run_method2_analysis(row, _method2())
                    _analysis_cache()[str(row["file_path"])] = analysis
                st.rerun()
            except (ValueError, FileNotFoundError) as exc:
                st.error(f"Method 2 could not analyze this image: {exc}")
        st.caption("Runs the frozen Gradient/Edge metrology for the selected image only.")

    with st.expander("Image Metadata"):
        st.table(
            pd.DataFrame(
                {
                    "Field": [
                        "image_id", "wafer_id", "die_id", "pattern_position",
                        "source", "capture_region", "resolution", "ground_truth_label",
                    ],
                    "Value": [
                        row["image_id"], row["wafer_id"], row["die_id"],
                        row["pattern_position"], row["source"], row["capture_region"],
                        row["resolution"], dataset_label,
                    ],
                }
            ),
            hide_index=True,
        )

    if analysis is not None:
        result = analysis.result
        with st.expander("Measurement Overlay / Traceability"):
            st.image(
                analysis_overlay(analysis.image, analysis.diagnostic, result),
                width="stretch",
            )
            st.caption(
                "The ROI, five-band boundaries, measurement x direction, and representative candidate "
                "E1–E6 x positions show where the image was analyzed. This overlay is diagnostic evidence; "
                "the Analysis Summary is the source of truth for image-level median Outer/Inner/Gap values. "
                "The displayed E1–E6 y extent is visual only, and the stored y coordinate is an ROI-center "
                "placeholder from the 1D median-y profile."
            )

        with st.expander("Reliability / Local Diagnostic"):
            st.dataframe(
                pd.DataFrame([diagnostic_summary(analysis.diagnostic)]),
                hide_index=True,
                width="stretch",
            )
            st.markdown("**Five-band local diagnostic**")
            st.caption(
                "Local-band results are diagnostic only and do not modify the whole-image measurement."
            )
            st.dataframe(
                pd.DataFrame(local_band_records(analysis.diagnostic)),
                hide_index=True,
                width="stretch",
            )

        with st.expander("Structured Result / Traceability"):
            record = build_evaluation_record(analysis)
            st.dataframe(_result_record_table(record), hide_index=True, width="stretch")

    with st.expander("Dataset / Session Results"):
        table = catalog[[
            "image_id", "source", "wafer_id", "die_id", "pattern_position",
            "status_label", "defect_label",
        ]].copy()
        cache = _analysis_cache()
        status_by_id = {
            bundle.result.image_id: bundle.result.status.value for bundle in cache.values()
        }
        table["method"] = table["image_id"].map(
            lambda value: "GRADIENT_CANNY" if value in status_by_id else "NOT RUN"
        )
        table["measurement_status"] = table["image_id"].map(
            lambda value: status_by_id.get(value, "NOT RUN")
        )
        table_status = st.selectbox(
            "Measurement status filter",
            ["All"] + sorted(table["measurement_status"].unique()),
            key="result_status_filter",
        )
        table_method = st.selectbox(
            "Method filter",
            ["All"] + sorted(table["method"].unique()),
            key="result_method_filter",
        )
        if table_status != "All":
            table = table.loc[table["measurement_status"].eq(table_status)]
        if table_method != "All":
            table = table.loc[table["method"].eq(table_method)]
        st.dataframe(table, hide_index=True, width="stretch", height=320)


def _render_method_comparison() -> None:
    selected_id = st.session_state.get("selected_image_id")
    analyses = list(_analysis_cache().values())
    analysis = next(
        (item for item in analyses if item.result.image_id == selected_id),
        analyses[-1] if analyses else None,
    )
    if analysis is None:
        st.info("Run Method 2 for a selected image in Image Analysis to populate the comparison.")
        record = None
    else:
        st.write(f"Selected result: **{analysis.result.image_id}**")
        record = build_evaluation_record(analysis)
    st.dataframe(
        pd.DataFrame(method_comparison_records(record)),
        hide_index=True,
        width="stretch",
    )
    st.caption("Method 1 and Method 3 have no result adapters yet. No placeholder measurement is generated.")


def _render_defect(catalog: pd.DataFrame) -> None:
    st.subheader("Defect Analysis")
    st.caption("Detection and classification algorithms are not implemented in this MVP.")
    image_id = st.selectbox("Image", sorted(catalog["image_id"].unique()), key="defect_image")
    row = catalog.loc[catalog["image_id"].eq(image_id)].iloc[0]
    left, right = st.columns([2, 1])
    try:
        image = load_dashboard_image(row)
        left.image(image_to_rgb(image), caption="Original image", width="stretch")
    except (FileNotFoundError, ValueError) as exc:
        left.error(f"Image loading failed: {exc}")
    right.table(
        pd.DataFrame(
            [
                ("Ground-truth label", row["defect_label"]),
                ("Label status", row["status_label"]),
                ("Detection", "Not available"),
                ("Classification prediction", "Not available"),
            ],
            columns=["Field", "Value"],
        ),
        hide_index=True,
    )


def _render_wafer(catalog: pd.DataFrame) -> None:
    st.subheader("Wafer Analysis")
    known = catalog.loc[catalog["wafer_id"].ne("unknown")]
    if known.empty:
        st.info("Wafer identity is not available.")
        return
    wafer = st.selectbox("Wafer ID", sorted(known["wafer_id"].unique()), key="wafer_tab_id")
    wafer_rows = known.loc[known["wafer_id"].eq(wafer)]
    die_summary = (
        wafer_rows.groupby("die_id", as_index=False)
        .agg(image_count=("image_id", "count"), pattern_count=("pattern_position", "nunique"))
        .sort_values("die_id")
    )
    st.markdown("**Wafer Summary → Die Table**")
    st.dataframe(die_summary, hide_index=True, width="stretch")
    die = st.selectbox("Die ID", sorted(wafer_rows["die_id"].unique()), key="wafer_tab_die")
    pattern_rows = wafer_rows.loc[wafer_rows["die_id"].eq(die), [
        "image_id", "pattern_position", "status_label", "defect_label", "source"
    ]]
    st.markdown("**Pattern Result Metadata**")
    st.dataframe(pattern_rows, hide_index=True, width="stretch")
    st.info("Physical die coordinates are not available. No geometric wafer map is inferred.")


def _render_simulator() -> None:
    st.subheader("Experimental / Simulation")
    st.info("Not implemented in MVP")
    st.write("Future scenario inputs may include width/gap variation, blur, noise, rotation, illumination, edge loss and particle occlusion.")
    st.write("Simulation outputs will remain separate from measured production results.")


def _render_process_hypothesis() -> None:
    st.subheader("Process Hypothesis / Future Research")
    st.write("Required future inputs: wafer pattern, calibrated CD variation, defect distribution and known process metadata.")
    st.write("Possible structure: observed data → exploratory association → testable hypothesis → independent validation.")
    st.info("Causal AI, root-cause prediction and process diagnosis are not implemented.")


def main() -> None:
    st.set_page_config(page_title="AI CD Metrology", layout="wide")
    st.title("AI CD Metrology Dashboard MVP")
    st.caption("Selected-image analysis and structured result inspection. No batch metrology is run automatically.")

    inventory = _inventory()
    catalog = curated_catalog(inventory)
    if inventory.empty or catalog.empty:
        st.error("Dataset inventory or curated image catalog is unavailable.")
        return

    tabs = st.tabs(
        [
            "Overview",
            "Image Analysis",
            "Method Comparison",
            "Defect Analysis",
            "Wafer Analysis",
            "What-if Simulator",
            "Process Hypothesis",
        ]
    )
    with tabs[0]:
        _render_overview(inventory)
    with tabs[1]:
        _render_image_analysis(inventory)
    with tabs[2]:
        _render_method_comparison()
    with tabs[3]:
        _render_defect(catalog)
    with tabs[4]:
        _render_wafer(catalog)
    with tabs[5]:
        _render_simulator()
    with tabs[6]:
        _render_process_hypothesis()


if __name__ == "__main__":
    main()
