"""Shared MLflow display boundary. No API client, metrics or model execution."""
from dashboard.components.layout import placeholder


def render_mlflow():
    placeholder("MLflow · 미연결",
                "최근 Run · 모델 버전 · 모델 비교 · Precision / Recall / mAP 표시 예정입니다. 실제 API 호출이나 성능지표는 없습니다.")


# The model-selection page keeps render_mlflow() above until it is explicitly connected.
# This opt-in browser is used only by the model-performance page.
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import mlflow_provider as provider


@st.cache_data(ttl=60, max_entries=128, show_spinner=False)
def fetch(connection, operation, identifier="", limit=50, token="", path="", revision=0):
    client = provider.ReadonlyClient(connection)
    if operation == "experiments":
        return client.experiments()
    if operation == "runs":
        return client.runs(identifier, limit, token)
    if operation == "all_runs":
        return client.all_runs(identifier)
    if operation == "run":
        return client.run(identifier)
    if operation == "artifacts":
        return client.artifacts(identifier, path, token)
    raise provider.MLflowError("허용되지 않은 조회입니다.")


def _pager(namespace, context):
    state_key = "_mf_" + namespace + "_pages"
    if st.session_state.get(state_key, {}).get("context") != context:
        st.session_state[state_key] = {"context": context, "tokens": [""]}
    return st.session_state[state_key]["tokens"]


def _page_buttons(namespace, tokens, next_token):
    back, caption, forward = st.columns([1, 2, 1])
    if back.button("이전", key="mf_" + namespace + "_previous", disabled=len(tokens) == 1):
        tokens.pop()
        st.rerun()
    caption.caption(f"{len(tokens)} 페이지 · 비교 범위는 현재 조회된 Run입니다." if namespace == "runs"
                    else f"{len(tokens)} 페이지")
    if forward.button("다음", key="mf_" + namespace + "_next",
                      disabled=not next_token or next_token in tokens):
        tokens.append(next_token)
        st.rerun()


def _kv(title, values, *, numeric=False):
    st.markdown("**" + title + "**")
    if values:
        st.dataframe(pd.DataFrame([
            {"항목": str(key), "값": value if numeric else str(value)}
            for key, value in sorted(values.items())
        ]), hide_index=True, width="stretch")
    else:
        st.caption("미기록")


def _render_artifacts(connection, selected_run, revision):
    rid = provider.run_id(selected_run)
    state = st.session_state
    if state.get("_mf_artifact_run") != rid:
        state["_mf_artifact_run"], state["_mf_artifact_path"] = rid, ""
    path = state.get("_mf_artifact_path", "")
    st.markdown("**Artifact 목록**")
    st.caption("목록만 조회합니다. 파일 다운로드·내용 분석·가중치 로딩은 수행하지 않습니다.")
    st.text("경로: " + (path or "/"))
    if path and st.button("상위 폴더", key="mf_artifact_parent"):
        state["_mf_artifact_path"] = path.rsplit("/", 1)[0] if "/" in path else ""
        st.rerun()
    tokens = _pager("artifacts", (rid, path, revision))
    try:
        response = fetch(connection, "artifacts", rid, token=tokens[-1], path=path, revision=revision)
    except provider.MLflowError as error:
        st.warning(str(error))
        return
    files = response.get("files", [])
    if not files:
        st.info("조회된 Artifact가 없습니다.")
    else:
        st.dataframe(pd.DataFrame([{
            "파일명": str(item.get("path", "")).rsplit("/", 1)[-1],
            "경로": item.get("path", ""),
            "형식": "폴더" if item.get("is_dir") else (
                str(item.get("path", "")).rsplit(".", 1)[-1] if "." in str(item.get("path", "")) else "확장자 없음"),
            "크기 (bytes)": item.get("file_size") if not item.get("is_dir") else None,
        } for item in files]), hide_index=True, width="stretch")
        folders = [str(item["path"]) for item in files if item.get("is_dir") and item.get("path")]
        if folders:
            folder = st.selectbox("Artifact 폴더", [""] + folders,
                                  format_func=lambda value: value or "폴더를 선택하세요",
                                  key="mf_artifact_folder_" + rid + "_" + path)
            if st.button("폴더 열기", key="mf_artifact_open", disabled=not folder):
                state["_mf_artifact_path"] = folder
                st.rerun()
    _page_buttons("artifacts", tokens, response.get("next_page_token", ""))


def _render_run_detail(connection, experiment, runs, labels, revision, scope):
    options = [provider.run_id(run) for run in runs if provider.run_id(run)]
    run_key = "mf_run_" + scope
    if st.session_state.get(run_key) not in options:
        st.session_state.pop(run_key, None)
    rid = st.selectbox("Run 선택", options, format_func=labels.get, key=run_key)
    try:
        run = fetch(connection, "run", rid, revision=revision)
    except provider.MLflowError as error:
        st.warning(str(error))
        return
    if not run or str(run.get("info", {}).get("experiment_id")) != str(experiment["experiment_id"]):
        st.warning("선택한 Experiment의 Run 상세정보를 확인할 수 없습니다.")
        return
    info = run.get("info", {})
    st.text("Experiment: " + experiment["name"])
    st.text("Run 이름: " + provider.run_name(run))
    st.code(rid, language=None)
    st.caption("상태: " + str(info.get("status") or "미기록") +
               " · 시작: " + provider.recorded_time(info.get("start_time")) +
               " · 종료: " + provider.recorded_time(info.get("end_time")) + " · UTC")
    parameters, metrics, tags = st.tabs(["Parameters", "Metrics", "Tags"])
    with parameters:
        _kv("기록된 Parameter", provider.pairs(run, "params"))
    with metrics:
        _kv("기록된 Metric", provider.pairs(run, "metrics"), numeric=True)
    with tags:
        _kv("기록된 Tag", provider.pairs(run, "tags"))
    _render_artifacts(connection, run, revision)



def _default_metric(keys, experiment_name):
    """Choose a display default, never rank models or combine different keys."""
    import re
    stages = {"test": 0, "testing": 0, "val": 1, "valid": 1, "validation": 1,
              "eval": 2, "evaluation": 2, "train": 3, "training": 3}
    name = experiment_name.lower()
    if "width" in name or "metrology" in name:
        preferred = ("mae", "error", "rmse", "recall", "precision")
    elif "classification" in name:
        preferred = ("macro_f1", "f1", "recall", "accuracy", "precision")
    else:
        preferred = ("map", "macro_f1", "f1", "recall", "precision", "accuracy", "mae")
    def rank(key):
        lower = key.lower()
        tokens = re.split(r"[^a-z0-9]+", lower)
        stage = next((stages[token] for token in tokens if token in stages), 2)
        score = next((index for index, word in enumerate(preferred) if word in lower), len(preferred))
        return stage, score, lower
    return min(keys, key=rank) if keys else None


def _comparison_figure(runs, metric):
    from html import escape
    labels = provider.run_labels(runs)
    points = []
    for run in runs:
        value = provider.metric_value(run, metric)
        if value is None:
            continue
        name, rid = provider.run_name(run), provider.run_id(run)
        short = name if len(name) <= 24 else name[:23] + "…"
        points.append({"Run": escape(short) + "<br>" + escape(rid[:8]),
                       "값": value, "전체 Run 이름": escape(name), "전체 Run ID": rid})
    if not points:
        return None
    # A unique category per run also handles identical run names.
    categories = [item["Run"] for item in points]
    if len(categories) != len(set(categories)):
        for item, run in zip(points, [r for r in runs if provider.metric_value(r, metric) is not None]):
            item["Run"] = escape(labels[provider.run_id(run)])
    figure = px.bar(pd.DataFrame(points), x="Run", y="값",
                    custom_data=["전체 Run 이름", "전체 Run ID"], labels={"값": metric})
    figure.update_traces(texttemplate="%{y:.6g}", textposition="outside", cliponaxis=False,
                         hovertemplate="Run: %{customdata[0]}<br>ID: %{customdata[1]}<br>"
                         + escape(metric) + ": %{y:.12g}<extra></extra>")
    figure.update_layout(showlegend=False, height=320, margin=dict(l=8, r=8, t=28, b=8),
                         xaxis_title=None, yaxis_title=metric)
    figure.update_yaxes(rangemode="tozero", zeroline=True, automargin=True)
    figure.update_xaxes(automargin=True, tickangle=0)
    return figure


def _comparison_table(runs, metrics):
    labels = provider.run_labels(runs)
    rows = []
    for run in runs:
        params = provider.pairs(run, "params")
        rows.append({
            "Run": labels[provider.run_id(run)],
            **{metric: provider.metric_value(run, metric) for metric in metrics},
            "Parameter 요약": " · ".join(f"{key}={value}" for key, value in list(sorted(params.items()))[:2]) or "-",
            "상태": run.get("info", {}).get("status") or "-",
        })
    return pd.DataFrame(rows)


def _comparison_guidance(experiment_name, key):
    if experiment_name == "IDA_Width_Measurement":
        if key == "test/mae_um":
            return "기록된 Test MAE 비교이며 일반적으로 낮을수록 오차가 작습니다. 전체 모델 우열을 자동 판정하지 않습니다."
        if key == "test/runtime_median_ms":
            return "기록된 처리시간 비교이며 낮을수록 빠릅니다. 정확도와는 별개의 지표입니다."
    return "선택한 Metric의 Run별 기록값 비교입니다. 지표 정의를 확인해야 하며 학습 지표만으로 최종 모델 성능을 확정할 수 없습니다."



def _metric_description(experiment_name, key):
    descriptions = {
        "IDA_Defect_Classification": {
            "train/accuracy": "학습 데이터에서 기록된 정확도",
            "val/best_macro_f1": "검증 데이터에서 기록된 최고 Macro F1",
        },
        "IDA_Width_Measurement": {
            "test/mae_um": "기준 측정값 대비 평균 절대오차(µm)",
            "test/runtime_median_ms": "이미지당 처리시간 중앙값(ms)",
            "stability/success_rate": "측정 행 기준 성공 비율(해당 실험에 한함)",
        },
    }
    description = descriptions.get(experiment_name, {}).get(key)
    if description:
        return description
    if experiment_name == "IDA_Width_Measurement" and key in {
        "stability/repeat_target_sd_um", "stability/lighting_image_mean_range_um"
    }:
        return provider.metric_note(experiment_name, key)
    return "상세 정의 미등록"


def _mlflow_run_url(connection, run):
    """Only the current DagsHub MLflow web route was verified in a browser."""
    import re
    if connection is None or connection.tracking_uri != provider.DEFAULT_TRACKING_URI:
        return None
    experiment_id = str(run.get("info", {}).get("experiment_id", ""))
    rid = provider.run_id(run)
    if not re.fullmatch(r"[0-9]+", experiment_id) or not re.fullmatch(r"[0-9a-fA-F]{32}", rid):
        return None
    return connection.tracking_uri + "/#/experiments/" + experiment_id + "/runs/" + rid


def _metric_counts(runs, metric):
    recorded = sum(provider.metric_value(run, metric) is not None for run in runs)
    missing = sum(metric not in provider.pairs(run, "metrics") for run in runs)
    return recorded, missing, len(runs) - recorded - missing


def _comparison_display(runs, metrics, connection):
    # Keep numeric data/sorting intact; only the rendered missing-cell text changes.
    frame = _comparison_table(runs, metrics)
    if metrics:
        selected = metrics[0]
        urls = [_mlflow_run_url(connection, run)
                if selected not in provider.pairs(run, "metrics") else None for run in runs]
        if any(urls):
            frame["MLflow 열기"] = urls
    styled = frame.style
    if metrics:
        styled = styled.format({metric: str for metric in metrics}, subset=metrics, na_rep="-")
        for metric in metrics:
            invalid = [index for index, run in enumerate(runs)
                       if metric in provider.pairs(run, "metrics") and provider.metric_value(run, metric) is None]
            if invalid:
                styled = styled.format(subset=pd.IndexSlice[invalid, [metric]], na_rep="유효하지 않은 값")
    return styled


def _connection_and_experiments(revision):
    try:
        settings = {"mlflow": dict(st.secrets.get("mlflow", {}))}
    except (FileNotFoundError, KeyError):
        settings = {}
    connection = provider.connection_from_settings(settings)
    return connection, fetch(connection, "experiments", revision=revision)


HIGHLIGHT = "background-color: rgba(53,168,107,.22); font-weight: 700"


def _render_comparison(experiment_name, runs):
    """TASK18-B-19: the experiment's default comparison table, every evaluated model/condition at once."""
    from dashboard.components import mlflow_tables
    table = mlflow_tables.build(experiment_name, runs)
    frame = table["frame"]
    if frame.empty:
        st.info("비교할 평가 Run이 없습니다.")
        return table
    best = set(table["best"])
    styled = frame.style.apply(lambda column: [HIGHLIGHT if (index, column.name) in best else ""
                                               for index in range(len(column))], axis=0)
    config = {column: st.column_config.TextColumn(column, help=text) for column, text in table["help"].items()
              if column in frame.columns}
    st.dataframe(styled, hide_index=True, width="stretch", key="mf_default_table", column_config=config,
                 height=min(35 * (len(frame) + 1) + 3, 560))
    for note in table["notes"]:
        st.caption(note.replace("굵게:", "강조:"))
    return table


def render_mlflow_browser():
    """Experiment 선택 → 모델 비교표 → MLflow 상세 조회 (TASK18-B-19). Read-only over the cached provider."""
    experiment_control, refresh_control = st.columns([4, 1], vertical_alignment="bottom")
    with refresh_control:
        if st.button("새로 조회", key="mf_refresh", width="stretch"):
            st.session_state["_mf_revision"] = st.session_state.get("_mf_revision", 0) + 1
    revision = st.session_state.get("_mf_revision", 0)
    try:
        with st.spinner("실험 목록 조회 중"):
            connection, experiments = _connection_and_experiments(revision)
    except provider.MLflowError as error:
        st.warning("조회 실패 · " + str(error))
        st.caption("서버 관리자: Tracking URI·네트워크·읽기 권한 확인 (인증은 서버 환경변수 또는 secrets).")
        return
    if not experiments:
        st.info("조회 가능한 Experiment가 없습니다.")
        return
    by_id = {str(experiment["experiment_id"]): experiment for experiment in experiments}
    if st.session_state.get("mf_experiment") not in by_id:
        st.session_state.pop("mf_experiment", None)
    with experiment_control:
        exp_id = st.selectbox("Experiment 선택", list(by_id),
                              format_func=lambda value: by_id[value]["name"], key="mf_experiment")
    experiment = by_id[exp_id]
    try:
        with st.spinner("Run 조회 중"):
            everything = fetch(connection, "all_runs", exp_id, revision=revision)
    except provider.MLflowError as error:
        st.warning("조회 실패 · " + str(error))
        return
    runs = [run for run in everything.get("runs", [])
            if str(run.get("info", {}).get("experiment_id")) == exp_id and provider.run_id(run)]
    runs.sort(key=lambda run: run.get("info", {}).get("start_time") or 0, reverse=True)
    if not runs:
        st.info("이 Experiment에 기록된 Run이 없습니다.")
        table = None
    else:
        table = _render_comparison(experiment["name"], runs)
        if everything.get("truncated"):
            st.caption(f"Run이 많아 최근 {len(runs)}개까지 표시합니다.")
    with st.expander("MLflow 상세 조회", expanded=False, key="mf_details_" + exp_id, on_change="rerun") as details:
        if details.open:
            _render_details(connection, experiments, experiment, exp_id, revision, table)


def _render_details(connection, experiments, experiment, exp_id, revision, table):
    """The former browser (metric selector, graph, extra metrics, raw keys, Run detail), collapsed by default."""
    tab_names = ["Metric 비교", "Run 상세정보"]
    extra = (table or {}).get("extra", {})
    # Per-class AP@50 (O·S·PD) is shown in the main comparison table; no separate class tab here.
    if extra.get("connection"):
        tab_names.append("연결 확인 Run")
    tab_names.append("조회 정보")
    tabs = dict(zip(tab_names, st.tabs(tab_names)))
    with tabs["조회 정보"]:
        st.caption("Tracking Server: " + connection.tracking_uri + " · 조회 캐시 최대 60초")
        st.text("Experiment: " + experiment["name"] + " · ID " + exp_id)
        st.caption(f"조회 가능한 활성 Experiment {len(experiments)}개")
        limit = st.selectbox("페이지당 Run 수", [25, 50, 100, 200], index=1, key="mf_limit")
    tokens = _pager("runs", (exp_id, limit, revision))
    try:
        response = fetch(connection, "runs", exp_id, limit, tokens[-1], revision=revision)
    except provider.MLflowError as error:
        st.warning("조회 실패 · " + str(error))
        return
    runs = [run for run in response.get("runs", [])
            if str(run.get("info", {}).get("experiment_id")) == exp_id and provider.run_id(run)]
    runs.sort(key=lambda run: run.get("info", {}).get("start_time") or 0, reverse=True)
    with tabs["조회 정보"]:
        if runs:
            statuses = pd.Series([run.get("info", {}).get("status") or "미기록" for run in runs]).value_counts()
            st.caption("실행 상태: " + ", ".join(f"{key} {value}" for key, value in statuses.items()))
        _page_buttons("runs", tokens, response.get("next_page_token", ""))
    if "연결 확인 Run" in tabs:
        with tabs["연결 확인 Run"]:
            st.dataframe(pd.DataFrame([{"Run 이름": provider.run_name(run), "Run ID": provider.run_id(run),
                                        "기록 Metric": ", ".join(sorted(provider.pairs(run, "metrics"))) or "-"}
                                       for run in extra["connection"]]), hide_index=True, width="stretch")
    if not runs:
        with tabs["Metric 비교"]:
            st.info("이 페이지에 조회된 Run이 없습니다.")
        return
    labels = provider.run_labels(runs)
    scope = f"{exp_id}_{len(tokens)}_{limit}_{revision}"
    metric_key = "mf_metric_" + scope
    keys = sorted({key for run in runs for key in provider.pairs(run, "metrics")})
    with tabs["Metric 비교"]:
        st.caption(f"현재 페이지 Run {len(runs)}개 · 원본 Metric Key 그대로 표시")
        key = None
        if keys:
            if st.session_state.get(metric_key) not in keys:
                st.session_state[metric_key] = _default_metric(keys, experiment["name"])
            key = st.selectbox("비교 Metric", keys, key=metric_key)
            st.caption(_metric_description(experiment["name"], key))
            st.caption(provider.metric_note(experiment["name"], key))
            figure = _comparison_figure(runs, key)
            recorded, missing, invalid = _metric_counts(runs, key)
            st.caption(f"기록값 {recorded}개 · 미기록 Run {missing}개 · 유효하지 않은 값 {invalid}개 · 미기록은 0으로 대체하지 않음")
            if figure is not None:
                st.plotly_chart(figure, width="stretch", key="mf_metric_chart",
                                config={"displayModeBar": False, "scrollZoom": False, "responsive": True})
            else:
                st.info("선택한 Metric의 유효한 수치가 없습니다.")
        else:
            st.info("조회된 Run에 기록된 Metric이 없습니다.")
        table_keys = []
        if key is not None:
            extra_options = [candidate for candidate in keys if candidate != key]
            table_key = "mf_table_metrics_" + scope
            if table_key in st.session_state:
                st.session_state[table_key] = [value for value in st.session_state[table_key]
                                               if value in extra_options][:7]
            extra_metrics = st.multiselect("추가 비교 Metric", extra_options, max_selections=7, key=table_key)
            table_keys = [key] + extra_metrics
        st.dataframe(_comparison_display(runs, table_keys, connection), hide_index=True, width="stretch",
                     key="mf_comparison_table",
                     column_config={"Run": st.column_config.TextColumn("Run 이름", width="large"),
                                    "MLflow 열기": st.column_config.LinkColumn("MLflow", display_text="미기록 · MLflow에서 확인")})
    with tabs["Run 상세정보"]:
        _render_run_detail(connection, experiment, runs, labels, revision, scope)
