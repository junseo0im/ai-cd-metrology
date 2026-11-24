"""Record-only defaults from TASK 10. Never imported by analysis engines."""
from copy import deepcopy

STATUSES=("확정","검토 필요","미입력")
NOTICE="이 페이지의 정보는 실험 환경 및 판정 기준 기록용이며, 실제 AI 분석·선폭 계측 코드에 자동 적용되지 않습니다."


def row(identity,item,value="",unit="",status=None,note="",kind="text"):
    return {"id":identity,"item":item,"value":str(value),"unit":unit,
            "status":status or ("확정" if str(value) else "미입력"),"note":note,"kind":kind}


EXPERIMENT=[
("equipment","장비 사양",[
    row("instrument","촬영 장비","광학현미경"),
    row("software","이미지 획득 프로그램","OptiView"),
    row("magnification","사용 배율","20","×",kind="positive"),
    row("model","장비 제조사 및 모델명"),
    row("resolution","이미지 해상도",note="실제 이미지·장비 확인 전 추정하지 않음"),
    row("auto_exposure","자동노출 사용 여부",status="검토 필요"),
    row("lighting","조명 조건"),row("gain","촬영 Gain",kind="nonnegative"),
    row("other","기타 촬영 환경")]),
("structure","Wafer / Die / Pattern 구조",[
    row("layout","Wafer 배열","3행 × 8열"),
    row("dies","Die 수","24","개",kind="count"),
    row("die_codes","Die 위치 번호","11~18, 21~28, 31~38"),
    row("positions","Die당 패턴 위치","lt / c / rb"),
    row("sites","물리적 Site 수","72","개",note="물리 위치 수이며 실제 등록 이미지 수가 항상 72장은 아님",kind="count"),
    row("experiment_unit","실험 시 사용 단위","Row"),
    row("rows","Row 수","3","행",kind="count"),
    row("dies_per_row","Row당 Die 수","8","개",kind="count"),
    row("repeat","동일 Site 반복 촬영","가능"),
    row("multiple_batches","같은 물리적 Wafer의 여러 Batch 등록","가능",note="Batch는 이미지 등록 작업 단위이며 Wafer 자체와 다름")]),
("process","공정 조건",[
    row("exposure","노광 조건","50 / 60 / 70 / 80",status="검토 필요",note="비교 조건 후보; 촬영 완료·정확한 단위·결과 미확인"),
    row("exposure_unit","공정 조건 단위",status="검토 필요"),
    row("pr","PR 종류","Negative PR",status="검토 필요",note="실제 사용 공정과 일치 여부 확인"),
    row("recipe","공정 Recipe"),row("batch_links","공정 조건별 Batch 연결 정보"),
    row("date","실험 날짜"),row("purpose","실험 목적"),row("notes","비고"),
    row("capture_complete","실제 촬영 완료 여부",status="검토 필요")]),
("mask","Mask 설계 / 제작값",[
    row("design_line","설계 Line Width","10","µm",note="설계값; 실제 계측 결과나 제작 실측값이 아님",kind="positive"),
    row("design_gap","설계 Gap",unit="µm",kind="positive"),
    row("design_edge","설계 Edge 관련 기준"),
    row("made_line","Mask 제작 Line Width",unit="µm",kind="positive"),
    row("made_gap","Mask 제작 Gap",unit="µm",kind="positive"),
    row("drawing","설계 도면 또는 자료"),row("variation","제작 편차",note="산정 방식·단위 확인 전 수치 입력하지 않음")]),
("error","오차율 계산 기준(허용값)",[
    row("design_line","설계 기준 선폭","10","µm",kind="positive"),
    row("image_mean","이미지별 평균","해당 이미지에서 유효하게 측정된 모든 Line Width의 산술평균"),
    row("overall_mean","전체 평균","PD 제외 후 유효 이미지별 평균값들의 산술평균"),
    row("pd_exclude","PD 검출 이미지","전체 평균 계산에서 제외"),
    row("same_die","같은 Die의 PD 미검출 이미지","유효 계측 결과가 있으면 평균에 포함"),
    row("measure_count","개별 이미지 기록","측정값 수와 이미지별 평균을 함께 기록"),
    row("image_counts","전체 집계 기록","전체 대상 / 제외 / 유효 이미지 수 구분"),
    row("excluded_ids","제외 이력","제외 이미지 ID 및 제외 사유 기록"),
    row("signed","Signed Error (%)","(측정값 - 설계값) / 설계값 × 100","%"),
    row("absolute","Absolute Error (%)","|측정값 - 설계값| / 설계값 × 100","%"),
    row("calculation","실제 계산 시점","실제 분석 결과 연결 후 계산",note="이 페이지는 계산 방법만 기록하며 계산하지 않음"),
    row("tolerance","허용 오차율",unit="%",status="검토 필요",kind="nonnegative")]),
("calibration","Calibration",[
    row("calibration_id","Calibration ID"),
    row("magnification","사용 배율","20","×",note="촬영 배율 기록; 실제 Calibration 수행·유효성은 별도 확인",kind="positive"),
    row("tool","기준 측정 도구","Stage Micrometer"),
    row("scale_x","X축 Calibration 값",note="실제 값·단위 미입력",kind="positive"),
    row("scale_y","Y축 Calibration 값",note="실제 값·단위 미입력",kind="positive"),
    row("conversion","px/µm 환산값",note="환산 방향·단위(µm/px 또는 px/µm) 확인 필요",kind="positive"),
    row("date","Calibration 수행 날짜"),row("reference","기준 이미지"),
    row("conditions","적용 조건"),row("valid","Calibration 유효 여부"),
    row("remeasure","재측정 필요 여부"),row("notes","비고"),
    row("magnification_change","배율 변경 시","Calibration 재확인"),
    row("reuse","배율 고정 시","동일 Calibration 재사용 가능",note="축별 Scale·왜곡 및 적용 조건 확인 필요"),
    row("lighting_only","조명 변경만 있는 경우","자동 재보정하지 않음"),
    row("axis_check","축별 확인","축별 Scale 및 왜곡 확인 필요")]),
("measurement","계측 방법",[
    row("target","측정 대상","IDA 패턴 Line/Finger"),
    row("y","ROI 기준 Y 위치","25%, 45%, 65%"),
    row("all_lines","측정 범위","각 Y 위치에서 검출되는 모든 Line/Finger"),
    row("image_mean","이미지별 선폭 평균","유효 선폭 측정값의 산술평균"),
    row("raw_unit","측정 원시 단위","px"),
    row("calibrated_unit","Calibration 적용 후 출력 단위","µm"),
    row("design_line","설계 선폭","10","µm",kind="positive"),
    row("order","실행 순서 방향","결함 검출 → 선폭 계측",status="검토 필요",note="방향 기록이며 실제 Pipeline 미연결"),
    row("pd","PD 포함 이미지","계측 및 평균에서 제외"),
    row("failure","실패·제외 구분","계측 실패와 PD 제외를 별개 사유로 기록"),
    row("roi_method","ROI 자동 검출 방법",status="검토 필요"),
    row("edge_method","Edge 검출 알고리즘",status="검토 필요"),
    row("implementation","정확한 구현 세부사항",status="검토 필요")])]

CRITICAL=[
("measurement","측정 Critical 기준",[
    row("design_line","설계 Line Width","10","µm",kind="positive"),
    row("pn","PN 상한(이하)","8","µm",note="8 µm 포함",kind="positive"),
    row("pt","PT 하한(이상)","12","µm",note="12 µm 포함",kind="positive"),
    row("pd_reference","PD 관련 참고 기준(이하)","4","µm",status="검토 필요",note="AI의 PD 검출 결과와 자동으로 같은 의미로 처리하지 않음",kind="positive"),
    row("width_labels","별도 선폭 판정 라벨","pn / pt",note="Defect AI 클래스 목록과 별개"),
    row("pd_source","PD 검출 주체","Defect AI 직접 검출",note="정책 기록이며 실제 엔진 연결은 아님"),
    row("priority","최종 라벨 우선순위",status="검토 필요",note="PN/PT 경계와 PD 참고 기준이 겹칠 수 있음")]),
("defect","결함 Critical 기준",[
    row("classes","Defect AI 클래스 코드","s / o / sc / p / pd",note="실제 팀 인터페이스·학습 데이터의 정의와 확인 필요"),
    row("class_count","현재 코드 수","5","개",kind="count"),
    row("s_meaning","s 상세 의미",status="검토 필요"),
    row("o_meaning","o 상세 의미",status="검토 필요"),
    row("sc_meaning","sc 상세 의미",status="검토 필요"),
    row("p_meaning","p 상세 의미",status="검토 필요"),
    row("pd_meaning","pd 상세 의미",status="검토 필요"),
    row("pd_source","PD 검출 주체","Defect AI 직접 검출"),
    row("pd_exclude","PD 포함 이미지","선폭 계측 및 평균 제외"),
    row("unusable","Die 사용 불가 조건","o / s / pd 중 하나라도 검출된 이미지가 있으면 해당 Die 사용 불가"),
    row("confidence","Defect AI confidence 임계값",status="검토 필요",note="임의 수치를 설정하지 않음",kind="nonnegative"),
    row("importance","클래스별 중요도",status="검토 필요"),
    row("severity","심각도",status="검토 필요")]),
("die_row","Die / Row 판정",[
    row("sites","Die 구성","lt / c / rb 3개 위치"),
    row("die_fail","Die 사용 불가 조건","세 위치 중 한 이미지라도 o / s / pd 검출 시 사용 불가"),
    row("other_images","같은 Die의 PD 미포함 이미지","유효 계측 결과가 있으면 평균에 포함"),
    row("separation","판정과 집계 구분","Die 사용 가능 판정과 이미지별 계측 포함 여부는 별도로 관리"),
    row("layout","Wafer 배열","3행 × 8열"),row("usage","실험 사용 단위","Row"),
    row("row_dies","한 Row의 Die 수","8","개",kind="count"),
    row("row_pass","Row 사용 가능 최소 정상 Die 수","5","개",note="5개 이상이면 사용 가능; 향후 결과 확인 후 변경 가능",kind="count"),
    row("row_fail","Row 사용 불가 최대 정상 Die 수","4","개",note="4개 이하이면 사용 불가; 향후 결과 확인 후 변경 가능",kind="count"),
    row("unknown","미촬영 / 미분석 / 분석 실패","정상으로 임의 분류하지 않음"),
    row("normal_definition","정상 Die의 상세 정의",status="검토 필요"),
    row("incomplete","이미지 부족 시 최종 Row 판정 가능 여부",status="검토 필요"),
    row("exceptions","기타 예외 조건",status="검토 필요")]),
("aggregation","집계 / 제외 정책",[
    row("exclude","계측 제외 기본 조건","Defect AI에서 PD 검출"),
    row("scope","제외 범위","PD가 있는 해당 이미지"),
    row("other_images","같은 Die의 PD 미검출 이미지","유효 계측 결과가 있으면 포함"),
    row("overall_mean","전체 평균 방식","이미지별 평균의 산술평균"),
    row("image_mean","이미지별 평균 방식","해당 이미지의 모든 유효 선폭 측정값의 산술평균"),
    row("total_count","전체 이미지 수","구분하여 표시"),
    row("pd_count","PD 제외 이미지 수","구분하여 표시"),
    row("other_count","기타 사유로 계측하지 못한 이미지 수","구분하여 표시"),
    row("used_count","평균 계산에 실제 사용된 이미지 수","구분하여 표시"),
    row("excluded_ids","제외 이력","제외된 이미지 목록 및 제외 사유 기록"),
    row("version","기록 버전","저장 시 record_version 증가",note="기록 관리용이며 실제 분석 코드에는 자동 적용하지 않음")])]

DOCUMENTS={
    "experiment_conditions":{"title":"실험 환경 및 조건","tabs":EXPERIMENT},
    "critical_rules":{"title":"Critical 기준","tabs":CRITICAL},
}


def definition(document_id):
    if document_id not in DOCUMENTS:
        raise ValueError("알 수 없는 기록 문서입니다.")
    return deepcopy(DOCUMENTS[document_id])
