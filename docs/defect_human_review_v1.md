# Defect Human Review v1

## 기록 범위: pre-annotation history

아래 HIGH 32장 판정과 priority queue는 annotation 이전 기록으로 보존한다. 현재 CVAT Pilot 상태는 [최신 completed review](../data/metadata/defect_annotation_pilot_review_v1_completed_draft.csv)를 우선하며 현재 운영 기준은 [Guideline v2](yolov8_seg_annotation_guideline_v2.md)다. 기존 표/판정을 소급 수정하지 않는다.

현재 달라진 Pilot 판정: P009 (`w3_14_c`)는 particle 6개의 DEFECT, P014/P015는 confirmed particle을 유지한 image-level HOLD, P016은 short + particle, P025는 NORMAL이다. 아래 “최종” 또는 “이번 freeze”는 당시 pre-annotation review 시점을 뜻하며 현재 annotation 판정과 구분한다.

## Review queue policy

- HIGH: defect-vs-non-defect uncertainty, source-label/appearance conflict, compound class, scratch/background anomaly, severe acquisition/crop issue, or non-training reference/evaluation asset.
- MEDIUM: class/polygon boundary confirmation, tiny object, border contact, or representative class-edge case.
- LOW: visually clear normal/defect or remaining unresolved unlabeled/`nc`/particle pool after higher-risk cases. LOW does not mean automatically approved.
- Unique contents marked `needs_review`: **733**. Priority table below lists **70 HIGH/MEDIUM representatives**; remaining review-required unique contents stay discoverable in `defect_inventory_v1.csv` via `needs_review=True`.

## Human adjudication status

- HIGH 32개 고유 콘텐츠는 사용자가 원본을 직접 확인해 판정했다.
- v1 training eligibility: **KEEP 26 / EXCLUDE 5 / TBD 1**.
- 동일 SHA-256 복사본에는 같은 판정을 적용했다. Manifest에서는 HIGH 32개가 42개 경로에 대응한다.
- MEDIUM/LOW 항목은 이번 freeze에서 판정하지 않았다.
- 기존 `observed_issue`와 source-label provenance는 유지하고, 아래 세 열에 최종 판정을 추가했다.

## Priority cases

| priority | visual_index | source_label | relative_path | observed_issue | decision_needed | human_decision | training_eligibility | final_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HIGH | 23 | pattern_die | images/images/FAIL/pattern_die/스크린샷 2026-08-17 195420.png | screenshot/UI/export framing; acquisition-native status not established | decide defect-training scope and class; inspect full resolution | 실제 결함: 패턴 폭이 과도하게 두꺼움 | KEEP | excessive pattern width; nearly connected; human-approved KEEP despite screenshot source |
| HIGH | 31 | pattern_narrowing | images/images/FAIL/pattern_narrowing/스크린샷 2026-08-17 200155.png | screenshot/UI/export framing; acquisition-native status not established | decide defect-training scope and class; inspect full resolution | 길이 측정 비교용 이미지 | EXCLUDE | measurement comparison image; minor dust present; not defect training |
| HIGH | 100 | normal | images/images/PASS_normal/w2_14_c.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | 극소 scratch 경계 사례 | TBD | candidate=scratch; retain as tiny-object-rule borderline case; no fixed pixel threshold |
| HIGH | 137 | normal | images/images/PASS_normal/w2_32_lt.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | scratch와 edge particle 모두 결함 | KEEP | classes=scratch,particle; annotate as separate instances |
| HIGH | 161 | normal | images/images/PASS_normal/w3_14_c.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | 정상 | KEEP | normal repeating pattern; no defect polygon |
| HIGH | 169 | normal | images/images/PASS_normal/w3_17_c.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | edge에 붙은 이물질 | KEEP | particle attached to pattern edge |
| HIGH | 178 | normal | images/images/PASS_normal/w3_22_lt.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | edge에 붙은 이물질 | KEEP | particle present; remaining pattern is normal |
| HIGH | 230 | open+pattern_die | 결함 총합/결함 총합/60_wafer 결함만/E60-F/2-1(2)_o, pd.png | unique-content visual review index 0230; compound filename label; object/class boundaries must be reviewed | decide defect-training scope and class; inspect full resolution | 한 이미지에 두 결함 | KEEP | do not create a compound class; annotate each visible defect separately |
| HIGH | 233 | deformation_unknown+particle | 결함 총합/결함 총합/60_wafer 결함만/E60-F/2-8(2)_d, p.png | unique-content visual review index 0233; compound filename label; object/class boundaries must be reviewed | decide defect-training scope and class; inspect full resolution | 기존 두 이상과 wafer 횡단 scratch | KEEP | preserve source labels; include scratch as separate visible instance |
| HIGH | 234 | deformation_unknown | 결함 총합/결함 총합/60_wafer 결함만/E60-F/2-8(3)_d.png | unique-content visual review index 0234; pattern/acquisition anomaly visible but physical class is not identifiable | decide defect-training scope and class; inspect full resolution | deformation과 wafer 횡단 scratch | KEEP | classes=deformation,scratch; separate instances |
| HIGH | 237 | deformation_unknown+pattern_narrowing+particle | 결함 총합/결함 총합/60_wafer 결함만/E60-F/3-6(2)_d, pn, p.png | unique-content visual review index 0237; compound filename label; object/class boundaries must be reviewed | decide defect-training scope and class; inspect full resolution | deformation, narrowing, particle 복합 결함 | KEEP | difficulty=HARD; separate visible instances; unclear instance boundaries may remain UNCERTAIN |
| HIGH | 277 | unlabeled | 결함 총합/결함 총합/노광 60/wafer1/w1_13_c.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | edge에 붙은 이물질 | KEEP | particle attached to pattern edge |
| HIGH | 300 | unlabeled | 결함 총합/결함 총합/노광 60/wafer1/w1_22_rb.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | 정상 | KEEP | normal repeating pattern; no defect polygon |
| HIGH | 302 | unlabeled | 결함 총합/결함 총합/노광 60/wafer1/w1_23_lt.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | 검출 대상 scratch | KEEP | linear surface mark; annotate only visible scratch region |
| HIGH | 312 | unlabeled | 결함 총합/결함 총합/노광 60/wafer1/w1_26_rb.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | 실제 defect, 세부 class 미정 | KEEP | defect confirmed by human; do not invent a class before annotation review |
| HIGH | 321 | pattern_narrowing+particle | 결함 총합/결함 총합/노광 60/wafer1/w1_31_rb_pn, p.png | unique-content visual review index 0321; exact content appears in 2 raw paths; compound filename label; object/class boundaries must be reviewed | decide defect-training scope and class; inspect full resolution | 여러 particle과 배경 scratch | KEEP | classes=particle,scratch; multi_instance=YES; do not split particle by filled/hollow appearance |
| HIGH | 330 | unlabeled | 결함 총합/결함 총합/노광 60/wafer1/w1_34_rb.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | scratch/spot은 검출 대상 | KEEP | candidate=scratch and/or particle; resolve instances during annotation |
| HIGH | 341 | unlabeled | 결함 총합/결함 총합/노광 60/wafer1/w1_38_lt.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | 결함으로 판정, 세부 class 미정 | KEEP | defect confirmed by human; no arbitrary class assignment |
| HIGH | 437 | unlabeled | 결함 총합/결함 총합/노광 60/wafer3/w3_18_lt.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | scratch/spot이 결함 | KEEP | candidate=scratch and/or particle; resolve instances during annotation |
| HIGH | 460 | deformation_unknown | 결함 총합/결함 총합/노광 60/wafer3/w3_28_c_d.png | unique-content visual review index 0460; pattern/acquisition anomaly visible but physical class is not identifiable | decide defect-training scope and class; inspect full resolution | 실제 pattern deformation | KEEP | physical pattern defect confirmed by human |
| HIGH | 477 | unlabeled | 결함 총합/결함 총합/노광 60/wafer3/w3_35_rb_ld1.png | unique-content visual review index 0477; pattern/acquisition anomaly visible but physical class is not identifiable | decide defect-training scope and class; inspect full resolution | 여러 이물질이 한꺼번에 붙은 사례 | KEEP | multi_instance=YES; difficulty=HARD; no new cluster class |
| HIGH | 480 | particle+deformation_unknown | 결함 총합/결함 총합/노광 60/wafer3/w3_36_rb_p, d.png | unique-content visual review index 0480; compound filename label; object/class boundaries must be reviewed | decide defect-training scope and class; inspect full resolution | particle과 deformation 각각 존재 | KEEP | classes=particle,deformation; separate instances |
| HIGH | 493 | deformation_unknown | 결함 총합/결함 총합/노광 60/wafer4/w4_13_c_d.png | severe partial-field/crop or blank-like field; defect interpretability limited | decide defect-training scope and class; inspect full resolution | pattern edge가 두꺼움 | KEEP | pattern_thickening confirmed against neighboring repeating pattern |
| HIGH | 532 | deformation_unknown | 결함 총합/결함 총합/노광 60/wafer4/w4_28_c_d.png | severe partial-field/crop or blank-like field; defect interpretability limited | decide defect-training scope and class; inspect full resolution | pattern edge가 두꺼움 | KEEP | pattern_thickening confirmed against neighboring repeating pattern |
| HIGH | 556 | deformation_unknown | 결함 총합/결함 총합/노광 60/wafer4/w4_38_c_d.png | severe partial-field/crop or blank-like field; defect interpretability limited | decide defect-training scope and class; inspect full resolution | 패턴이 제대로 찍히지 않음 | EXCLUDE | human decision gate=physical process defect vs acquisition-quality issue; excluded from v1 training without forcing a class |
| HIGH | 557 | deformation_unknown | 결함 총합/결함 총합/노광 60/wafer4/w4_38_lt_d.png | severe partial-field/crop or blank-like field; defect interpretability limited | decide defect-training scope and class; inspect full resolution | 제외 | EXCLUDE | excluded by human adjudication |
| HIGH | 558 | deformation_unknown | 결함 총합/결함 총합/노광 60/wafer4/w4_38_rb_d.png | severe partial-field/crop or blank-like field; defect interpretability limited | decide defect-training scope and class; inspect full resolution | 패턴이 제대로 찍히지 않은 실제 defect | KEEP | physical pattern failure; v1 canonical mapping=deformation |
| HIGH | 691 | short+particle | 결함 총합/결함 총합/노광 70/wafer2/w2_35_c_s, p.png | unique-content visual review index 0691; exact content appears in 2 raw paths; compound filename label; object/class boundaries must be reviewed | decide defect-training scope and class; inspect full resolution | short와 particle이 각각 존재 | KEEP | classes=short,particle; separate polygons |
| HIGH | 798 | pattern_die+pattern_narrowing | 결함 총합/결함 총합/노광 70/wafer4/w4_18_rb_pd, pn.png | unique-content visual review index 0798; exact content appears in 2 raw paths; compound filename label; object/class boundaries must be reviewed | decide defect-training scope and class; inspect full resolution | pattern deformation과 narrowing이 분리됨 | KEEP | classes=deformation,pattern_narrowing; separate polygons; pattern_die remains source label only |
| HIGH | 815 | unlabeled | 결함 총합/결함 총합/노광 70/wafer4/w4_26_lt.png | visible linear surface mark/scratch or local spot | decide defect-training scope and class; inspect full resolution | scratch/spot이 실제 defect | KEEP | candidate=scratch and/or particle; resolve instances during annotation |
| HIGH | 919 | unlabeled | optiview_ref_image.png | manual ruler/reference overlay; not a defect-training specimen | decide defect-training scope and class; inspect full resolution | reference asset 제외 | EXCLUDE | reference asset; no annotation |
| HIGH | 920 | unlabeled | team_protocol_compare/target_01.png | evaluation-only target; acquisition-native scale metadata unconfirmed | decide defect-training scope and class; inspect full resolution | CD 길이 측정 평가/비교 전용 | EXCLUDE | CD measurement evaluation asset; no annotation |
| MEDIUM | 2 | particle | images/images/FAIL/particle/particle1 (1).png | particle may be surface contamination rather than physical pattern damage | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 3 | particle | images/images/FAIL/particle/particle1 (2).png | particle may be surface contamination rather than physical pattern damage | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 4 | particle | images/images/FAIL/particle/particle1 (3).png | particle may be surface contamination rather than physical pattern damage | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 212 | nc | images/images/WARNING_nc/w1_11_lt_nc.png | nc meaning and acquisition-vs-defect boundary not defined | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 213 | nc | images/images/WARNING_nc/w1_18_lt_nc.png | nc meaning and acquisition-vs-defect boundary not defined | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 214 | nc | images/images/WARNING_nc/w1_32_rb_nc.png | nc meaning and acquisition-vs-defect boundary not defined | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 218 | nc | images/images/WARNING_nc/w2_16_lt_nc.png | nc meaning and acquisition-vs-defect boundary not defined | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 220 | nc | images/images/WARNING_nc/w3_24_lt_nc.png | nc meaning and acquisition-vs-defect boundary not defined | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 223 | nc | images/images/WARNING_nc/w3_34_lt_nc.png | nc meaning and acquisition-vs-defect boundary not defined | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 252 | pattern_die | 70 2/E70 F/1-8_1_pd.png | unique-content visual review index 0252; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 282 | nc | 결함 총합/결함 총합/노광 60/wafer1/w1_14_rb_nc.png | nc meaning and acquisition-vs-defect boundary not defined | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 309 | nc | 결함 총합/결함 총합/노광 60/wafer1/w1_25_rb_nc.png | nc meaning and acquisition-vs-defect boundary not defined | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 335 | nc | 결함 총합/결함 총합/노광 60/wafer1/w1_36_lt_nc.png | nc meaning and acquisition-vs-defect boundary not defined | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 347 | pattern_die | 결함 총합/결함 총합/노광 60/wafer2/w2_12_lt_pd.png | unique-content visual review index 0347 | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 366 | pattern_narrowing | 결함 총합/결함 총합/노광 60/wafer2/w2_18_rb_pn.png | unique-content visual review index 0366 | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 367 | pattern_die | 결함 총합/결함 총합/노광 60/wafer2/w2_21_c_pd.png | unique-content visual review index 0367 | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 392 | pattern_narrowing | 결함 총합/결함 총합/노광 60/wafer2/w2_31_lt_pn.png | unique-content visual review index 0392 | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 407 | short | 결함 총합/결함 총합/노광 60/wafer2/w2_36_lt_s.png | unique-content visual review index 0407 | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 418 | pattern_die | 결함 총합/결함 총합/노광 60/wafer3/w3_12_c_pd.png | unique-content visual review index 0418 | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 419 | pattern_narrowing | 결함 총합/결함 총합/노광 60/wafer3/w3_12_lt_pn.png | unique-content visual review index 0419 | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 438 | pattern_die | 결함 총합/결함 총합/노광 60/wafer3/w3_18_rb_pd.png | unique-content visual review index 0438 | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 449 | particle | 결함 총합/결함 총합/노광 60/wafer3/w3_24_lt_p.png | particle may be surface contamination rather than physical pattern damage | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 490 | pattern_die | 결함 총합/결함 총합/노광 60/wafer4/w4_12_c_pd.png | unique-content visual review index 0490 | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 563 | pattern_die | 결함 총합/결함 총합/노광 70/wafer1/w1_12_lt_pd.png | unique-content visual review index 0563; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 573 | pattern_narrowing | 결함 총합/결함 총합/노광 70/wafer1/w1_15_rb_pn.png | unique-content visual review index 0573; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 582 | pattern_die | 결함 총합/결함 총합/노광 70/wafer1/w1_18_rb_pd.png | unique-content visual review index 0582; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 635 | pattern_die | 결함 총합/결함 총합/노광 70/wafer2/w2_12_lt_pd.png | unique-content visual review index 0635; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 654 | pattern_die | 결함 총합/결함 총합/노광 70/wafer2/w2_18_rb_pd.png | unique-content visual review index 0654; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 681 | pattern_narrowing | 결함 총합/결함 총합/노광 70/wafer2/w2_31_rb_pn.png | unique-content visual review index 0681; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 707 | pattern_die | 결함 총합/결함 총합/노광 70/wafer3/w3_12_lt_pd.png | unique-content visual review index 0707; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 726 | pattern_die | 결함 총합/결함 총합/노광 70/wafer3/w3_18_rb_pd.png | unique-content visual review index 0726; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 751 | pattern_narrowing | 결함 총합/결함 총합/노광 70/wafer3/w3_31_c_pn.png | unique-content visual review index 0751; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 753 | pattern_narrowing | 결함 총합/결함 총합/노광 70/wafer3/w3_31_rb_pn.png | unique-content visual review index 0753; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 778 | pattern_narrowing | 결함 총합/결함 총합/노광 70/wafer4/w4_12_c_pn.png | unique-content visual review index 0778; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 779 | pattern_die | 결함 총합/결함 총합/노광 70/wafer4/w4_12_lt_pd .png | unique-content visual review index 0779; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 820 | nc | 결함 총합/결함 총합/노광 70/wafer4/w4_28_c_nc.png | nc meaning and acquisition-vs-defect boundary not defined | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 851 | pattern_die | 결함 총합/결함 총합/노광 70/wafer5/w5_12_lt_pd.png | unique-content visual review index 0851; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |
| MEDIUM | 870 | pattern_die | 결함 총합/결함 총합/노광 70/wafer5/w5_18_rb_pd.png | unique-content visual review index 0870; exact content appears in 2 raw paths | confirm visual class and polygon feasibility at full resolution | — | — | 이번 freeze에서 미판정 |

## Contact sheets

- `data/derived/defect_taxonomy_review_v1/human_review_contact_sheets/human_review_01.jpg`
- `data/derived/defect_taxonomy_review_v1/human_review_contact_sheets/human_review_02.jpg`
- `data/derived/defect_taxonomy_review_v1/human_review_contact_sheets/human_review_03.jpg`
- `data/derived/defect_taxonomy_review_v1/human_review_contact_sheets/human_review_04.jpg`

The 16 exhaustive unique-content sheets remain at `data/derived/defect_taxonomy_review_v1/all_unique_contact_sheets/`. Contact sheets are for review only; they are resized derivatives and must not be used as training images or annotations.

## Required decisions

1. visual index 100의 극소 scratch가 반복 가능한 polygon 경계를 갖는지 최종 판정한다.
2. visual index 556은 physical/process 문제와 acquisition 문제를 구분할 근거가 부족하므로 v1 학습에서 제외한 상태로 보존한다.
3. visual index 312, 330, 341, 437, 815는 defect 존재는 확인됐지만 instance별 세부 class를 annotation 전에 확정해야 한다.
4. MEDIUM/LOW 사례는 이번 freeze 범위에서 판정하지 않았다.
