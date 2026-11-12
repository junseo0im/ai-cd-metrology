# Defect Taxonomy Review v1

## Scope and method

- Scope: every supported image recursively present under `data/raw/` at audit time.
- Total files: **1,379**; readable: **1,379**; unique SHA-256 contents: **920**.
- Exact duplicate groups: **459**; files participating in duplicate groups: **918**; redundant copies beyond one per group: **459**.
- Visual review: all **920 unique contents** were inspected in 16 contact-sheet pages. Thumbnail review is sufficient for triage, not for final tiny-defect or polygon decisions.
- These are provisional review categories, not canonical training labels. No raw image was changed, resized, moved, or renamed.

Two images are newer than the prior 1,377-file four-root inventory and are included here: `optiview_ref_image.png` and `team_protocol_compare/target_01.png`. They are reference/evaluation assets, not assumed defect-training samples.

## Provisional taxonomy

| source_label | observed_visual_pattern | provisional_category | candidate_final_class | keep/merge/exclude/TBD | reason | count | needs_human_review |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal | regular repeating lines; a few visible scratch/spot conflicts | NORMAL (except flagged conflicts) | normal | KEEP/TBD | keep clean cases; review visible-surface conflicts | 178 | True |
| open | line break/opening or disrupted connection | PHYSICAL_PATTERN_DEFECT | open | KEEP | structural discontinuity is visually evident in representative cases | 25 | False |
| short | bridge/short between neighboring pattern features | PHYSICAL_PATTERN_DEFECT | short | KEEP | structural bridge is visually evident; border cases still need polygon guidance | 20 | True |
| pattern_narrowing | localized or extended narrowing/shape thinning | PHYSICAL_PATTERN_DEFECT | pattern_narrowing | KEEP/TBD | class boundary against deformation and acquisition blur requires agreement | 65 | True |
| pattern_die | dark/weak/deformed repeated pattern; often low contrast | PHYSICAL_PATTERN_DEFECT or ACQUISITION_QUALITY_ISSUE | pattern_die (TBD definition) | TBD | some cases resemble exposure/contrast failure rather than a local object | 37 | True |
| particle | localized spots/objects from tiny specks to large occlusions | SURFACE_OR_BACKGROUND_ARTIFACT | particle (TBD inclusion) | TBD | must decide whether contamination is a target defect class | 139 | True |
| nc | mixed weak spots, scratches, local shape changes and visually near-normal fields | UNCERTAIN | TBD: define nc | TBD | label semantics are not self-evident and visual consistency is low | 333 | True |
| deformation_unknown | large partial-field, diagonal line, deformation or blank-like views | UNCERTAIN / ACQUISITION_QUALITY_ISSUE | TBD | physical defect vs crop/acquisition failure is unresolved | 7 | True | None |
| compound labels | two or more source labels in one image | UNCERTAIN | multi-label TBD | TBD | must define multi-object/multi-class annotation and precedence | 21 | True |
| unlabeled | mostly regular patterns mixed with scratches/spots and two reference/evaluation assets | UNCERTAIN | TBD | absence of a label cannot be treated as normal | 554 | True | None |

## Provisional category balance

| category | file_count | unique_count |
| --- | --- | --- |
| ACQUISITION_QUALITY_ISSUE | 7 | 7 |
| NORMAL | 173 | 173 |
| PHYSICAL_PATTERN_DEFECT | 153 | 99 |
| SURFACE_OR_BACKGROUND_ARTIFACT | 159 | 105 |
| UNCERTAIN | 887 | 536 |

## Existing source-label inventory

Counts retain every path. `unique_count` de-duplicates only exact SHA-256 content.

| source_label | file_count | unique_count | resolution_types | top_resolutions | representatives | needs_review_unique |
| --- | --- | --- | --- | --- | --- | --- |
| deformation_unknown | 7 | 7 | 7 | 2302x1251 (1); 2295x1260 (1); 2298x1246 (1); 2293x1254 (1); 2302x1254 (1) | 결함 총합/결함 총합/60_wafer 결함만/E60-F/2-8(3)_d.png; 결함 총합/결함 총합/노광 60/wafer3/w3_28_c_d.png; 결함 총합/결함 총합/노광 60/wafer4/w4_13_c_d.png | 7 |
| deformation_unknown+particle | 1 | 1 | 1 | 2299x1255 (1) | 결함 총합/결함 총합/60_wafer 결함만/E60-F/2-8(2)_d, p.png | 1 |
| deformation_unknown+pattern_narrowing+particle | 1 | 1 | 1 | 2292x1251 (1) | 결함 총합/결함 총합/60_wafer 결함만/E60-F/3-6(2)_d, pn, p.png | 1 |
| nc | 333 | 209 | 146 | 2296x1251 (9); 2296x1252 (9); 2305x1252 (8); 2299x1252 (7); 2302x1249 (7) | images/images/WARNING_nc/w1_11_lt_nc.png; images/images/WARNING_nc/w1_18_lt_nc.png; images/images/WARNING_nc/w1_32_rb_nc.png | 209 |
| normal | 178 | 178 | 87 | 2304x1257 (7); 2305x1259 (6); 2306x1258 (6); 2304x1258 (5); 2306x1257 (5) | images/images/PASS_normal/w1_11_c.png; images/images/PASS_normal/w1_11_rb.png; images/images/PASS_normal/w1_12_c.png | 5 |
| open | 25 | 14 | 12 | 2302x1248 (4); 2299x1254 (4); 2295x1260 (2); 2299x1249 (2); 2296x1255 (2) | images/images/FAIL/open/open.png; 결함 총합/결함 총합/60_wafer 결함만/E60-F/1-2(2)_o.png; 결함 총합/결함 총합/70 wafer 결함만/E70 F/1-1_1_o.png | 0 |
| open+particle | 2 | 1 | 1 | 2296x1252 (2) | 결함 총합/결함 총합/70 wafer 결함만/E70 F/1-1_3_o, p,.png | 1 |
| open+pattern_die | 1 | 1 | 1 | 2290x1248 (1) | 결함 총합/결함 총합/60_wafer 결함만/E60-F/2-1(2)_o, pd.png | 1 |
| particle | 139 | 92 | 77 | 2302x1255 (8); 2296x1248 (4); 2337x1279 (4); 2299x1252 (4); 2338x1285 (3) | images/images/FAIL/particle/particle1 (1).png; images/images/FAIL/particle/particle1 (2).png; images/images/FAIL/particle/particle1 (3).png | 92 |
| particle+deformation_unknown | 1 | 1 | 1 | 2304x1255 (1) | 결함 총합/결함 총합/노광 60/wafer3/w3_36_rb_p, d.png | 1 |
| particle+pattern_narrowing | 2 | 1 | 1 | 2337x1289 (2) | 결함 총합/결함 총합/노광 70/wafer1/w1_18_c_p, pn.png | 1 |
| pattern_die | 37 | 25 | 24 | 2307x1254 (2); 2298x1255 (2); 2296x1249 (2); 2302x1251 (2); 2336x1285 (2) | images/images/FAIL/pattern_die/w1_12_lt_pd.png; images/images/FAIL/pattern_die/w1_18_rb_pd.png; images/images/FAIL/pattern_die/w2_12_lt_pd.png | 25 |
| pattern_die+open | 1 | 1 | 1 | 2304x1260 (1) | 결함 총합/결함 총합/60_wafer 결함만/E60-F/3-2(2)_pd, o.png | 1 |
| pattern_die+pattern_narrowing | 2 | 1 | 1 | 2296x1251 (2) | 결함 총합/결함 총합/노광 70/wafer4/w4_18_rb_pd, pn.png | 1 |
| pattern_narrowing | 65 | 42 | 40 | 2302x1251 (4); 2306x1258 (2); 2313x1258 (2); 2293x1261 (2); 2338x1289 (2) | images/images/FAIL/pattern_narrowing/w1_12_rb_pn.png; images/images/FAIL/pattern_narrowing/w2_12_c_pn.png; images/images/FAIL/pattern_narrowing/w2_31_rb_pn.png | 42 |
| pattern_narrowing+particle | 4 | 2 | 2 | 2298x1252 (2); 2301x1252 (2) | 결함 총합/결함 총합/노광 60/wafer1/w1_31_rb_pn, p.png; 결함 총합/결함 총합/노광 70/wafer5/w5_12_c_pn, p.png | 2 |
| short | 20 | 15 | 13 | 2298x1251 (4); 2299x1258 (2); 2302x1251 (2); 2293x1248 (2); 2302x1260 (2) | images/images/FAIL/short/short.png; images/images/FAIL/short/w2_35_c_s.png; 결함 총합/결함 총합/60_wafer 결함만/E60-F/1-2(3)_s.png | 15 |
| short+open | 2 | 1 | 1 | 2296x1252 (2) | 결함 총합/결함 총합/70 wafer 결함만/E70 F/3-8_2_s, o.png | 1 |
| short+particle | 2 | 1 | 1 | 2337x1285 (2) | 결함 총합/결함 총합/노광 70/wafer2/w2_35_c_s, p.png | 1 |
| short+pattern_narrowing | 2 | 1 | 1 | 2305x1251 (2) | 결함 총합/결함 총합/70 wafer 결함만/E70 F/1-2_4_s, pn.png | 1 |
| unlabeled | 554 | 325 | 189 | 2296x1251 (15); 2296x1252 (13); 2302x1249 (10); 2305x1258 (8); 2296x1249 (8) | optiview_ref_image.png; team_protocol_compare/target_01.png; 결함 총합/결함 총합/노광 60/wafer1/w1_11_c.png | 325 |

## Visual consistency and overlap findings

- `open` and `short` representatives usually show a structural discontinuity/bridge and are the most visually direct physical-pattern candidates.
- `pattern_narrowing` is recognizable in many cases, but overlaps with broader deformation and weak/low-contrast imaging.
- `pattern_die` is not visually homogeneous: strong dark/deformed pattern cases coexist with weak, low-contrast and screenshot/export cases.
- `particle` ranges from tiny specks to large occlusions. It should not be equated automatically with physical pattern damage.
- `nc` is visually heterogeneous and includes small spots, scratches, mild pattern changes, and near-normal fields. Its meaning must be defined by the team.
- `unlabeled` is not equivalent to normal. The reviewed pool contains apparent normal patterns, linear scratches/surface marks, spots, and non-training reference assets.
- Compound filename labels appear in **13 unique contents** and require a multi-label policy; a single-class collapse would discard source information.
- PASS/normal visual conflicts were flagged at visual indices 100, 137, 161, 169, 178; these show visible linear marks or spots and require human scope decisions.

## Resolution and source leakage risk

- Distinct resolutions: **352**.
- Resolutions associated with only one source-label value: **214**, covering **420 files / 269 unique contents**.
- With hundreds of crop sizes, a model can learn acquisition/export/source signatures instead of defect morphology. Screenshot/export examples and rare resolutions amplify this risk.
- Exact duplicate groups are cross-copy provenance risks. Splits must be assigned by `file_hash`/duplicate group before any augmentation; otherwise identical pixels can enter train and validation/test.
- Wafer/die/capture adjacency should also be grouped during splitting to reduce near-duplicate process leakage.
- Do not resize raw files. Model-time letterbox/resize belongs only in the later training pipeline after class and split policy approval.

### Complete resolution distribution

| resolution | file_count | unique_content_count |
| --- | --- | --- |
| 2296x1252 | 30 | 19 |
| 2296x1251 | 27 | 17 |
| 2302x1249 | 23 | 13 |
| 2299x1252 | 20 | 13 |
| 2299x1251 | 19 | 14 |
| 2301x1252 | 18 | 11 |
| 2296x1249 | 17 | 10 |
| 2299x1254 | 16 | 11 |
| 2302x1255 | 16 | 10 |
| 2296x1257 | 15 | 8 |
| 2302x1248 | 15 | 9 |
| 2298x1252 | 14 | 7 |
| 2302x1251 | 14 | 9 |
| 2305x1252 | 14 | 10 |
| 2299x1248 | 13 | 9 |
| 2302x1252 | 13 | 9 |
| 2298x1251 | 12 | 9 |
| 2304x1257 | 12 | 11 |
| 2301x1255 | 11 | 8 |
| 2304x1252 | 11 | 8 |
| 2293x1248 | 10 | 6 |
| 2296x1248 | 10 | 6 |
| 2296x1255 | 10 | 6 |
| 2304x1260 | 10 | 9 |
| 2305x1258 | 10 | 6 |
| 2293x1251 | 9 | 6 |
| 2296x1254 | 9 | 6 |
| 2298x1255 | 9 | 6 |
| 2299x1255 | 9 | 7 |
| 2299x1257 | 9 | 7 |
| 2304x1255 | 9 | 7 |
| 2338x1285 | 9 | 5 |
| 2293x1246 | 8 | 5 |
| 2298x1246 | 8 | 5 |
| 2301x1254 | 8 | 5 |
| 2302x1254 | 8 | 8 |
| 2305x1251 | 8 | 5 |
| 2305x1254 | 8 | 7 |
| 2306x1258 | 8 | 8 |
| 2330x1283 | 8 | 4 |
| 2331x1280 | 8 | 4 |
| 2335x1282 | 8 | 4 |
| 2335x1285 | 8 | 4 |
| 2337x1284 | 8 | 4 |
| 2293x1252 | 7 | 4 |
| 2298x1254 | 7 | 5 |
| 2299x1249 | 7 | 6 |
| 2299x1258 | 7 | 6 |
| 2301x1257 | 7 | 6 |
| 2302x1258 | 7 | 5 |
| 2304x1253 | 7 | 7 |
| 2304x1258 | 7 | 6 |
| 2305x1255 | 7 | 6 |
| 2307x1255 | 7 | 6 |
| 2293x1254 | 6 | 5 |
| 2293x1257 | 6 | 3 |
| 2295x1255 | 6 | 5 |
| 2296x1245 | 6 | 4 |
| 2298x1249 | 6 | 4 |
| 2299x1243 | 6 | 5 |
| 2301x1251 | 6 | 5 |
| 2302x1257 | 6 | 5 |
| 2304x1251 | 6 | 4 |
| 2305x1259 | 6 | 6 |
| 2305x1260 | 6 | 5 |
| 2306x1255 | 6 | 6 |
| 2307x1251 | 6 | 4 |
| 2331x1288 | 6 | 3 |
| 2332x1286 | 6 | 3 |
| 2333x1284 | 6 | 3 |
| 2334x1280 | 6 | 3 |
| 2335x1280 | 6 | 3 |
| 2336x1287 | 6 | 3 |
| 2337x1280 | 6 | 3 |
| 2337x1285 | 6 | 3 |
| 2337x1289 | 6 | 3 |
| 2338x1284 | 6 | 3 |
| 2339x1284 | 6 | 3 |
| 2341x1287 | 6 | 3 |
| 2293x1243 | 5 | 3 |
| 2295x1249 | 5 | 4 |
| 2298x1257 | 5 | 3 |
| 2301x1249 | 5 | 3 |
| 2301x1258 | 5 | 4 |
| 2305x1257 | 5 | 4 |
| 2306x1256 | 5 | 5 |
| 2306x1257 | 5 | 5 |
| 2295x1251 | 4 | 2 |
| 2298x1248 | 4 | 4 |
| 2301x1260 | 4 | 3 |
| 2303x1255 | 4 | 4 |
| 2303x1257 | 4 | 4 |
| 2303x1258 | 4 | 4 |
| 2303x1259 | 4 | 4 |
| 2304x1254 | 4 | 3 |
| 2304x1256 | 4 | 4 |
| 2304x1259 | 4 | 4 |
| 2305x1246 | 4 | 2 |
| 2305x1253 | 4 | 4 |
| 2325x1286 | 4 | 2 |
| 2327x1277 | 4 | 2 |
| 2327x1286 | 4 | 2 |
| 2329x1285 | 4 | 2 |
| 2330x1282 | 4 | 2 |
| 2332x1278 | 4 | 2 |
| 2333x1283 | 4 | 2 |
| 2333x1286 | 4 | 2 |
| 2333x1289 | 4 | 2 |
| 2334x1283 | 4 | 2 |
| 2334x1285 | 4 | 2 |
| 2334x1291 | 4 | 2 |
| 2335x1287 | 4 | 2 |
| 2335x1289 | 4 | 2 |
| 2336x1281 | 4 | 2 |
| 2336x1282 | 4 | 2 |
| 2336x1283 | 4 | 2 |
| 2336x1288 | 4 | 2 |
| 2336x1290 | 4 | 2 |
| 2336x1292 | 4 | 2 |
| 2337x1279 | 4 | 2 |
| 2337x1286 | 4 | 2 |
| 2338x1281 | 4 | 2 |
| 2338x1289 | 4 | 2 |
| 2339x1287 | 4 | 2 |
| 2339x1290 | 4 | 2 |
| 2340x1283 | 4 | 2 |
| 2340x1284 | 4 | 2 |
| 2340x1289 | 4 | 2 |
| 2341x1288 | 4 | 2 |
| 2343x1281 | 4 | 2 |
| 2293x1249 | 3 | 3 |
| 2293x1255 | 3 | 2 |
| 2293x1258 | 3 | 2 |
| 2295x1245 | 3 | 2 |
| 2295x1248 | 3 | 2 |
| 2295x1252 | 3 | 3 |
| 2295x1260 | 3 | 2 |
| 2296x1243 | 3 | 2 |
| 2296x1258 | 3 | 2 |
| 2298x1245 | 3 | 2 |
| 2299x1245 | 3 | 2 |
| 2299x1246 | 3 | 2 |
| 2300x1258 | 3 | 3 |
| 2301x1246 | 3 | 2 |
| 2301x1256 | 3 | 3 |
| 2302x1260 | 3 | 2 |
| 2303x1251 | 3 | 3 |
| 2303x1254 | 3 | 3 |
| 2305x1248 | 3 | 2 |
| 2305x1256 | 3 | 3 |
| 2306x1259 | 3 | 3 |
| 2307x1254 | 3 | 3 |
| 2308x1257 | 3 | 2 |
| 2309x1256 | 3 | 3 |
| 2332x1289 | 3 | 2 |
| 2338x1286 | 3 | 2 |
| 2259x1257 | 2 | 1 |
| 2287x1246 | 2 | 1 |
| 2287x1263 | 2 | 1 |
| 2289x1240 | 2 | 1 |
| 2289x1248 | 2 | 1 |
| 2290x1249 | 2 | 1 |
| 2290x1254 | 2 | 1 |
| 2292x1249 | 2 | 1 |
| 2292x1251 | 2 | 2 |
| 2292x1252 | 2 | 2 |
| 2292x1254 | 2 | 1 |
| 2292x1258 | 2 | 1 |
| 2293x1261 | 2 | 1 |
| 2295x1243 | 2 | 1 |
| 2295x1246 | 2 | 1 |
| 2295x1254 | 2 | 2 |
| 2296x1246 | 2 | 2 |
| 2296x1260 | 2 | 2 |
| 2298x1243 | 2 | 1 |
| 2298x1260 | 2 | 1 |
| 2299x1236 | 2 | 1 |
| 2300x1254 | 2 | 2 |
| 2300x1256 | 2 | 2 |
| 2300x1259 | 2 | 2 |
| 2300x1260 | 2 | 2 |
| 2301x1261 | 2 | 2 |
| 2302x1245 | 2 | 2 |
| 2302x1246 | 2 | 1 |
| 2302x1256 | 2 | 2 |
| 2302x1261 | 2 | 1 |
| 2302x1264 | 2 | 1 |
| 2304x1242 | 2 | 1 |
| 2304x1248 | 2 | 2 |
| 2305x1240 | 2 | 1 |
| 2305x1245 | 2 | 1 |
| 2306x1254 | 2 | 2 |
| 2306x1260 | 2 | 2 |
| 2307x1256 | 2 | 2 |
| 2307x1258 | 2 | 2 |
| 2308x1258 | 2 | 2 |
| 2311x1254 | 2 | 1 |
| 2313x1258 | 2 | 1 |
| 2321x1277 | 2 | 1 |
| 2325x1279 | 2 | 1 |
| 2326x1284 | 2 | 1 |
| 2326x1285 | 2 | 1 |
| 2326x1288 | 2 | 1 |
| 2326x1294 | 2 | 1 |
| 2327x1289 | 2 | 1 |
| 2328x1279 | 2 | 1 |
| 2328x1280 | 2 | 1 |
| 2328x1287 | 2 | 1 |
| 2328x1293 | 2 | 1 |
| 2329x1278 | 2 | 1 |
| 2329x1280 | 2 | 1 |
| 2329x1282 | 2 | 1 |
| 2329x1283 | 2 | 1 |
| 2329x1284 | 2 | 1 |
| 2329x1286 | 2 | 1 |
| 2329x1287 | 2 | 1 |
| 2330x1278 | 2 | 1 |
| 2330x1280 | 2 | 1 |
| 2330x1286 | 2 | 1 |
| 2330x1287 | 2 | 1 |
| 2330x1288 | 2 | 1 |
| 2331x1277 | 2 | 1 |
| 2331x1278 | 2 | 1 |
| 2331x1279 | 2 | 1 |
| 2331x1281 | 2 | 1 |
| 2331x1283 | 2 | 1 |
| 2331x1285 | 2 | 1 |
| 2332x1273 | 2 | 1 |
| 2332x1277 | 2 | 1 |
| 2332x1279 | 2 | 1 |
| 2332x1282 | 2 | 1 |
| 2332x1283 | 2 | 1 |
| 2332x1287 | 2 | 1 |
| 2333x1279 | 2 | 1 |
| 2333x1280 | 2 | 1 |
| 2333x1281 | 2 | 1 |
| 2333x1285 | 2 | 1 |
| 2333x1287 | 2 | 1 |
| 2333x1291 | 2 | 1 |
| 2333x1292 | 2 | 1 |
| 2333x1293 | 2 | 1 |
| 2334x1276 | 2 | 1 |
| 2334x1282 | 2 | 1 |
| 2334x1284 | 2 | 1 |
| 2334x1286 | 2 | 1 |
| 2334x1288 | 2 | 1 |
| 2334x1292 | 2 | 1 |
| 2334x1294 | 2 | 1 |
| 2335x1272 | 2 | 1 |
| 2335x1277 | 2 | 1 |
| 2335x1278 | 2 | 1 |
| 2335x1281 | 2 | 1 |
| 2335x1283 | 2 | 1 |
| 2335x1284 | 2 | 1 |
| 2335x1286 | 2 | 1 |
| 2335x1288 | 2 | 1 |
| 2336x1279 | 2 | 1 |
| 2336x1285 | 2 | 1 |
| 2336x1286 | 2 | 1 |
| 2336x1291 | 2 | 1 |
| 2337x1282 | 2 | 1 |
| 2337x1283 | 2 | 1 |
| 2337x1287 | 2 | 1 |
| 2337x1288 | 2 | 1 |
| 2337x1290 | 2 | 1 |
| 2337x1291 | 2 | 1 |
| 2338x1278 | 2 | 1 |
| 2338x1279 | 2 | 1 |
| 2338x1287 | 2 | 1 |
| 2338x1291 | 2 | 1 |
| 2338x1299 | 2 | 1 |
| 2339x1277 | 2 | 1 |
| 2339x1279 | 2 | 1 |
| 2339x1282 | 2 | 1 |
| 2339x1286 | 2 | 1 |
| 2339x1292 | 2 | 1 |
| 2340x1278 | 2 | 1 |
| 2340x1280 | 2 | 1 |
| 2340x1281 | 2 | 1 |
| 2340x1282 | 2 | 1 |
| 2340x1285 | 2 | 1 |
| 2341x1286 | 2 | 1 |
| 2341x1292 | 2 | 1 |
| 2342x1284 | 2 | 1 |
| 2342x1294 | 2 | 1 |
| 2342x1295 | 2 | 1 |
| 2343x1288 | 2 | 1 |
| 2344x1289 | 2 | 1 |
| 2345x1288 | 2 | 1 |
| 2347x1289 | 2 | 1 |
| 1418x883 | 1 | 1 |
| 2265x1248 | 1 | 1 |
| 2281x1248 | 1 | 1 |
| 2284x1248 | 1 | 1 |
| 2284x1255 | 1 | 1 |
| 2286x1249 | 1 | 1 |
| 2287x1249 | 1 | 1 |
| 2290x1248 | 1 | 1 |
| 2290x1252 | 1 | 1 |
| 2290x1257 | 1 | 1 |
| 2292x1245 | 1 | 1 |
| 2292x1246 | 1 | 1 |
| 2292x1248 | 1 | 1 |
| 2292x1255 | 1 | 1 |
| 2295x1257 | 1 | 1 |
| 2296x1239 | 1 | 1 |
| 2296x1264 | 1 | 1 |
| 2297x1248 | 1 | 1 |
| 2297x1250 | 1 | 1 |
| 2297x1255 | 1 | 1 |
| 2297x1257 | 1 | 1 |
| 2297x1258 | 1 | 1 |
| 2298x1258 | 1 | 1 |
| 2298x1264 | 1 | 1 |
| 2299x1240 | 1 | 1 |
| 2299x1253 | 1 | 1 |
| 2299x1259 | 1 | 1 |
| 2299x1261 | 1 | 1 |
| 2300x1252 | 1 | 1 |
| 2300x1253 | 1 | 1 |
| 2300x1257 | 1 | 1 |
| 2301x1240 | 1 | 1 |
| 2301x1248 | 1 | 1 |
| 2301x1259 | 1 | 1 |
| 2302x1243 | 1 | 1 |
| 2302x1250 | 1 | 1 |
| 2302x1259 | 1 | 1 |
| 2303x1252 | 1 | 1 |
| 2303x1253 | 1 | 1 |
| 2303x1260 | 1 | 1 |
| 2303x1261 | 1 | 1 |
| 2304x1246 | 1 | 1 |
| 2304x1249 | 1 | 1 |
| 2304x1261 | 1 | 1 |
| 2305x1261 | 1 | 1 |
| 2306x1252 | 1 | 1 |
| 2306x1253 | 1 | 1 |
| 2307x1246 | 1 | 1 |
| 2307x1253 | 1 | 1 |
| 2307x1257 | 1 | 1 |
| 2307x1259 | 1 | 1 |
| 2307x1260 | 1 | 1 |
| 2308x1247 | 1 | 1 |
| 2308x1251 | 1 | 1 |
| 2308x1256 | 1 | 1 |
| 2308x1259 | 1 | 1 |
| 2308x1260 | 1 | 1 |
| 2309x1250 | 1 | 1 |
| 2309x1253 | 1 | 1 |
| 2309x1255 | 1 | 1 |
| 2310x1248 | 1 | 1 |
| 2310x1256 | 1 | 1 |

## Class imbalance snapshot

The current source labels are not training-ready classes. Nevertheless, their unique-content counts show severe imbalance: `normal` 178, `nc` 209, `particle` 92, `pattern_narrowing` 42, `pattern_die` 25, `short` 15, `open` 14, and `unlabeled` 325. Compound and deformation labels are smaller still. Augmentation must not be planned until final classes, annotation eligibility, and hash-grouped splits are fixed.

## Decision gate

The single next step is a **full-resolution human adjudication session for HIGH-priority cases**, resulting in a written scope decision for particle, scratch/surface marks, `nc`, `pattern_die`, compound labels, and acquisition-failure exclusion. Only then should canonical classes and annotation rules be frozen.
