# YOLOv8-seg Annotation Review v1

## Status

This is a decision draft only. No image has been annotated and no final class list is asserted.

## Object scope principle

Mask only the **localized defect region**, not the entire normal repeating pattern and not the whole image. A source filename is evidence to review, not sufficient evidence to draw a polygon.

## Candidate class rules

| Candidate | Definition | Include | Exclude | Polygon boundary | Ambiguous cases |
| --- | --- | --- | --- | --- | --- |
| open | local physical discontinuity/break in an intended pattern | visible broken/open region with a traceable local boundary | crop edge, weak contrast alone, missing content caused by framing | tight polygon around altered/broken material and immediate malformed edge; do not mask unaffected line length | open plus particle; open touching image border; weak/blurred discontinuity |
| short/bridge | unintended physical connection between adjacent features | visible bridge/connection | optical blur that merely closes a gap visually; border truncation | tight polygon around the connecting material and altered junction | short plus particle/open; bridge boundary broader than the neck |
| pattern_narrowing | localized abnormal reduction of intended pattern width | locally narrowed segment that differs from neighboring baseline | global scale/crop difference; uniformly thin acquisition without reference | polygon around the narrowed segment including transition shoulders, excluding normal continuation | gradual taper; narrowing plus deformation/particle; whole-field change |
| pattern_die/deformation | TBD: visually damaged/deformed or weak pattern | only cases accepted after team definition | low contrast, defocus, screenshot/export artifact if acquisition-only | TBD; local altered structure only if a physical boundary is defensible | strong overlap with exposure/quality failure and narrowing |
| particle | TBD operational class for localized foreign object/contamination | include only if team declares particles detection targets | background dust, sensor/optical artifact, stain, and non-target contamination | trace visible object silhouette; do not include shadow/halo unless operationally part of object | tiny speck, touching pattern, uncertain depth/source |
| scratch/surface mark | TBD | include only if wafer-surface marks are explicit detection targets | illumination gradient, pattern lines, acquisition streak, compression artifact | trace the visible anomalous mark, not crossed normal patterns | faint/long marks, border entry/exit, scratch vs debris trail |

## Normal and non-object handling

- NORMAL images receive no defect polygon.
- Low contrast, blur, defocus, illumination gradient, saturation, UI/ruler overlay, resize/export artifact, crop/border failure and resolution difference are **quality metadata**, not YOLO objects, unless the team explicitly defines a separate operational task.
- Reference/evaluation assets (`optiview_ref_image.png`, `team_protocol_compare/target_01.png`) are not annotation candidates until dataset scope explicitly includes them.

## Multiple defects

- Preserve each visible object as a separate instance and class only after the class list is approved.
- Overlapping defects may have overlapping polygons; do not collapse compound filename labels into one invented class.
- If the physical boundary or class cannot be separated, mark the image/instance UNCERTAIN and do not force an annotation.

## Polygon guidance

- Annotate at native resolution on the untouched raw image.
- Use enough vertices to follow the defect boundary without tracing optical noise pixel-by-pixel.
- For a defect touching the image border, close the polygon on the border and mark it as a border case for review.
- For tiny objects that cannot be reliably bounded at native view, defer rather than create an arbitrary box-like mask.
- Do not include the normal pattern area merely to provide context.

## Decisions required before annotation starts

1. Final inclusion/exclusion of particle, scratch/surface mark and contamination.
2. Operational definition of `nc`.
3. Whether `pattern_die` is physical, acquisition/process-condition, or must be split.
4. Multi-label/overlap policy for the **13 compound-label unique contents**.
5. Minimum visible object size and handling of tiny/border objects.
6. Native-image eligibility policy for screenshots, severe crops, blank-like fields, overlays and unconfirmed resize/export cases.
7. Human adjudication of HIGH-priority cases in `docs/defect_human_review_v1.md`.

## Required sequence

Raw audit → taxonomy/annotation decision → canonical class approval → hash/wafer-grouped split → annotation → training-only augmentation. Do not augment before the split is frozen.
