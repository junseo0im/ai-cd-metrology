"""TASK 18-B-14B: Wafer Map AI provisional risk, 기준 v0.1 (pure functions, synthetic stored results)."""
import unittest

from dashboard.die_assessment import (NOT_EVALUATED, assess_die, cd_score, map_risk, pnpt_score)


def item(widths=(10.0,), pn=0, pt=0, defects=(), outcome="COMPLETE", metrology="SUCCESS"):
    return {"outcome": outcome, "defect_status": "SUCCESS", "engine_status": "SUCCESS", "metrology_status": metrology,
            "line_widths_um": list(widths) if metrology == "SUCCESS" else [], "configuration": {"design_width_um": 10.0},
            "judgments": [{"label": "pn"}] * pn + [{"label": "pt"}] * pt + [{"label": "normal"}],
            "defect_details": [{"class_code": c, "source": "MODEL_POSTPROCESSED"} for c in defects],
            "partial_measurement": 0, "is_adopted": 0}


def die(**positions):
    """positions: lt/c/rb -> list of items (several = re-shots of one Site)."""
    analysis, ids, modes = {}, {}, {}
    for pos in ("lt", "c", "rb"):
        ids[pos] = []
        for n, it in enumerate(positions.get(pos, [item()])):
            key = f"{pos}{n}"; analysis[key] = it; ids[pos].append(key); modes[key] = "wafer_batch"
    if "missing" in positions:
        ids[positions["missing"]] = []
    return assess_die(ids, analysis, modes)


class MapRiskTests(unittest.TestCase):
    def test_01_boundaries(self):
        self.assertEqual([cd_score(e) for e in (0, 10, 10.0001, 20, 20.0001, 30, 30.0001, 80)], [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(cd_score(10.000000000001), 0)                     # Float noise at the boundary.
        self.assertEqual([pnpt_score(n) for n in (0, 1, 2, 3, 5, 6, 40)], [0, 1, 1, 2, 2, 3, 3])

    def test_02_cd_and_pnpt_take_the_higher(self):
        risk = map_risk(die(c=[item(widths=(11.5,))]))                     # 15 % -> MINOR.
        self.assertEqual((risk["display_risk"], risk["s_cd"], risk["s_pnpt"], risk["score"]), ("MINOR", 1, 0, 1))
        risk = map_risk(die(lt=[item(pn=1)], c=[item(pn=1)], rb=[item(pt=1)]))  # CD 0 %, PN/PT 3 -> 2.
        self.assertEqual((risk["display_risk"], risk["pnpt_count"], risk["s_cd"]), ("WARNING", 3, 0))
        risk = map_risk(die(c=[item(widths=(10.5, 12.5))]))                  # Worst line 25 % -> WARNING.
        self.assertEqual((risk["display_risk"], round(risk["cd_worst_error_pct"], 6)), ("WARNING", 25.0))

    def test_03_ospd_adds_one_once_and_caps_at_three(self):
        one = map_risk(die(c=[item(defects=("o",))]))
        three = map_risk(die(lt=[item(defects=("s", "s"))], c=[item(defects=("o",))], rb=[item(defects=("pd",))]))
        self.assertEqual((one["display_risk"], one["ospd"], one["ospd_images"]), ("MINOR", 1, 1))
        self.assertEqual((three["display_risk"], three["ospd"], three["ospd_images"]), ("MINOR", 1, 3))   # No stacking.
        capped = map_risk(die(c=[item(widths=(14.0,), defects=("o",))]))      # 40 % (3) + 1 -> 3.
        self.assertEqual((capped["display_risk"], capped["score"]), ("CRITICAL", 3))

    def test_04_incomplete_and_unmeasurable_stay_not_evaluated(self):
        self.assertEqual(map_risk(die(missing="rb"))["display_risk"], NOT_EVALUATED)
        partial = item(); partial["partial_measurement"] = 1
        self.assertEqual(map_risk(die(c=[partial]))["display_risk"], NOT_EVALUATED)     # Existing completeness rule.
        pd_all = [item(outcome="MODEL_PD_EXCLUDED", metrology="SKIPPED", defects=("pd",))]
        risk = map_risk(die(lt=pd_all, c=pd_all, rb=pd_all))
        self.assertEqual((risk["display_risk"], risk["state"], risk["ospd_images"]), (NOT_EVALUATED, "NO_VALID_MEASUREMENT", 3))
        mixed = map_risk(die(c=pd_all))                                        # PN/PT from measured images only.
        self.assertEqual((mixed["display_risk"], mixed["pnpt_partial"]), ("MINOR", True))

    def test_05_reshots_not_double_counted_and_psc_not_scored(self):
        risk = map_risk(die(c=[item(pn=2), item(pn=2)]))                       # One Site shot twice.
        self.assertEqual((risk["pnpt_count"], risk["s_pnpt"]), (2, 1))
        psc = map_risk(die(c=[item(defects=("p", "sc"))]))
        self.assertEqual((psc["display_risk"], psc["ospd"]), ("NORMAL", 0))
        self.assertTrue(any("P/SC" in r for r in die(c=[item(defects=("p",))])["review"]))   # Review kept.

    def test_06_existing_decision_unchanged(self):
        assessment = die(c=[item(defects=("o",))])
        self.assertEqual((assessment["decision"], assessment["display_risk"]), ("FAIL", "UNUSABLE"))  # v0.1 as before.
        self.assertEqual(map_risk(assessment)["display_risk"], "MINOR")                               # New colour only.


if __name__ == "__main__":
    unittest.main()
