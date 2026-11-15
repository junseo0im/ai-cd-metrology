"""Focused Sidebar/detail navigation QA. Only two new detail screenshots."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_phase12 import ready
from browser_smoke import choose_theme

OUT=Path(__file__).resolve().parents[1]/".validation/sidebar_details"
EXPERIMENT_TABS=["장비 사양","Wafer / Die / Pattern 구조","공정 조건",
                 "Mask 설계 / 제작값","오차율 계산 기준(허용값)","Calibration"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8502")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    report={"status":"RUNNING","screenshots":[],"browser_errors":[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe",headless=True)
        try:
            page=browser.new_page(viewport={"width":1920,"height":1080},device_scale_factor=1,color_scheme="light")
            page.set_default_timeout(18000)
            page.on("pageerror",lambda e:report["browser_errors"].append(str(e)))
            page.goto(args.url+"/",wait_until="networkidle")
            page.get_by_role("button",name="실험 환경 및 조건 열기",exact=True).wait_for()
            ready(page)
            sidebar=page.locator("[data-testid=stSidebar]")
            def side_link(name):
                sidebar.get_by_role("link",name=name,exact=True).click()
            text=sidebar.inner_text()
            positions=[text.index(label) for label in ["모델 성능","실험 환경 및 조건","Critical 기준","AI 활용","모델 선정 기준","설정"]]
            assert positions==sorted(positions),positions
            page.get_by_role("button",name="실험 환경 및 조건 열기",exact=True).click()
            page.get_by_role("heading",name="실험 환경 및 조건",exact=True,level=1).wait_for()
            page.wait_for_url("**/experiment-environment")
            page.get_by_role("tab",name="Calibration",exact=True).wait_for()
            assert page.get_by_role("tab").all_text_contents()==EXPERIMENT_TABS
            for label in EXPERIMENT_TABS:
                page.get_by_role("tab",name=label,exact=True).click()
            page.get_by_text("Calibration 상세 · Placeholder",exact=True).wait_for()
            assert page.locator("[data-testid=stDataFrame]").count()==1
            page.screenshot(path=str(OUT/"experiment_calibration_light.png"),animations="disabled")
            report["screenshots"].append("experiment_calibration_light.png")
            page.get_by_role("button",name="Main Home으로 돌아가기",exact=True).click()
            page.get_by_role("button",name="Data Control Center 열기",exact=True).click()
            page.get_by_role("heading",name="Data Control Center",exact=True,level=1).wait_for()
            side_link("straighten Calibration")
            page.get_by_role("heading",name="Calibration",exact=True,level=1).wait_for()
            page.get_by_text("Calibration 상세 · Placeholder",exact=True).wait_for()
            assert page.locator("[data-testid=stDataFrame]").count()==1
            report["existing_home_card_and_calibration_reuse"]="PASS"
            side_link("rule Critical 기준")
            page.get_by_role("heading",name="Critical 기준",exact=True,level=1).wait_for()
            page.get_by_role("tab",name="결함 Critical 기준",exact=True).wait_for()
            assert page.get_by_role("tab").all_text_contents()==["측정 Critical 기준","결함 Critical 기준"]
            for label in ["측정 Critical 기준","결함 Critical 기준"]:
                page.get_by_role("tab",name=label,exact=True).click()
                page.get_by_text(label+" · 미확정",exact=True).wait_for()
            side_link("psychology AI 활용")
            page.get_by_role("heading",name="AI 활용",exact=True,level=1).wait_for()
            page.get_by_role("tab",name="AI Tool",exact=True).wait_for()
            assert page.get_by_role("tab").all_text_contents()==["Before & After AI","AI Tool"]
            page.get_by_role("tab",name="AI Tool",exact=True).click()
            page.get_by_text("활용 도구 및 역할",exact=True).wait_for()
            report["critical_ai_tabs"]="PASS"
            choose_theme(page,"Dark")
            side_link("fact_check 모델 선정 기준")
            page.get_by_role("heading",name="모델 선정 기준",exact=True,level=1).wait_for()
            left=page.locator(".st-key-model_selection_mlflow")
            right=page.locator(".st-key-model_selection_criteria")
            left.wait_for()
            a,b=left.bounding_box(),right.bounding_box()
            assert a["x"]+a["width"]<=b["x"] and abs(a["y"]-b["y"])<2,(a,b)
            assert page.locator(".stApp").evaluate("e=>getComputedStyle(e).colorScheme")=="dark"
            model_mlflow=left.locator(".placeholder").inner_text()
            assert "실제 API 호출이나 성능지표는 없습니다." in model_mlflow
            page.get_by_text("모델 후보",exact=True).click()
            page.get_by_text("모델 후보 · 미확정",exact=True).wait_for()
            ready(page)
            page.screenshot(path=str(OUT/"model_selection_dark.png"),animations="disabled")
            report["screenshots"].append("model_selection_dark.png")
            report["model_columns_and_dark_theme"]="PASS"
            page.get_by_role("button",name="Main Home으로 돌아가기",exact=True).click()
            page.get_by_role("button",name="Inspection & Analysis 열기",exact=True).click()
            page.get_by_role("heading",name="Inspection & Analysis",exact=True,level=1).wait_for()
            side_link("repeat 반복촬영 분석")
            page.get_by_role("heading",name="Repeatability",exact=True,level=1).wait_for()
            page.wait_for_url("**/inspection-repeatability")
            assert page.get_by_role("button",name="Repeatability",exact=True).is_disabled()
            assert "BATCH-DEMO-REP" in page.locator(".st-key-platform_header").inner_text()
            side_link("light_mode 밝기 분석")
            page.get_by_role("heading",name="Lighting",exact=True,level=1).wait_for()
            page.wait_for_url("**/inspection-lighting")
            assert page.get_by_role("button",name="Lighting",exact=True).is_disabled()
            assert "BATCH-DEMO-LGT" in page.locator(".st-key-platform_header").inner_text()
            report["sidebar_analysis_mode_sync"]="PASS"
            side_link("analytics Inspection & Analysis")
            page.get_by_role("heading",name="Inspection & Analysis",exact=True,level=1).wait_for()
            page.get_by_role("tab",name="MLflow",exact=True).click()
            page.get_by_text("MLflow · 미연결",exact=True).wait_for()
            assert page.locator(".st-key-card_ia_trend .placeholder").inner_text()==model_mlflow
            assert page.locator(".st-key-card_ia_map .barlayer .point").count()==72
            report["shared_mlflow_content"]="PASS"
            assert not report["browser_errors"],report["browser_errors"]
            report["status"]="PASS"
        except Exception:
            report["status"]="FAIL"
            raise
        finally:
            browser.close()
            (OUT/"browser_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
