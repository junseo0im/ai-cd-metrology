"""Focused Phase 1.3 browser regression: only I&A/Data Control screenshots."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_phase12 import ready,click_region
from browser_smoke import choose_theme

DASH=Path(__file__).resolve().parents[1]
OUT=DASH/".validation/phase13"


def check_cards(page,count):
    page.wait_for_function("(n)=>document.querySelectorAll('[class*=st-key-card_]').length===n",arg=count)
    boxes=page.locator("[class*=st-key-card_]").evaluate_all("es=>es.map(e=>({key:e.className,top:e.getBoundingClientRect().top,bottom:e.getBoundingClientRect().bottom}))")
    assert all(b["top"]>=0 and b["bottom"]<=1080 for b in boxes),boxes
    return boxes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8501")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    report={"status":"RUNNING","screenshots":[],"browser_errors":[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe",headless=True)
        try:
            page=browser.new_page(viewport={"width":1920,"height":1080},device_scale_factor=1,color_scheme="light")
            page.set_default_timeout(15000)
            page.on("pageerror",lambda e:report["browser_errors"].append(str(e)))
            page.goto(args.url+"/",wait_until="networkidle")
            page.locator("[data-testid=stSidebar]").get_by_role("link",name="analytics Inspection & Analysis",exact=True).click()
            page.locator(".st-key-card_ia_results [data-testid=stDataFrame]").first.wait_for()
            ready(page)
            report["inspection_cards"]=check_cards(page,3)
            assert page.locator(".st-key-card_ia_results [data-testid=stDataFrame]").count()==2
            assert page.locator(".st-key-card_ia_map .barlayer .point").count()==72
            img=page.locator(".st-key-card_ia_results img")
            img.wait_for()
            assert img.bounding_box()["height"]>156
            page.screenshot(path=str(OUT/"inspection_light.png"))
            report["screenshots"].append("inspection_light.png")
            page.get_by_role("tab",name="MLflow",exact=True).click()
            page.get_by_text("MLflow · 미연결",exact=True).wait_for()
            page.get_by_role("tab",name="공정 조건별 그래프",exact=True).click()
            report["tabs"]="PASS"
            click_region(page,"22","lt")
            page.get_by_text("선택 Die 22 / lt",exact=False).wait_for()
            page.locator(".st-key-card_ia_results").get_by_text("IMG-DEMO-1-10",exact=False).wait_for()
            page.get_by_text("Overlay 예시",exact=True).click()
            page.locator(".st-key-card_ia_results").get_by_text("원본 영역 예시 / Mock",exact=False).wait_for()
            page.get_by_role("button",name="Simulation",exact=True).click()
            page.get_by_role("heading",name="Simulation",exact=True).wait_for()
            page.get_by_role("button",name="Inspection & Analysis로 돌아가기",exact=True).click()
            page.get_by_text("선택 Die 22 / lt",exact=False).wait_for()
            page.locator(".st-key-card_ia_results").get_by_text("IMG-DEMO-1-10",exact=False).wait_for()
            assert not page.get_by_role("switch",name="Overlay 예시",exact=True).is_checked()
            report["simulation_return_and_overlay"]="PASS"
            choose_theme(page,"Dark")
            check_cards(page,3)
            report["inspection_dark"]="PASS"
            sidebar=page.locator("[data-testid=stSidebar]")
            sidebar.get_by_role("link",name="database Data Control Center",exact=True).click()
            page.get_by_role("heading",name="Data Control Center",exact=True).wait_for()
            page.locator(".st-key-card_data_progress [data-testid=stProgress]").wait_for()
            ready(page)
            report["data_cards"]=check_cards(page,4)
            assert page.locator(".image-tiles .status-tile").count()==5
            assert page.locator(".pipeline .pipeline-step").count()==5
            assert page.locator(".st-key-card_data_pipeline").get_by_text("전체 Image",exact=False).count()>0
            report["data_dark"]="PASS"
            choose_theme(page,"Light")
            page.screenshot(path=str(OUT/"data_control_light.png"))
            report["screenshots"].append("data_control_light.png")
            page.get_by_role("button",name="BATCH-DEMO-070",exact=True).click()
            page.get_by_role("heading",name="Batch 상세",exact=True).wait_for()
            assert page.get_by_role("combobox",name="Batch 선택 (예시)").input_value()=="BATCH-DEMO-070"
            assert "WAFER-DEMO-02" in page.locator(".st-key-platform_header").inner_text()
            report["batch_navigation"]="PASS"
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
