"""Map-only follow-up QA; never regenerates whole-dashboard screenshots."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_phase12 import ready,click_region

OUT=Path(__file__).resolve().parents[1]/".validation/wafer_8cols_3rows"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8502")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    report={"status":"RUNNING","browser_errors":[],"checked_bindings":[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe",headless=True)
        try:
            page=browser.new_page(viewport={"width":1920,"height":1080},device_scale_factor=1,color_scheme="light")
            page.set_default_timeout(15000)
            page.on("pageerror",lambda e:report["browser_errors"].append(str(e)))
            page.goto(args.url+"/",wait_until="networkidle")
            page.locator("[data-testid=stSidebar]").get_by_role("link",name="analytics Inspection & Analysis",exact=True).click()
            page.wait_for_function("document.querySelectorAll('.st-key-card_ia_map .barlayer .point').length===72")
            ready(page)
            plot=page.locator(".st-key-card_ia_map .js-plotly-plot")
            assert page.get_by_text("Mock 8열×3행",exact=False).count()==1
            assert set(page.locator(".st-key-card_ia_map .annotation-text").all_text_contents())=={
                f"{row}{col}" for row in range(1,4) for col in range(1,9)}
            assert len(page.locator(".st-key-card_ia_map .legendtext").all_text_contents())==6
            before=plot.evaluate("e=>e.data.map(t=>t.marker.color)")
            geometry=plot.evaluate("""e=>({
                columns:new Set(e.data.flatMap(t=>t.x)).size,
                rows:new Set(e.data.flatMap(t=>t.customdata.map((r,i)=>r[1]==='c'?t.base[i]+t.y[i]/2:null)).filter(v=>v!==null)).size,
                height:e._fullLayout.height
            })""")
            assert geometry=={"columns":8,"rows":3,"height":250},geometry
            page.screenshot(path=str(OUT/"wafer_light.png"))
            for die,position,index in [
                ("11","lt",1),("18","rb",8),("21","c",9),
                ("28","c",16),("31","rb",17),("38","lt",24)
            ]:
                click_region(page,die,position)
                page.get_by_text(f"선택 Die {die} / {position}",exact=False).wait_for()
                image_id=f"IMG-DEMO-1-{index:02}"
                page.locator(".st-key-card_ia_results").get_by_text(image_id,exact=False).wait_for()
                assert page.get_by_role("switch",name="Overlay 예시",exact=True).is_checked()
                report["checked_bindings"].append({"die_id":die,"position":position,"image_id":image_id})
            click_region(page,"38","lt")
            page.get_by_text("선택 없음 · 기본 예시 이미지",exact=False).wait_for()
            page.locator(".st-key-card_ia_results").get_by_text("IMG-DEMO-1-01",exact=False).wait_for()
            report["reclick_deselect"]="PASS"
            click_region(page,"22","c")
            page.get_by_text("선택 Die 22 / c",exact=False).wait_for()
            page.locator(".st-key-card_ia_results").get_by_text("IMG-DEMO-1-10",exact=False).wait_for()
            page.get_by_text("Overlay 예시",exact=True).click()
            page.locator(".st-key-card_ia_results").get_by_text("원본 영역 예시 / Mock",exact=False).wait_for()
            page.get_by_role("button",name="전체 보기 / 선택 초기화",exact=True).click()
            page.get_by_text("선택 없음 · 기본 예시 이미지",exact=False).wait_for()
            page.locator(".st-key-card_ia_results").get_by_text("IMG-DEMO-1-01",exact=False).wait_for()
            assert not page.get_by_role("switch",name="Overlay 예시",exact=True).is_checked()
            assert plot.evaluate("e=>e.data.map(t=>t.marker.color)")==before
            ready(page)
            assert not report["browser_errors"],report["browser_errors"]
            report.update(status="PASS",die_count=24,position_count=72,geometry=geometry,
                          reset_and_overlay="PASS",criticality_colors_unchanged="PASS")
        except Exception:
            report["status"]="FAIL"
            raise
        finally:
            browser.close()
            (OUT/"browser_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
