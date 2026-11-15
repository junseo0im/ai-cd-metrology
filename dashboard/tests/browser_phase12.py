"""Focused Phase 1.2 browser QA. Exactly four new Light screenshots."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

DASH=Path(__file__).resolve().parents[1]
OUT=DASH/".validation/phase12"


def ready(page):
    page.locator(".brand-row").wait_for()
    page.wait_for_function("document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state')==='notRunning'")
    assert page.locator("[data-testid=stException]").count()==0


def click_region(page,die_id,position):
    plot=page.locator(".st-key-card_ia_map .js-plotly-plot")
    point=plot.evaluate("""(e,wanted)=>{
        for(const t of e._fullData){
            const i=t.customdata.findIndex(r=>r[0]===wanted[0]&&r[1]===wanted[1]);
            if(i>=0) return {
                x:e._fullLayout.xaxis._offset+e._fullLayout.xaxis.l2p(t.x[i]),
                y:e._fullLayout.yaxis._offset+e._fullLayout.yaxis.l2p(t.base[i]+t.y[i]/2)
            };
        }
        throw Error('Unknown Mock region');
    }""",[die_id,position])
    box=plot.bounding_box()
    page.mouse.click(box["x"]+point["x"],box["y"]+point["y"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8501")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    result={"status":"RUNNING","screenshots":[],"browser_errors":[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe",headless=True)
        try:
            page=browser.new_page(viewport={"width":1920,"height":1080},device_scale_factor=1,color_scheme="light")
            page.set_default_timeout(15000)
            page.on("pageerror",lambda e:result["browser_errors"].append(str(e)))
            def shot(name):
                page.screenshot(path=str(OUT/(name+".png")))
                result["screenshots"].append(name+".png")
            page.goto(args.url,wait_until="networkidle")
            page.get_by_role("button",name="실험 환경 및 조건 열기",exact=True).wait_for()
            ready(page)
            sidebar=page.locator("[data-testid=stSidebar]")
            assert "테마: 우상단" not in sidebar.inner_text()
            assert "MOCK DATA ONLY" in sidebar.inner_text()
            a=page.get_by_role("button",name="Data Control Center 열기",exact=True)
            b=page.get_by_role("button",name="Inspection & Analysis 열기",exact=True)
            assert a.evaluate("e=>getComputedStyle(e).backgroundColor")==b.evaluate("e=>getComputedStyle(e).backgroundColor")
            shot("home_light")
            page.get_by_role("button",name="실험 환경 및 조건 열기",exact=True).click()
            page.get_by_role("heading",name="실험 환경 및 조건",exact=True).wait_for()
            page.get_by_role("button",name="Main Home으로 돌아가기",exact=True).click()
            page.get_by_role("button",name="Data Control Center 열기",exact=True).wait_for()
            result["environment_round_trip"]="PASS"
            page.get_by_role("button",name="Data Control Center 열기",exact=True).click()
            page.get_by_role("heading",name="Data Control Center",exact=True).wait_for()
            sidebar.get_by_role("link",name="home Main Home",exact=True).click()
            page.get_by_role("button",name="Inspection & Analysis 열기",exact=True).click()
            page.locator(".st-key-card_ia_map .js-plotly-plot").wait_for()
            page.wait_for_function("document.querySelectorAll('.st-key-card_ia_map .barlayer .point').length===72")
            ready(page)
            labels=page.locator(".st-key-card_ia_map .annotation-text").all_text_contents()
            assert set(labels)=={f"{r}{c}" for r in range(1,4) for c in range(1,9)},labels
            names=page.locator(".st-key-card_ia_map .legendtext").all_text_contents()
            assert {n.split(" · ")[0] for n in names}=={"NORMAL","MINOR","WARNING","CRITICAL","UNUSABLE","NOT_EVALUATED"}
            before=page.locator(".st-key-card_ia_map .js-plotly-plot").evaluate("e=>e.data.map(t=>t.marker.color)")
            shot("wafer_full_light")
            click_region(page,"22","lt")
            page.get_by_text("선택 Die 22 / lt",exact=False).wait_for()
            page.locator(".st-key-card_ia_results").get_by_text("IMG-DEMO-1-10",exact=False).wait_for()
            ready(page)
            shot("wafer_selected_light")
            click_region(page,"22","lt")
            page.get_by_text("선택 없음 · 기본 예시 이미지",exact=False).wait_for()
            page.locator(".st-key-card_ia_results").get_by_text("IMG-DEMO-1-01",exact=False).wait_for()
            result["same_region_toggle"]="PASS"
            click_region(page,"22","lt")
            page.get_by_text("선택 Die 22 / lt",exact=False).wait_for()
            click_region(page,"24","rb")
            page.get_by_text("선택 Die 24 / rb",exact=False).wait_for()
            page.locator(".st-key-card_ia_results").get_by_text("IMG-DEMO-1-12",exact=False).wait_for()
            result["different_region_image_sync"]="PASS"
            page.get_by_role("button",name="전체 보기 / 선택 초기화",exact=True).click()
            page.get_by_text("선택 없음 · 기본 예시 이미지",exact=False).wait_for()
            page.locator(".st-key-card_ia_results").get_by_text("IMG-DEMO-1-01",exact=False).wait_for()
            ready(page)
            after=page.locator(".st-key-card_ia_map .js-plotly-plot").evaluate("e=>e.data.map(t=>t.marker.color)")
            assert before==after
            assert page.get_by_role("switch",name="Overlay 예시",exact=True).is_checked()
            shot("wafer_reset_light")
            result.update({"status":"PASS","die_count":24,"region_count":72,
                           "home_buttons":"PASS","criticality_legend":"PASS",
                           "reset_colors_and_overlay":"PASS"})
            assert not result["browser_errors"],result["browser_errors"]
        except Exception:
            result["status"]="FAIL"
            raise
        finally:
            browser.close()
            (OUT/"browser_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
