"""Actual Chrome UI checks. Only localhost and bundled Mock data are used."""
import argparse
import json
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dashboard.navigation import PAGE_SPECS

DASH=ROOT/"dashboard"
OUT=DASH/".validation/screenshots"
CHROME=Path("C:/Program Files/Google/Chrome/Application/chrome.exe")


def wait_ready(page):
    page.locator(".brand-row").wait_for(timeout=20000)
    page.wait_for_function("""() => document.querySelector('[data-testid="stApp"]')?.getAttribute('data-test-script-state') === 'notRunning'""")


def bounds(page):
    return page.evaluate("""() => ({
        header:document.querySelector('.brand-row').getBoundingClientRect().toJSON(),
        native:document.querySelector('[data-testid="stHeader"]').getBoundingClientRect().toJSON(),
        bodyBg:getComputedStyle(document.querySelector('.stApp')).backgroundColor,
        scheme:getComputedStyle(document.querySelector('.stApp')).colorScheme,
        cards:[...document.querySelectorAll('[class*="st-key-card_"]')].map(e=>({
            key:e.className.match(/st-key-(card_[^ ]+)/)?.[1],rect:e.getBoundingClientRect().toJSON(),
            bg:getComputedStyle(e).backgroundColor})),
        errorCount:document.querySelectorAll('[data-testid="stException"]').length
    })""")


def check_layout(page,expect_cards=False,compact=True,card_count=4):
    if expect_cards:
        page.wait_for_function("(n)=>document.querySelectorAll('[class*=st-key-card_]').length===n",arg=card_count)
    result=bounds(page)
    assert result["errorCount"]==0,result
    assert result["header"]["top"]>=result["native"]["bottom"]-1,result
    if expect_cards:
        assert len(result["cards"])==card_count,result
        if compact:
            assert all(c["rect"]["top"]>=0 and c["rect"]["bottom"]<=1080 for c in result["cards"]),result
    return result


def choose_theme(page,value):
    page.get_by_role("button",name="Main menu",exact=True).click()
    icon="dark_mode" if value=="Dark" else "light_mode"
    page.get_by_role("menuitemradio",name=icon+" "+value,exact=True).click()
    page.wait_for_function("(expected)=>getComputedStyle(document.querySelector('.stApp')).colorScheme===expected",
                           arg=value.lower())
    page.get_by_role("button",name="Main menu",exact=True).click()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8501")
    parser.add_argument("--browser-path",type=Path,default=CHROME)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    report={"status":"RUNNING","screenshots":[],"layouts":{},"browser_errors":[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=str(args.browser_path),headless=True)
        context=browser.new_context(viewport={"width":1920,"height":1080},device_scale_factor=1,color_scheme="light")
        page=context.new_page()
        page.set_default_timeout(12000)
        page.on("pageerror",lambda e:report["browser_errors"].append(str(e)))
        def visit(route):
            dark=page.locator(".stApp").count() and page.locator(".stApp").evaluate("e=>getComputedStyle(e).colorScheme")=="dark"
            page.goto(args.url+route,wait_until="networkidle")
            wait_ready(page)
            if dark:
                choose_theme(page,"Dark")
        def shot(name):
            target=OUT/(name+".png")
            page.screenshot(path=str(target),full_page=False)
            report["screenshots"].append(target.name)

        visit("/")
        report["layouts"]["home_light"]=check_layout(page)
        shot("home_light_1920")
        page.get_by_role("button",name="Data Control Center 열기",exact=True).click()
        page.get_by_role("heading",name="Data Control Center",exact=True).wait_for()
        wait_ready(page)
        report["layouts"]["data_light"]=check_layout(page,expect_cards=True)
        shot("data_light_1920")
        sidebar=page.locator('[data-testid="stSidebar"]')
        assert sidebar.get_by_role("link",name="folder_open Data Home",exact=True).count()==1
        assert sidebar.locator(".nav-current").bounding_box()["y"] < sidebar.get_by_text("COMMON NAVIGATION",exact=True).bounding_box()["y"]
        assert page.get_by_role("button",name="BATCH-DEMO-070 →",exact=True).count()==0
        page.get_by_role("button",name="BATCH-DEMO-070",exact=True).click()
        page.get_by_role("heading",name="Batch 상세",exact=True).wait_for()
        wait_ready(page)
        assert page.get_by_role("combobox",name="Batch 선택 (예시)").input_value()=="BATCH-DEMO-070"
        assert "WAFER-DEMO-02" in page.locator(".st-key-platform_header").inner_text()
        shot("batch_detail_light_1920")
        visit("/data-home")
        page.get_by_role("button",name="BATCH-DEMO-060",exact=True).click()
        page.get_by_role("heading",name="Batch 상세",exact=True).wait_for()
        assert page.get_by_role("combobox",name="Batch 선택 (예시)").input_value()=="BATCH-DEMO-060"
        report["batch_link_navigation"]="PASS"

        visit("/inspection-home")
        page.locator(".st-key-card_ia_map .js-plotly-plot").wait_for()
        page.wait_for_function("document.querySelectorAll('.st-key-card_ia_map .barlayer .point').length===72")
        report["layouts"]["inspection_light"]=check_layout(page,expect_cards=True,card_count=3)
        labels=page.locator(".st-key-card_ia_map .annotation-text").all_text_contents()
        assert set(labels)=={f"{r}{c}" for r in range(1,4) for c in range(1,9)},labels
        assert page.locator(".st-key-card_ia_results [data-testid=stDataFrame]").count()==2
        shot("inspection_light_1920")
        from browser_phase12 import click_region
        click_region(page,"31","c")
        page.locator(".st-key-card_ia_results").get_by_text("IMG-DEMO-1-17",exact=False).wait_for()
        report["pattern_click_image_sync"]="PASS"
        shot("inspection_selected_p17_light_1920")
        toggle=page.get_by_role("switch",name="Overlay 예시",exact=True)
        page.get_by_text("Overlay 예시",exact=True).click()
        assert not toggle.is_checked()
        page.locator(".st-key-card_ia_results").get_by_text("원본 영역 예시",exact=False).wait_for()
        page.get_by_text("Overlay 예시",exact=True).click()
        assert toggle.is_checked()
        page.locator(".st-key-card_ia_results").get_by_text("Overlay 예시 /",exact=False).wait_for()
        report["overlay_toggle"]="PASS"
        for mode in ["Repeatability","Lighting","Wafer Batch"]:
            page.get_by_role("button",name=mode,exact=True).click()
            expected="Inspection & Analysis" if mode=="Wafer Batch" else mode
            page.get_by_role("heading",name=expected,exact=True).wait_for()
            wait_ready(page)
            check_layout(page)
        report["mode_switches"]="PASS"

        choose_theme(page,"Dark")
        report["layouts"]["inspection_dark"]=check_layout(page,expect_cards=True,card_count=3)
        shot("inspection_dark_1920")
        assert report["layouts"]["inspection_dark"]["bodyBg"]!="rgb(0, 0, 0)"
        assert len(set(c["bg"] for c in report["layouts"]["inspection_dark"]["cards"]))==1
        report["plotly_dark"]=page.locator(".st-key-card_ia_trend .js-plotly-plot").evaluate(
            """e=>({paper:e._fullLayout.paper_bgcolor,text:e._fullLayout.font.color,
            xTick:e._fullLayout.xaxis.tickfont.color,legend:e._fullLayout.legend.font.color,
            hover:e._fullLayout.hoverlabel})""")
        visit("/data-home")
        report["layouts"]["data_dark"]=check_layout(page,expect_cards=True)
        shot("data_dark_1920")
        assert page.locator(".stApp").evaluate("e=>getComputedStyle(e).colorScheme")=="dark"
        page.get_by_role("button",name="BATCH-DEMO-070",exact=True).click()
        page.get_by_role("heading",name="Batch 상세",exact=True).wait_for()
        assert page.locator(".stApp").evaluate("e=>getComputedStyle(e).colorScheme")=="dark"
        page.get_by_role("link",name="database Data Control Center",exact=True).click()
        page.locator(".st-key-batch_table").wait_for()
        collapse=page.locator('[data-testid="stSidebarCollapseButton"] button')
        page.locator('[data-testid="stSidebar"]').hover()
        collapse.click()
        page.wait_for_function("document.querySelector('[data-testid=stSidebar]').getAttribute('aria-expanded') === 'false'")
        report["layouts"]["data_dark_collapsed"]=check_layout(page,expect_cards=True)
        shot("data_dark_sidebar_collapsed_1920")
        page.locator('[data-testid="stExpandSidebarButton"]').click()
        page.wait_for_function("document.querySelector('[data-testid=stSidebar]').getAttribute('aria-expanded') === 'true'")
        report["sidebar_collapse_expand"]="PASS"
        for key,(_,title,route,_) in PAGE_SPECS.items():
            visit("/"+("" if key=="main_home" else route))
            check_layout(page)
            if key in {"dataset","settings","batch_detail","repeatability","lighting"}:
                shot(key+"_dark_1920")
        report["all_registered_pages"]="PASS"

        page.set_viewport_size({"width":1366,"height":768})
        visit("/data-home")
        report["layouts"]["data_dark_small"]=check_layout(page,compact=False)
        shot("data_dark_1366")
        visit("/inspection-home")
        report["layouts"]["inspection_dark_small"]=check_layout(page,compact=False)
        shot("inspection_dark_1366")
        choose_theme(page,"Light")
        report["layouts"]["inspection_light_small"]=check_layout(page,compact=False)
        shot("inspection_light_1366")
        assert not report["browser_errors"],report["browser_errors"]
        browser.close()
    report["status"]="PASS"
    (DASH/".validation/browser_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in {"layouts","plotly_dark","screenshots"}},ensure_ascii=False,indent=2))
    print(f"Screenshots: {len(report['screenshots'])}; detailed report: {DASH/'.validation/browser_results.json'}")


if __name__=="__main__":
    main()
