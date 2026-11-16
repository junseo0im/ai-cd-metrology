@echo off
call "%~dp0run_with_python.bat" "%~dp0..\training\mask_editor.py" --manifest "%~dp0..\..\analysis\training\manual_reviews\single_boundary\review_manifest.json"
