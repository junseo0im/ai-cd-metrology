@echo off
setlocal
cd /d "%~dp0.."
call "%~dp0run_with_python.bat" "%~dp0..\training\mask_editor.py"
if errorlevel 1 pause
