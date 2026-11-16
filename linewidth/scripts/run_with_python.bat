@echo off
setlocal
if defined LINEWIDTH_PYTHON (
    "%LINEWIDTH_PYTHON%" %*
    exit /b %errorlevel%
)
if exist "%USERPROFILE%\anaconda3\python.exe" (
    "%USERPROFILE%\anaconda3\python.exe" %*
    exit /b %errorlevel%
)
python %*
exit /b %errorlevel%
