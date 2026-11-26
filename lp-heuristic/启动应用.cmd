
@echo off
setlocal
set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%APP_PYTHON%" set "APP_PYTHON=%~dp0..\lp-heuristic_app_1126_syy_local\.venv\Scripts\python.exe"
if not exist "%APP_PYTHON%" (
    echo Install the dependencies in this project folder first.
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
pushd "%~dp0" || exit /b 1
"%APP_PYTHON%" -X utf8 app\main.py
set "APP_STATUS=%ERRORLEVEL%"
popd
if not "%APP_STATUS%"=="0" pause
exit /b %APP_STATUS%
