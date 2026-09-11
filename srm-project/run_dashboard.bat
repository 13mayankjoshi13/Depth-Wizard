@echo off
setlocal
cd /d "%~dp0"

echo =======================================================
echo   Starting SRM Dashboard (SIH26142 - NTRO)
echo =======================================================

if exist "%~dp0.venv-gpu\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv-gpu\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    echo Error: No virtual environment found.
    echo Please make sure .venv or .venv-gpu exists in this folder!
    pause
    exit /b 1
)

echo Using Python: %PYTHON_EXE%
echo Launching Streamlit on http://localhost:8501 ...
"%PYTHON_EXE%" -m streamlit run "%~dp0dashboard\app.py"
pause

