@echo off
echo =======================================================
echo   Starting SRM Dashboard (SIH26142 - NTRO)
echo =======================================================

if exist .venv-gpu\Scripts\streamlit.exe (
    set VENV_PATH=.venv-gpu
) else if exist .venv\Scripts\streamlit.exe (
    set VENV_PATH=.venv
) else (
    echo Error: No virtual environment found. Please run setup_gpu.bat first!
    pause
    exit /b 1
)

echo Using environment: %VENV_PATH%
echo Launching Streamlit on http://localhost:8501 ...
%VENV_PATH%\Scripts\streamlit run dashboard\app.py
pause
