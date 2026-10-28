@echo off
set VENV_PY=%~dp0.venv\Scripts\python.exe
if not exist "%VENV_PY%" (
    echo Error: .venv virtual environment not found!
    exit /b 1
)
if "%1"=="demo" (
    shift
    "%VENV_PY%" "%~dp0demo_pipeline.py" %*
) else (
    "%VENV_PY%" "%~dp0app.py" %*
)
