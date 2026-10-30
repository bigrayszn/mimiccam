@echo off
rem Run MimicCam from source.
rem Uses .venv in this folder when present, otherwise the active Python.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m mimiccam %*
) else (
    python -m mimiccam %*
)

if errorlevel 1 pause
endlocal
