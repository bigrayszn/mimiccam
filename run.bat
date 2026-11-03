@echo off
REM SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
REM SPDX-License-Identifier: GPL-2.0-or-later
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
