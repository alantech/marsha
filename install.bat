@echo off
setlocal
rem Install a 'marsha' launcher on Windows (the Windows twin of the
rem Makefile's install target): build the venv, then write a launcher
rem script into PREFIX\bin (default: %USERPROFILE%\.local).
rem Usage: install.bat [PREFIX]
cd /d "%~dp0"
set "PREFIX=%~1"
if "%PREFIX%"=="" set "PREFIX=%USERPROFILE%\.local"
where uv >nul 2>nul || (
    echo marsha: uv not found on PATH. Install it with: pip install uv 1>&2
    exit /b 1
)
if not exist venv\Scripts\python.exe uv venv venv || goto :error
uv pip install --python venv\Scripts\python.exe --upgrade . || goto :error
venv\Scripts\python.exe make_launcher.py "%PREFIX%\bin" "%CD%\venv" || goto :error
exit /b 0
:error
echo marsha: install failed 1>&2
exit /b 1
