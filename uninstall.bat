@echo off
setlocal
rem Remove the 'marsha' launcher installed by install.bat.
rem Usage: uninstall.bat [PREFIX]
set "PREFIX=%~1"
if "%PREFIX%"=="" set "PREFIX=%USERPROFILE%\.local"
if exist "%PREFIX%\bin\marsha.bat" (
    del "%PREFIX%\bin\marsha.bat"
    echo removed %PREFIX%\bin\marsha.bat
) else (
    echo marsha: no launcher found at %PREFIX%\bin\marsha.bat
)
exit /b 0
