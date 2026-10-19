@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem The launcher and installer share this user-venv contract.  It never
rem falls back to an arbitrary pythonw found on PATH.
for %%I in ("%~dp0.") do set "PROJECT_ROOT=%%~fI"
if defined VIDEO_ENHANCE_VENV_DIR (
    set "VENV_DIR=%VIDEO_ENHANCE_VENV_DIR%"
) else if defined LOCALAPPDATA (
    set "VENV_DIR=%LOCALAPPDATA%\video-enhance-lite\venv"
) else (
    set "VENV_DIR=%USERPROFILE%\AppData\Local\video-enhance-lite\venv"
)
set "PYTHONW_EXE=%VENV_DIR%\Scripts\pythonw.exe"
set "RECEIPT=%VENV_DIR%\..\install-root.txt"

if not exist "%PYTHONW_EXE%" (
    echo [ERROR] installed pythonw.exe not found: %PYTHONW_EXE% 1>&2
    echo         run install-enhance.bat from this directory first. 1>&2
    exit /b 4
)

if not exist "%RECEIPT%" (
    echo [ERROR] install receipt not found: %RECEIPT% 1>&2
    echo         refusing to launch an unbound or stale checkout. 1>&2
    exit /b 5
)
set "RECEIPT_ROOT="
set /p "RECEIPT_ROOT="<"%RECEIPT%"
if /I not "%RECEIPT_ROOT%"=="%PROJECT_ROOT%" (
    echo [ERROR] installed environment is bound to another checkout. 1>&2
    echo         expected: %PROJECT_ROOT% 1>&2
    echo         receipt:  %RECEIPT_ROOT% 1>&2
    echo         rerun install-enhance.bat from this checkout. 1>&2
    exit /b 6
)

start "" /D "%PROJECT_ROOT%" "%PYTHONW_EXE%" -m video_enhance_lite.gui %*
exit /b 0
