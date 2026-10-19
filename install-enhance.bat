@echo off
rem ============================================================
rem video-enhance-lite user install
rem
rem The source checkout is not used as a writable runtime directory.  The
rem venv and install receipt live below the user's local application data.
rem External engines are never downloaded or bundled by this script.
rem ============================================================
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0.") do set "PROJECT_ROOT=%%~fI"
if not exist "%PROJECT_ROOT%\pyproject.toml" (
    echo [ERROR] pyproject.toml not found under %PROJECT_ROOT% 1>&2
    exit /b 2
)
if not exist "%PROJECT_ROOT%\src\video_enhance_lite\__init__.py" (
    echo [ERROR] video_enhance_lite package marker not found. 1>&2
    exit /b 2
)

if defined VIDEO_ENHANCE_VENV_DIR (
    set "VENV_DIR=%VIDEO_ENHANCE_VENV_DIR%"
) else if defined LOCALAPPDATA (
    set "APP_ROOT=%LOCALAPPDATA%\video-enhance-lite"
    rem Do not reference APP_ROOT in this parenthesized block. cmd.exe expands
    rem percent variables before executing the block, which previously turned
    rem this into \venv when APP_ROOT was not already defined.
    set "VENV_DIR=%LOCALAPPDATA%\video-enhance-lite\venv"
) else (
    set "APP_ROOT=%USERPROFILE%\AppData\Local\video-enhance-lite"
    set "VENV_DIR=%USERPROFILE%\AppData\Local\video-enhance-lite\venv"
)
if not defined APP_ROOT for %%I in ("%VENV_DIR%\..") do set "APP_ROOT=%%~fI"
set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
set "PYW_EXE=%VENV_DIR%\Scripts\pythonw.exe"
set "RECEIPT=%APP_ROOT%\install-root.txt"
set "LOCK_FILE=%PROJECT_ROOT%\requirements-lock-win-x64.txt"

if not exist "%LOCK_FILE%" (
    echo [ERROR] dependency lock not found: %LOCK_FILE% 1>&2
    exit /b 3
)

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 through 3.13 is required on PATH to bootstrap the user venv. 1>&2
    exit /b 4
)
python -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info < (3,14) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.11 through 3.13 is required. 1>&2
    python --version 1>&2
    exit /b 4
)

if not exist "%VENV_DIR%" (
    echo [INFO] creating user venv: %VENV_DIR%
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] failed to create the user venv. 1>&2
        exit /b 5
    )
) else (
    echo [INFO] reusing user venv: %VENV_DIR%
)
if not exist "%PY_EXE%" (
    echo [ERROR] venv python.exe not found: %PY_EXE% 1>&2
    exit /b 5
)

echo [INFO] installing hash-locked Windows x64 dependencies...
if defined VIDEO_ENHANCE_WHEELHOUSE (
    if not exist "%VIDEO_ENHANCE_WHEELHOUSE%\" (
        echo [ERROR] wheelhouse not found: %VIDEO_ENHANCE_WHEELHOUSE% 1>&2
        exit /b 3
    )
    echo [INFO] using offline wheelhouse: %VIDEO_ENHANCE_WHEELHOUSE%
    "%PY_EXE%" -m pip --isolated --disable-pip-version-check install --no-index --find-links "%VIDEO_ENHANCE_WHEELHOUSE%" --only-binary :all: --require-hashes -r "%LOCK_FILE%"
) else (
    "%PY_EXE%" -m pip --isolated --disable-pip-version-check install --index-url https://pypi.org/simple/ --only-binary :all: --require-hashes -r "%LOCK_FILE%"
)
if errorlevel 1 (
    echo [ERROR] locked dependency installation failed. 1>&2
    exit /b 6
)

echo [INFO] binding this checkout without dependency re-resolution...
"%PY_EXE%" -m pip --isolated --disable-pip-version-check install --no-build-isolation --no-deps -e "%PROJECT_ROOT%"
if errorlevel 1 (
    echo [ERROR] package source binding failed. 1>&2
    exit /b 6
)

set "VIDEO_ENHANCE_PROJECT_ROOT=%PROJECT_ROOT%"
set "VIDEO_ENHANCE_INSTALL_RECEIPT=%RECEIPT%"
"%PY_EXE%" -c "import os; from pathlib import Path; p=Path(os.environ['VIDEO_ENHANCE_INSTALL_RECEIPT']); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(os.environ['VIDEO_ENHANCE_PROJECT_ROOT'], encoding='utf-8')"
if errorlevel 1 (
    echo [ERROR] failed to write install receipt: %RECEIPT% 1>&2
    exit /b 6
)

"%PY_EXE%" -m video_enhance_lite --help >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CLI self-import failed after installation. 1>&2
    exit /b 7
)

echo.
echo [INFO] capability diagnostics (missing optional engines are allowed):
"%PY_EXE%" -m video_enhance_lite --self-check
if errorlevel 1 (
    echo [ERROR] capability diagnostics failed. 1>&2
    exit /b 7
)

echo.
echo [DONE] package installed for this checkout.
echo        launcher: "%PROJECT_ROOT%\video-enhance-lite.bat"
echo        settings: %%LOCALAPPDATA%%\video-enhance-lite\config\
echo        temp:     %%LOCALAPPDATA%%\video-enhance-lite\temp\
echo        external engines are optional and must be configured explicitly.
exit /b 0
