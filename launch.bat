@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  vLLM Windows Server Launcher
echo  =============================
echo.

REM ============================================================
REM  Check Python installation
REM ============================================================
if not exist "%~dp0python\python.exe" (
    echo  Python not found. Running installer...
    echo.
    call "%~dp0install.bat"
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo  Installation failed. Please check errors above.
        pause
        exit /b 1
    )
    echo.
)

REM ============================================================
REM  Configure environment
REM ============================================================
set "PATH=%~dp0python;%~dp0python\Scripts;%~dp0python\Library\bin;%PATH%"
set "VLLM_ATTENTION_BACKEND=FLASH_ATTN"
set "VLLM_HOST_IP=127.0.0.1"

REM Suppress tokenizer parallelism warning
set "TOKENIZERS_PARALLELISM=false"

REM ============================================================
REM  Launch vLLM server
REM ============================================================
REM All arguments are forwarded to vllm_launcher.py.
REM If no --model is passed, the interactive model selector activates.

"%~dp0python\python.exe" "%~dp0vllm_launcher.py" %*

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo  Server exited with error code !ERRORLEVEL!
    pause
)

endlocal
