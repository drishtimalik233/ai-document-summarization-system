@echo off
setlocal
title AI Document Summarization System - Launcher
cd /d "%~dp0"

echo ================================================
echo   AI Document Summarization System - Launcher
echo ================================================
echo.

REM ---- 1. Python check ------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Install Python from https://www.python.org/downloads/
    echo During setup, tick "Add python.exe to PATH", then run this file again.
    echo.
    pause
    exit /b 1
)

REM ---- 2. Virtual environment -----------------------------------------
if not exist "venv\Scripts\activate.bat" (
    echo [1/4] Creating virtual environment, please wait...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
)
call "venv\Scripts\activate.bat"

REM ---- 3. Python packages ---------------------------------------------
if not exist "venv\.deps_installed" (
    echo [2/4] Installing packages. This is needed only on the first run...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Package installation failed. Check your internet connection.
        pause
        exit /b 1
    )
    echo done> "venv\.deps_installed"
)

REM ---- 4. Ollama -------------------------------------------------------
where ollama >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not installed.
    echo Install it from https://ollama.com/download and run this file again.
    echo.
    pause
    exit /b 1
)

ollama list >nul 2>&1
if errorlevel 1 (
    echo [3/4] Starting Ollama...
    start "" /min ollama serve
    timeout /t 8 /nobreak >nul
)

ollama list 2>nul | findstr /i "llama3.2:1b" >nul
if errorlevel 1 (
    echo [3/4] Downloading the AI model llama3.2:1b, about 1.3 GB. First run only...
    ollama pull llama3.2:1b
    if errorlevel 1 (
        echo [ERROR] Model download failed. Check your internet connection.
        pause
        exit /b 1
    )
)

REM ---- 5. Skip the Streamlit first-run email prompt --------------------
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" >nul 2>&1
    >  "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

REM ---- 6. Launch -------------------------------------------------------
echo.
echo [4/4] Starting the app. Your browser will open at http://localhost:8501
echo Keep this window open while using the app. Press Ctrl+C or close it to stop.
echo.
python -m streamlit run app.py

echo.
pause
