@echo off
REM Start both servers in separate windows
echo Starting Local AI Chat...
echo.
echo Make sure Ollama is running! (ollama serve)
echo.
start "Local AI Chat - Backend" cmd /k python -m uvicorn main:app --reload
timeout /t 2 /nobreak >nul
start "Local AI Chat - Frontend" cmd /k python -m http.server 5500
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:5500
echo.
echo App should be opening in your browser.
echo Close this window when done.
pause
