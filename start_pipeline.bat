@echo off
title AI Training Pipeline Launcher

echo ======================================
echo   INITIALIZING AUTONOMOUS AI LOOP
echo ======================================

cd /d C:\AI

echo [1/4] Activating environment...
call .venv\Scripts\activate

echo [2/4] Running data pipeline...
python scripts\data_processor.py

echo [3/4] Starting training loop...
python scripts\train.py

echo [4/4] Running evaluation...
python scripts\eval.py

echo ======================================
echo Launching orchestrator (AUTONOMOUS MODE)
echo ======================================

python scripts\orchestrator.py

pause
