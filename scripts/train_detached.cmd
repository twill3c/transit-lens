@echo off
setlocal
cd /d C:\_ClaudeCode\transit-lens
set PY=C:\_ClaudeCode\transit-lens\.venv\Scripts\pythonw.exe
if "%~1"=="astronet_w1" goto astronet
if "%~1"=="cam_shuffled_w1" goto shuffled
if "%~1"=="cam_w1" goto camw1
echo unknown tag: %~1
exit /b 2

:astronet
%PY% training\train.py --head astronet --width 1.0 --epochs 40 --threads 4 --out data\models --tag astronet_w1 > logs\train_astronet_w1.log 2>&1
exit /b

:shuffled
%PY% training\train.py --head cam --width 1.0 --epochs 40 --threads 4 --shuffle-labels --out data\models --tag cam_shuffled_w1 > logs\train_shuffled_w1.log 2>&1
exit /b

:camw1
%PY% training\train.py --head cam --width 1.0 --epochs 40 --threads 4 --out data\models --tag cam_w1 > logs\train_cam_w1.log 2>&1
exit /b
