@echo off
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python agent.py %*
