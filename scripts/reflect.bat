@echo off
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
call .venv\Scripts\activate.bat 2>nul
python agent.py --reflect
exit /b %ERRORLEVEL%
