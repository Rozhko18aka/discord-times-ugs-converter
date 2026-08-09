@echo off
cd /d "%~dp0"
python "%~dp0ugs_converter.py"
if errorlevel 1 pause
