@echo off
cd /d "%~dp0"
set PYTHONPATH=%cd%
echo Starting TradeSwarm Interactive UI/UX Dashboard...
streamlit run app/dashboard.py --server.fileWatcherType none
pause
