@echo off
REM Start the training monitor in Cursor IDE
cd /d "%~dp0"
echo ========================================
echo MMHCL Training Monitor
echo ========================================
echo.
echo Monitoring training progress...
echo Press Ctrl+C to stop monitoring
echo.
python monitor_training.py
pause

