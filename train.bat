@echo off
REM Automatic MMHCL Training Script for Windows
REM This script runs the automatic training with default settings

echo ========================================
echo MMHCL Automatic Training
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Run the automatic training script
python auto_train.py --dataset Clothing --gpu_id 0

pause

