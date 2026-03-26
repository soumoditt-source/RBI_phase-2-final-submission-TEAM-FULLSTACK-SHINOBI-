@echo off
title MuleHunter.AI | Team FullStackShinobi
echo.
echo    [ Shinobi-Cortex: Initializing... ]
echo.
python supreme_winner_launch.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo    [!] Error: System could not reach critical mass.
    echo    Verify Python 3.10+ is installed and dependencies are met.
    pause
)
pause
