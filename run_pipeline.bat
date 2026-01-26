@echo off
REM Change directory to the location of the batch file
cd "%~dp0"

REM Change directory to the 'src' folder
#cd src

REM Run the Python script. Use 'python' or 'py' depending on your system's PATH configuration.
python src\main.py

REM Optional: Keep the window open after execution to see the output
REM pause
