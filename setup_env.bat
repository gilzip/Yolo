@echo off
REM Creates a virtual environment and installs project dependencies.
REM Usage: setup_env.bat
setlocal enabledelayedexpansion

if "%PYTHON_BIN%"=="" set PYTHON_BIN=python
if "%VENV_DIR%"=="" set VENV_DIR=.venv

echo Creating virtual environment in %VENV_DIR% using %PYTHON_BIN%...
%PYTHON_BIN% -m venv %VENV_DIR%
if errorlevel 1 goto :error

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto :error

echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup complete.
echo Activate the environment with: %VENV_DIR%\Scripts\activate.bat
goto :eof

:error
echo Setup failed. See the error above.
exit /b 1
