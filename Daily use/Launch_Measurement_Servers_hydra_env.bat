@echo off
setlocal

set "ROOT=C:\Users\SSPDs\Documents\Python Scripts"
set "HYDRA_ENV=C:\ProgramData\miniforge3\envs\hydra_env"
set "HYDRA_PYTHON=%HYDRA_ENV%\python.exe"

if /i "%~1"=="atlas" goto atlas
if /i "%~1"=="pico" goto pico

start "Atlas SSPD Pyro Server" cmd /k ""%~f0" atlas"
start "PicoHarp T2 Server" cmd /k ""%~f0" pico"
exit /b 0

:atlas
title Atlas SSPD Pyro Server (hydra_env)
cd /d "%ROOT%"
set "PATH=%HYDRA_ENV%;%HYDRA_ENV%\Library\bin;%HYDRA_ENV%\Scripts;%PATH%"
"%HYDRA_PYTHON%" "Daily use\atlas_sspd_pyro_server.py"
echo.
echo Atlas SSPD Pyro server exited with code %ERRORLEVEL%.
exit /b %ERRORLEVEL%

:pico
title PicoHarp T2 Server (hydra_env)
cd /d "%ROOT%"
set "PATH=%HYDRA_ENV%;%HYDRA_ENV%\Library\bin;%HYDRA_ENV%\Scripts;%PATH%"
"%HYDRA_PYTHON%" "Daily use\PicoHarp_srv_T2.py"
echo.
echo PicoHarp T2 server exited with code %ERRORLEVEL%.
exit /b %ERRORLEVEL%
