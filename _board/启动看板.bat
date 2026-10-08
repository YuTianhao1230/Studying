@echo off
chcp 65001 >nul 2>&1
title Studying 看板 - 启动本地服务
cd /d "%~dp0"

set "PY="
where python >nul 2>&1 && set "PY=python"
if not defined PY ( where py >nul 2>&1 && set "PY=py" )
if not defined PY ( if exist "%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe" set "PY=%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe" )
if not defined PY ( if exist "D:\Anaconda3\anaconda\python.exe" set "PY=D:\Anaconda3\anaconda\python.exe" )

if not defined PY (
  echo   [x] 没有找到 Python，请先安装 Python。
  echo.
  pause
  exit /b 1
)

echo.
echo   本地看板地址： http://localhost:3000
echo   打开后，左下角「重建数据」即可一键重建（改了 .md 内容后点它就行）。
echo   关掉这个黑窗口 = 停止服务。
echo.

start "" "http://localhost:3000"
"%PY%" server.py 3000

pause
