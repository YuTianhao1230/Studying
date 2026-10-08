@echo off
chcp 65001 >nul 2>&1
title Studying 看板 - 刷新数据
cd /d "%~dp0"

echo.
echo   正在扫描 Studying 目录下的所有 .md 笔记 ...
echo.

set "PY="
where python >nul 2>&1 && set "PY=python"
if not defined PY ( where py >nul 2>&1 && set "PY=py" )
if not defined PY ( if exist "%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe" set "PY=%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe" )
if not defined PY ( if exist "D:\Anaconda3\anaconda\python.exe" set "PY=D:\Anaconda3\anaconda\python.exe" )

if not defined PY (
  echo   [x] 没有找到 Python，请先安装 Python 或手动运行：python build.py
  echo.
  pause
  exit /b 1
)

"%PY%" build.py

echo.
echo   完成后，双击 index.html 打开看板即可看到最新内容。
echo.
pause
