@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Este iniciador usa uma instalacao compativel do Python, sem alterar seu Python padrao.
rem MediaPipe Tasks 1.2.x e pygame-ce oferecem suporte ao Python 3.14 (64 bits).
set "PYTHON="
where py >nul 2>&1
if not errorlevel 1 (
  for %%V in (3.14 3.13 3.12 3.11 3.10) do (
    if not defined PYTHON (
      py -%%V -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
      if not errorlevel 1 set "PYTHON=py -%%V"
    )
  )
)
if not defined PYTHON (
  where python >nul 2>&1
  if not errorlevel 1 set "PYTHON=python"
)
if not defined PYTHON (
  echo Python nao encontrado. Instale Python 3.12, 3.13 ou 3.14 de 64 bits.
  pause
  exit /b 1
)
%PYTHON% -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,14) and sys.maxsize > 2**32" >nul 2>&1
if errorlevel 1 (
  echo O Python encontrado nao e compativel. Use Python 3.12, 3.13 ou 3.14 de 64 bits.
  pause
  exit /b 1
)
echo Interpretador selecionado:
%PYTHON% --version

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Criando ambiente virtual...
  %PYTHON% -m venv .venv
  if errorlevel 1 goto falha
)
set "PY=.venv\Scripts\python.exe"
%PY% -c "import pygame,mediapipe,cv2" >nul 2>&1
if errorlevel 1 (
  echo [2/3] Instalando bibliotecas... ^(requer internet no primeiro uso^)
  %PY% -m pip install --upgrade pip
  if errorlevel 1 goto falha
  %PY% -m pip install -r requirements.txt
  if errorlevel 1 goto falha
)
echo [3/3] Abrindo laboratorio MediaPipe Interativo...
%PY% mediapipe_interativo.py
if errorlevel 1 goto falha
exit /b 0
:falha
echo.
echo Erro. Veja a mensagem acima.
pause
exit /b 1
