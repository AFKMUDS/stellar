@echo off
setlocal
cd /d "%~dp0"
set "RAP_TRAIN_CONFIG=configs\laptop.json"
if exist "runs\rap\config.json" set "RAP_TRAIN_CONFIG=runs\rap\config.json"
if "%~1"=="" goto localdata
python train.py --config "%RAP_TRAIN_CONFIG%" --dataset "%~1" --resume auto
goto finished
:localdata
python train.py --config "%RAP_TRAIN_CONFIG%" --resume auto
:finished
if errorlevel 1 echo Training stopped with an error. Read the message above.
pause

