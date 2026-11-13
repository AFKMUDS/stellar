@echo off
setlocal
cd /d "%~dp0"
echo Use your existing Miniconda environment if it already has CUDA PyTorch.
python -c "import sys; assert sys.version_info >= (3,10); print(sys.executable)"
if errorlevel 1 goto failed
python -c "import torch" >nul 2>nul
if not errorlevel 1 goto dependencies
echo Installing PyTorch 2.5.1 for CUDA 12.1. Existing installations are retained.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto failed
:dependencies
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
python verify.py --quick
if errorlevel 1 goto failed
echo Setup complete. Drag a dataset onto TRAIN_WINDOWS.bat.
pause
exit /b 0
:failed
echo Setup did not complete. Read the error above and use the Miniconda environment with your working PyTorch.
pause
exit /b 1

