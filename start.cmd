@echo off
setlocal
pushd "%~dp0project"
if errorlevel 1 exit /b 1
python -B stereo_live.py --height-map --height-max 0.5 --height-gamma 0.5 --change-mm 10 --size 640x360 --update-mode changes --check-interval 0.5 --update-interval 1 --projector-inset-x 268 --projector-inset-y 40 --fill surface --appearance smooth --near 0.5 --far 2.5 --lan-preview --lan-preview-port 8765 %*
set "ar_result=%errorlevel%"
popd
if not "%ar_result%"=="0" (
    echo The program stopped with an error. See the message above.
    pause
)
exit /b %ar_result%
