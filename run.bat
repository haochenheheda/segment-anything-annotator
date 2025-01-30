@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
".venv\Scripts\python.exe" annotator.py --app_resolution 1000,1600 --model_type vit_b --keep_input_size True --max_size 720
pause
