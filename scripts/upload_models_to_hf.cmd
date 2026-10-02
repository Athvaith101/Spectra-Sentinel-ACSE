@echo off
setlocal
if "%HF_MODEL_REPO%"=="" (
  echo Set HF_MODEL_REPO first, for example:
  echo   set HF_MODEL_REPO=Athvaith/sih26055-ew-final-model
  exit /b 1
)
python -m pip install -U huggingface_hub
python scripts\upload_models_to_hf.py
