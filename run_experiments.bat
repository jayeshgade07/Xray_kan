@echo off
echo ==========================================
echo X-Ray KAN Project - Experiment Pipeline
echo ==========================================

echo [Step 1] Initializing Data...
python data/datasets/nih_preprocess.py

echo.
echo [Step 2] Training Baseline CNN Model...
python train.py --model_type cnn --epochs 5

echo.
echo [Step 3] Training Dense MLP Model...
python train.py --model_type dense --epochs 5

echo.
echo [Step 4] Training KAN Model...
python train.py --model_type kan --epochs 5

echo.
echo [Step 5] Evaluating All Models...
python evaluate.py --model_type cnn --load_ckpt
python evaluate.py --model_type dense --load_ckpt
python evaluate.py --model_type kan --load_ckpt

echo.
echo All experiments finished. Check results/metrics/evaluation_summary.csv for comparison!
pause
