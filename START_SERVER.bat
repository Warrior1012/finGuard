@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Training ML model (first time only)...
python model/train_model.py
echo.
echo Starting FinGuard server...
echo Open browser: http://localhost:5000
echo.
python app.py
pause
