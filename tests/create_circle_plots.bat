@echo off
REM Create circle plots with custom parameters

python -m main.create_circle_plot ^
  --input-dir data/semaglutide/plus50 ^
  --group-label-radius -2.0 ^
  --max-bar-radius 0.4 ^
  --show-separator-circle ^
  --separator-circle-margin 0.05

echo.
echo Plus50 plot complete!
echo.

python -m main.create_circle_plot ^
  --input-dir data/semaglutide/cvd ^
  --group-label-radius -2.0 ^
  --max-bar-radius 0.4 ^
  --show-separator-circle ^
  --separator-circle-margin 0.05

echo.
echo CVD plot complete!
echo.

python -m main.create_circle_plot ^
  --input-dir data/semaglutide/diab ^
  --group-label-radius -2.0 ^
  --max-bar-radius 0.4 ^
  --show-separator-circle ^
  --separator-circle-margin 0.05

echo.
echo DIAB plot complete!
echo.
echo All circle plots generated!
pause

