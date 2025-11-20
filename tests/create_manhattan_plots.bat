@echo off
REM Create Manhattan prevalence plots and volcano plots with custom parameters

REM ============================================================================
REM Manhattan Plots
REM ============================================================================

python -m main.create_manhattan_plot ^
  --input-dir data/semaglutide/plus50 ^
  --output-dir figures ^
  --method IPW ^
  --adjust bh ^
  --arm-pooling inter_intra_variance ^
  --annotate-top 5

echo.
echo Plus50 Manhattan plot complete!
echo.

python -m main.create_manhattan_plot ^
  --input-dir data/semaglutide/diab ^
  --output-dir figures ^
  --method IPW ^
  --adjust bh ^
  --arm-pooling inter_intra_variance ^
  --annotate-top 5

echo.
echo DIAB Manhattan plot complete!
echo.

REM ============================================================================
REM Volcano Plots
REM ============================================================================

python -m main.create_volcano_plot ^
  --input-dir data/semaglutide/plus50 ^
  --output-dir figures ^
  --effect-type log-RR ^
  --adjust bh ^
  --arm-pooling inter_intra_variance ^
  --diagnostics ^
  --verbose

echo.
echo Plus50 volcano plot complete!
echo.

python -m main.create_volcano_plot ^
  --input-dir data/semaglutide/diab ^
  --output-dir figures ^
  --effect-type log-RR ^
  --adjust bh ^
  --arm-pooling inter_intra_variance ^
  --diagnostics ^
  --verbose

echo.
echo DIAB volcano plot complete!
echo.

REM ============================================================================
REM Example with filtering for Manhattan plots (uncomment and modify as needed):
REM ============================================================================
REM python -m main.create_manhattan_plot ^
REM   --input-dir data/semaglutide ^
REM   --output-dir figures ^
REM   --method IPW ^
REM   --adjust bh ^
REM   --arm-pooling inter_intra_variance ^
REM   --annotate-top 5 ^
REM   --exclude-outcomes "A10BJ,A10BK" ^
REM   --exclude-groups "V,W"

REM ============================================================================
REM Example volcano plot with fast mode (skip extra plots, uncomment to use):
REM ============================================================================
REM python -m main.create_volcano_plot ^
REM   --input-dir data/semaglutide ^
REM   --output-dir figures ^
REM   --effect-type log-RR ^
REM   --adjust bh ^
REM   --arm-pooling inter_intra_variance ^
REM   --fast

REM ============================================================================
REM Example volcano plot with minimum prevalence filter (uncomment to use):
REM ============================================================================
REM python -m main.create_volcano_plot ^
REM   --input-dir data/semaglutide ^
REM   --output-dir figures ^
REM   --effect-type log-RR ^
REM   --adjust bh ^
REM   --arm-pooling inter_intra_variance ^
REM   --min-prevalence 0.01 ^
REM   --diagnostics

REM ============================================================================
REM Example volcano plot with RD (Risk Difference) instead of RR:
REM ============================================================================
REM python -m main.create_volcano_plot ^
REM   --input-dir data/semaglutide ^
REM   --output-dir figures ^
REM   --effect-type RD ^
REM   --adjust bh ^
REM   --arm-pooling inter_intra_variance ^
REM   --diagnostics

echo.
echo ============================================================================
echo All Manhattan and volcano plots generated!
echo ============================================================================
pause


