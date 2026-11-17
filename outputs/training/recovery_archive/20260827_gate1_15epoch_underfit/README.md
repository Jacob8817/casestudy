# Archived 15-epoch Gate-1 calibration run

This run was numerically valid and completed all 480 optimizer steps, but the
15-epoch cosine schedule reached zero learning rate before the fixed patch bank
was memorized. It achieved 68.68% loss reduction and 0.3739 represented-class
macro Dice, below the predeclared 80% and 0.70 acceptance thresholds.

A fresh train-only calibration with the same seed, 32 patches, model, loss,
optimizer, learning rate, and AMP scale reached 81.55% loss reduction and
0.8915 macro Dice at epoch 30 under a 50-epoch cosine schedule. At epoch 50 it
reached 84.51% loss reduction and 0.9463 macro Dice, with every represented
organ above 0.83 Dice. The authorized diagnostic is therefore now 50 epochs
with at least 1,500 optimizer steps. The research baseline remains 100 epochs.

The cached preprocessed arrays were preserved and remain reusable.
