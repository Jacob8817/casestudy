# Archived Gate-1 attempt

This directory preserves the aborted epoch-zero Gate-1 checkpoint and its
prewarm receipt from the run that used PyTorch's default AMP initial scale of
65536. The first real epoch stopped before optimizer batch 14 because the
scaled FP16 gradient became non-finite.

The training contract now explicitly uses an AMP initial scale of 1024. A
throwaway real-data diagnostic completed all 32 Gate-1 batches at that scale
with 32 optimizer steps, a final scale of 1024, and no non-finite gradients.
The cached preprocessed arrays were not removed and remain reusable.
