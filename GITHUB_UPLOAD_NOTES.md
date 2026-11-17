# GitHub upload contents

This repository includes project source, tests, configuration, dependency lists,
notebooks with saved outputs, the dataset split manifest, audit reports, result metadata,
figures, interactive HTML visualizations, trained models, and all result artifacts under
`outputs/`. The separate `Single_Patient_Study/` directory and its ZIP archive are included.

Model checkpoints, volume arrays, segmentation files, meshes, DICOM input files, and ZIP
archives are stored using Git Large File Storage (LFS). To download their actual contents,
install Git LFS and run:

```sh
git lfs install
git clone https://github.com/Jacob8817/casestudy.git
cd casestudy
git lfs pull
```

## Intentionally excluded

- `Totalsegmentator_dataset_v201/` and its ZIP archive, as requested.
- `data/cache/`: regenerable processed dataset caches.
- `.venv/`, `.tmp/`, Python bytecode, notebook checkpoints, and test caches: local environment and temporary files.
- Local credential files, if any.

All original files remain on the source computer at `F:\csstdy`.
Uploading does not delete or modify the original project files.

The original dataset source and project workflow are documented in
`PROJECT_CONTEXT_FOR_CHATGPT.md`. Recreate the Python environment using `requirements.txt`
and `requirements-gpu.txt`, and supply the excluded dataset before running workflows
that depend on it. Existing research reports describe prior local runs.
