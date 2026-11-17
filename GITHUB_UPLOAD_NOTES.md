# GitHub upload contents

This repository contains the project source code, tests, configuration, dependency lists,
notebooks with their saved outputs, dataset split manifest, audit reports, result metadata,
figures, and interactive HTML visualizations.

The original project files remain on the source computer at `F:\csstdy`.
This repository is not yet a full backup of that folder.

## Files remaining local

- `Totalsegmentator_dataset_v201/` and its ZIP archive: original dataset.
- `Single_Patient_Study/` and its ZIP archive: patient input data.
- `data/cache/`: processed arrays and caches.
- Binary result artifacts under `outputs/`: model checkpoints, volume arrays, segmentation files, and mesh files.
- `.venv/`, `.tmp/`, Python bytecode, and test caches: local environment and temporary files.

Large-data upload/storage is pending a separate choice. GitHub blocks ordinary Git files
larger than 100 MiB. Some files here exceed that limit, including the dataset ZIP archive.

The original dataset source and project workflow are documented in
`PROJECT_CONTEXT_FOR_CHATGPT.md`. Recreate the Python environment using `requirements.txt`
and `requirements-gpu.txt`, then supply the required datasets and checkpoints before
running workflows that depend on them. Existing research reports describe prior local runs;
they do not imply that this repository contains every input or trained model.
