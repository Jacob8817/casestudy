from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.dataset.label_mapping import load_organ_schema


class LabelMappingTests(unittest.TestCase):
    def test_project_schema_has_expected_channel_and_source_counts(self) -> None:
        schema = load_organ_schema(Path("configs/organs_v1.yaml"))
        self.assertEqual(len(schema.labels), 10)
        self.assertEqual(len(schema.source_masks), 13)
        self.assertEqual(schema.num_output_channels, 11)
        self.assertEqual(schema.overlap_policy, "priority")
        self.assertEqual(schema.overlap_priority[-2:], ("aorta", "inferior_vena_cava"))
        self.assertEqual(schema.labels[8].source_masks, ("lung_upper_lobe_left", "lung_lower_lobe_left"))

    def test_duplicate_source_assignment_is_rejected(self) -> None:
        text = """
schema_version: 1
name: invalid
description: test
background: {id: 0, name: background}
merge_mode: union
overlap_policy: error
labels:
  - {id: 1, name: first, source_masks: [same]}
  - {id: 2, name: second, source_masks: [same]}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source mask"):
                load_organ_schema(path)


if __name__ == "__main__":
    unittest.main()
