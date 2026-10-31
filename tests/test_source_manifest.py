"""Source provenance must remain distinguishable from unverified plans in Excel."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "manifest_export", ROOT / "pipeline/input_preparation/indicator_construction/index_definition_to_excel.py"
)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class SourceManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads((ROOT / "pipeline/definitions/data_sources_manifest.json").read_text(encoding="utf-8"))
        cls.indices = json.loads((ROOT / "pipeline/definitions/indices_manifest.json").read_text(encoding="utf-8"))

    def test_current_inputs_have_sources_and_plans_are_separate(self):
        tables = export.build_tables(self.indices, sources_manifest=self.registry)
        self.assertTrue(tables["Source_Coverage"]["mapping_status"].eq("mapped").all())
        self.assertTrue(tables["Indices_Summary"]["source_ids"].str.len().gt(0).all())
        planned = tables["Data_Sources"].query("usage_status == 'planned_only'")
        self.assertFalse(planned["verified"].any())
        self.assertTrue(planned["raster_folders"].eq("").all())

    def test_invalid_source_mapping_fails(self):
        for mutation in ("duplicate", "unknown_folder", "planned_mapping", "unknown_evidence"):
            registry = copy.deepcopy(self.registry)
            row = registry["sources"][0]
            if mutation == "duplicate":
                registry["sources"].append(copy.deepcopy(row))
            elif mutation == "unknown_folder":
                row["raster_folders"] = ["typo"]
            elif mutation == "unknown_evidence":
                row["evidence_ids"] = ["typo"]
            else:
                row["usage_status"] = "planned_only"
                row["raster_folders"] = ["NTL"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                export.build_source_tables(registry, self.indices)

    def test_excel_preserves_unknowns_booleans_links_and_literal_text(self):
        registry = copy.deepcopy(self.registry)
        registry["sources"][0]["notes"] = "=literal source note"
        registry["sources"][0]["rationale"] = None
        tables = export.build_tables(self.indices, sources_manifest=registry)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.xlsx"
            export.write_excel(tables, path)
            book = load_workbook(path)
            sheet = book["Data_Sources"]
            columns = {cell.value: cell.column for cell in sheet[1]}
            self.assertIs(sheet.cell(2, columns["verified"]).value, True)
            self.assertIsNone(sheet.cell(2, columns["rationale"]).value)
            self.assertEqual(sheet.cell(2, columns["notes"]).data_type, "s")
            self.assertEqual(sheet.cell(2, columns["link"]).hyperlink.target, registry["sources"][0]["link"])
            self.assertEqual(sheet.max_row, len(registry["sources"]) + 1)
            self.assertEqual(sheet.freeze_panes, "B2")
            self.assertTrue(sheet.auto_filter.ref)
            self.assertIn("Source_Evidence", book.sheetnames)
            book.close()


if __name__ == "__main__":
    unittest.main()
