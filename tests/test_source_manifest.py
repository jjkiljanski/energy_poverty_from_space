"""Source and prepared-artifact manifests export to separate, reviewable workbooks."""
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
        cls.preparation = json.loads((ROOT / "pipeline/definitions/data_preparation_manifest.json").read_text(encoding="utf-8"))

    def test_manifest_keeps_requested_fields_and_groups_sources(self):
        groups = self.registry["sources"]
        self.assertEqual(set(groups), {"used", "planned_only"})
        self.assertEqual((len(groups["used"]), len(groups["planned_only"])), (21, 18))
        for source in groups["used"]:
            self.assertEqual(list(source), export.SOURCE_COLUMNS)
            self.assertIsInstance(source["notes"], str)
            self.assertNotIn("source_description", source)
            self.assertNotIn("verification_scope", source)
        for source in groups["planned_only"]:
            self.assertEqual(list(source), export.NOT_USED_SOURCE_COLUMNS)
            self.assertEqual(source["why_not_used"], "")
            self.assertIsInstance(source["notes"], str)
        planned_ids = {source["id"] for source in groups["planned_only"]}
        self.assertTrue({"emc_greenness", "ghs_smod", "planned_sentinel3_lst"}.issubset(planned_ids))

    def test_source_and_artifact_tables_validate_ids(self):
        source_tables = export.build_source_tables(self.registry)
        self.assertEqual(list(source_tables["Data_Sources"].columns), export.SOURCE_COLUMNS)
        self.assertEqual(list(source_tables["Not_Used_Data_Sources"].columns), export.NOT_USED_SOURCE_COLUMNS)
        artifacts = export.build_artifact_table(self.preparation, self.registry)
        self.assertEqual(len(artifacts), len(self.preparation["artifacts"]))
        self.assertIn("source_description", artifacts.columns)

        invalid = copy.deepcopy(self.registry)
        invalid["sources"]["used"][0]["verified"] = "yes"
        with self.assertRaisesRegex(ValueError, "boolean verified"):
            export.build_source_tables(invalid)

        invalid_preparation = copy.deepcopy(self.preparation)
        invalid_preparation["artifacts"][0]["source_ids"] = ["unknown_source"]
        with self.assertRaisesRegex(ValueError, "unknown source IDs"):
            export.build_artifact_table(invalid_preparation, self.registry)

    def test_workbooks_are_split_and_planned_sources_are_below_used_sources(self):
        registry = copy.deepcopy(self.registry)
        registry["sources"]["used"][0]["notes"] = "=literal source note"
        registry["sources"]["used"][0]["rationale"] = None
        source_tables = export.build_source_tables(registry)
        artifacts = export.build_artifact_table(self.preparation, registry)
        index_tables = export.build_tables(self.indices)

        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "indices_manifest.xlsx"
            source_path = Path(directory) / "data_sources.xlsx"
            export.write_excel(index_tables, index_path)
            export.write_data_sources_excel(source_tables, artifacts, source_path)

            index_book = load_workbook(index_path, data_only=False)
            self.assertFalse({"Project", "Paths_Config", "Source_Evidence", "Source_Field_Guide", "Data_Sources"} & set(index_book.sheetnames))
            self.assertNotIn("input_source_years", {cell.value for cell in index_book["Indices_Summary"][1]})
            self.assertIn("Folder_Catalog", index_book.sheetnames)
            self.assertNotIn("FolderCatalog", index_book.sheetnames)
            self.assertEqual(
                [cell.value for cell in index_book["Folder_Catalog"][1]],
                ["folder", "description", "provenience", "folder_path", "tiff_crs", "tiff_resolution", "tiff_resolution_unit", "tiff_pixel_area", "sample_tiff"],
            )
            folder_rows = {
                row[0]: row
                for row in index_book["Folder_Catalog"].iter_rows(min_row=2, values_only=True)
            }
            self.assertIn("GHS-AGE", folder_rows["Age"][2])
            self.assertIn("Google Colab/GEE", folder_rows["ERA5-Land_HDD"][2])
            self.assertIn("ln(x + 2)", folder_rows["NTL"][2])
            index_book.close()

            source_book = load_workbook(source_path, data_only=False)
            self.assertEqual(set(source_book.sheetnames), {"README", "Data_Sources", "Not_Used_Data_Sources", "Artifacts"})
            sheet = source_book["Data_Sources"]
            self.assertEqual(sheet.cell(1, 1).value, "Sources used by the current pipeline")
            self.assertEqual([sheet.cell(2, col).value for col in range(1, len(export.SOURCE_COLUMNS) + 1)], export.SOURCE_COLUMNS)
            first_source_row = 3
            columns = {cell.value: cell.column for cell in sheet[2]}
            self.assertIs(sheet.cell(first_source_row, columns["verified"]).value, True)
            self.assertIsNone(sheet.cell(first_source_row, columns["rationale"]).value)
            self.assertEqual(sheet.cell(first_source_row, columns["notes"]).data_type, "s")
            self.assertEqual(sheet.cell(first_source_row, columns["link"]).hyperlink.target, registry["sources"]["used"][0]["link"])
            self.assertEqual(source_book["Not_Used_Data_Sources"].cell(1, 1).value, "Sources not used by the current pipeline")
            self.assertIn("not selected", source_book["Not_Used_Data_Sources"].cell(2, 1).value)
            self.assertEqual(source_book["Not_Used_Data_Sources"].cell(3, 1).value, "id")
            self.assertEqual(set(sheet.tables), {"UsedSources"})
            self.assertEqual(set(source_book["Not_Used_Data_Sources"].tables), {"NotUsedSources"})
            not_used_columns = {cell.value: cell.column for cell in source_book["Not_Used_Data_Sources"][3]}
            self.assertIsNone(source_book["Not_Used_Data_Sources"].cell(4, not_used_columns["why_not_used"]).value)
            self.assertEqual(source_book["Artifacts"].max_row, len(artifacts) + 1)
            source_book.close()


if __name__ == "__main__":
    unittest.main()
