"""Regression tests for FinQA cleaning transformations."""

import unittest

from src.data.cleaning import (
    normalize_answer_types,
    normalize_table_fields,
    normalize_text_fields,
    remove_invalid_samples,
)


def make_raw_dataset() -> dict[str, list[dict]]:
    """Create a small representative FinQA-shaped dataset for tests."""
    return {
        "train": [
            {
                "pre_text": ["First paragraph.", "Second paragraph."],
                "post_text": ["Closing paragraph."],
                "table": [["Metric", "2024"], ["Revenue", "$100"]],
                "qa": {"question": "What was revenue?", "answer": "$100", "exe_ans": 380.0},
            },
            {
                "pre_text": [],
                "post_text": ["Yes", "No"],
                "table": [["Answer"], ["yes"]],
                "qa": {"question": "Was the target met?", "answer": "yes", "exe_ans": "yes"},
            },
        ],
        "validation": [
            {
                "pre_text": ["Validation context."],
                "post_text": [],
                "table": [["Metric", "2023"], ["Revenue", "$90"]],
                "qa": {"question": "What is the count?", "answer": "5", "exe_ans": 5},
            }
        ],
    }


class NormalizeAnswerTypesTests(unittest.TestCase):
    """Tests for executable-answer normalization."""

    def test_converts_each_executable_answer_to_string(self) -> None:
        cleaned_data = normalize_answer_types(make_raw_dataset())

        answers = [
            record["qa"]["exe_ans"]
            for records in cleaned_data.values()
            for record in records
        ]

        self.assertEqual(answers, ["380.0", "yes", "5"])
        self.assertTrue(all(isinstance(answer, str) for answer in answers))

    def test_preserves_raw_data(self) -> None:
        raw_data = make_raw_dataset()
        normalize_answer_types(raw_data)

        self.assertEqual(raw_data["train"][0]["qa"]["exe_ans"], 380.0)


class NormalizeTextFieldsTests(unittest.TestCase):
    """Tests for context paragraph normalization."""

    def test_joins_paragraphs_with_blank_lines(self) -> None:
        normalized_data = normalize_text_fields(make_raw_dataset())

        first_record = normalized_data["train"][0]
        second_record = normalized_data["train"][1]
        self.assertEqual(
            first_record["pre_text"], "First paragraph.\n\nSecond paragraph."
        )
        self.assertEqual(first_record["post_text"], "Closing paragraph.")
        self.assertEqual(second_record["pre_text"], "")
        self.assertEqual(second_record["post_text"], "Yes\n\nNo")

    def test_preserves_raw_text_lists(self) -> None:
        raw_data = make_raw_dataset()
        normalize_text_fields(raw_data)

        self.assertEqual(
            raw_data["train"][0]["pre_text"],
            ["First paragraph.", "Second paragraph."],
        )
        self.assertIsInstance(raw_data["validation"][0]["post_text"], list)


class NormalizeTableFieldsTests(unittest.TestCase):
    """Tests for Markdown table normalization."""

    def test_converts_table_to_markdown(self) -> None:
        normalized_data = normalize_table_fields(make_raw_dataset())

        self.assertEqual(
            normalized_data["train"][0]["table"],
            "| Metric | 2024 |\n"
            "| --- | --- |\n"
            "| Revenue | $100 |",
        )

    def test_escapes_markdown_delimiters(self) -> None:
        raw_data = make_raw_dataset()
        raw_data["train"][0]["table"][1][0] = "Profit | loss\nvalue"

        normalized_data = normalize_table_fields(raw_data)

        self.assertIn("Profit \\| loss<br>value", normalized_data["train"][0]["table"])

    def test_preserves_raw_table_lists(self) -> None:
        raw_data = make_raw_dataset()
        normalize_table_fields(raw_data)

        self.assertEqual(raw_data["train"][0]["table"], [["Metric", "2024"], ["Revenue", "$100"]])
        self.assertIsInstance(raw_data["validation"][0]["table"], list)


class RemoveInvalidSamplesTests(unittest.TestCase):
    """Tests for FinQA record validation and filtering."""

    def test_removes_records_missing_required_content(self) -> None:
        raw_data = make_raw_dataset()
        raw_data["train"].extend(
            [
                {"pre_text": ["Context"], "post_text": [], "table": [["A"]], "qa": {"answer": "1"}},
                {"pre_text": ["Context"], "post_text": [], "table": [], "qa": {"question": "Q", "answer": "1"}},
                {"pre_text": [], "post_text": [], "table": [["A"]], "qa": {"question": "Q", "answer": "1"}},
                {"pre_text": ["Context"], "post_text": [], "table": [["A"]], "qa": {"question": "Q", "answer": ""}},
            ]
        )

        filtered_data = remove_invalid_samples(raw_data)

        self.assertEqual(len(filtered_data["train"]), 2)

    def test_keeps_executable_answer_when_human_answer_is_blank(self) -> None:
        raw_data = make_raw_dataset()
        raw_data["train"][0]["qa"]["answer"] = ""
        raw_data["train"][0]["qa"]["exe_ans"] = "yes"

        filtered_data = remove_invalid_samples(raw_data)

        self.assertEqual(len(filtered_data["train"]), 2)

    def test_preserves_original_records(self) -> None:
        raw_data = make_raw_dataset()
        filtered_data = remove_invalid_samples(raw_data)
        filtered_data["train"][0]["qa"]["question"] = "Changed"

        self.assertEqual(raw_data["train"][0]["qa"]["question"], "What was revenue?")


if __name__ == "__main__":
    unittest.main()
