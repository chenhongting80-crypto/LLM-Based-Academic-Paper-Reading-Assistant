from __future__ import annotations

import inspect
import os
import unittest

os.environ["PAPER_READER_SKIP_STREAMLIT_UI"] = "1"

from main import (
    EXPORT_CONTENT_WIDGET_PREFIX,
    EXPORT_STATE_DEFAULTS,
    clear_export_selection,
    export_download_ready,
    export_step_ready,
    reset_export_state,
    set_export_step,
    sync_export_content_type,
)


class ExportFlowStateTests(unittest.TestCase):
    def _complete_state(self, step: int = 4) -> dict:
        return {
            "workspace_id": "11111111-1111-1111-1111-111111111111",
            "database_data": {"papers": ["preserve"]},
            "other_page_state": "preserve",
            "export_step": step,
            "export_content_type": "Reading Cards",
            "export_content_type_reset": 0,
            "export_selected_ids": ["paper-1", "paper-2"],
            "export_format": "PDF",
            "export_options": {"include_sources": True},
            "export_generated_file": {
                "file_name": "reading_cards.pdf",
                "data": b"content",
                "mime": "application/pdf",
            },
            "export_status_message": "Export generated.",
            "export_error_message": "old error",
            "export_content_type_widget_0": "Reading Cards",
            "export_format_widget": "PDF",
            "export_card_paper-1": True,
            "export_card_paper-2": True,
            "export_download_generated": True,
        }

    def test_step_one_starts_without_selection(self) -> None:
        state = {key: value.copy() if isinstance(value, (dict, list)) else value for key, value in EXPORT_STATE_DEFAULTS.items()}
        self.assertEqual(state["export_step"], 1)
        self.assertIsNone(state["export_content_type"])
        self.assertFalse(export_step_ready(1, state))

    def test_each_step_can_clear_its_required_selection(self) -> None:
        state = self._complete_state()
        clear_export_selection(1, state)
        self.assertIsNone(state["export_content_type"])
        self.assertFalse(export_step_ready(1, state))

        state = self._complete_state()
        clear_export_selection(2, state)
        self.assertEqual(state["export_selected_ids"], [])
        self.assertNotIn("export_card_paper-1", state)
        self.assertFalse(export_step_ready(2, state))

        state = self._complete_state()
        clear_export_selection(3, state)
        self.assertEqual(state["export_format"], "")
        self.assertFalse(export_step_ready(3, state))

        state = self._complete_state()
        clear_export_selection(2, state)
        clear_export_selection(3, state)
        self.assertFalse(export_step_ready(4, state))

    def test_missing_required_selection_disables_each_next_action(self) -> None:
        state = self._complete_state()
        for step, key, empty_value in (
            (1, "export_content_type", ""),
            (2, "export_selected_ids", []),
            (3, "export_format", ""),
        ):
            test_state = self._complete_state()
            test_state[key] = empty_value
            self.assertFalse(export_step_ready(step, test_state))
            self.assertFalse(export_step_ready(4, test_state))
        self.assertTrue(export_step_ready(4, state))
        self.assertTrue(export_download_ready(state))
        state["export_format"] = ""
        self.assertFalse(export_download_ready(state))

    def test_back_changes_only_step_and_preserves_selections(self) -> None:
        state = self._complete_state()
        expected = {
            "export_content_type": state["export_content_type"],
            "export_selected_ids": list(state["export_selected_ids"]),
            "export_format": state["export_format"],
        }
        set_export_step(3, state)
        self.assertEqual(state["export_step"], 3)
        self.assertEqual({key: state[key] for key in expected}, expected)

    def test_cancel_from_any_step_resets_only_export_flow(self) -> None:
        for step in range(1, 5):
            state = self._complete_state(step)
            reset_export_state(state)
            self.assertEqual(state["export_step"], 1)
            self.assertIsNone(state["export_content_type"])
            self.assertEqual(state["export_selected_ids"], [])
            self.assertEqual(state["export_format"], "")
            self.assertEqual(state["export_options"], {})
            self.assertIsNone(state["export_generated_file"])
            self.assertEqual(state["export_status_message"], "")
            self.assertEqual(state["export_error_message"], "")
            self.assertNotIn("export_download_generated", state)
            self.assertEqual(state["workspace_id"], "11111111-1111-1111-1111-111111111111")
            self.assertEqual(state["database_data"], {"papers": ["preserve"]})
            self.assertEqual(state["other_page_state"], "preserve")

    def test_start_over_uses_complete_reset(self) -> None:
        state = self._complete_state()
        reset_export_state(state)
        self.assertEqual(
            {key: state[key] for key in EXPORT_STATE_DEFAULTS},
            EXPORT_STATE_DEFAULTS,
        )
        self.assertEqual(state["workspace_id"], "11111111-1111-1111-1111-111111111111")
        self.assertEqual(state["database_data"], {"papers": ["preserve"]})

    def test_content_radio_and_buttons_share_canonical_state(self) -> None:
        state = {
            **EXPORT_STATE_DEFAULTS,
            "export_content_type_reset": 0,
        }
        widget_key = f"{EXPORT_CONTENT_WIDGET_PREFIX}0"
        self.assertIsNone(state["export_content_type"])
        self.assertFalse(export_step_ready(1, state))

        state[widget_key] = "Reading Cards"
        sync_export_content_type(widget_key, state)
        self.assertEqual(state["export_content_type"], "Reading Cards")
        self.assertTrue(export_step_ready(1, state))

        unchanged = dict(state)
        self.assertTrue(export_step_ready(1, state))
        self.assertEqual(state, unchanged)

        clear_export_selection(1, state)
        self.assertIsNone(state["export_content_type"])
        self.assertNotIn(widget_key, state)
        self.assertEqual(state["export_content_type_reset"], 1)
        self.assertFalse(export_step_ready(1, state))

    def test_progress_separator_is_ascii(self) -> None:
        source = inspect.getsource(__import__("main").app)
        self.assertIn('progress = " > ".join(', source)
        mojibake_arrow = "".join(chr(codepoint) for codepoint in (0x00E2, 0x2020, 0x2019))
        self.assertNotIn(mojibake_arrow, source)
        self.assertNotIn(chr(0x2192), source)


if __name__ == "__main__":
    unittest.main()
