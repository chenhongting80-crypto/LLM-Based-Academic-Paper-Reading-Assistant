from __future__ import annotations

import inspect
import os
import unittest
from unittest.mock import patch

os.environ["PAPER_READER_SKIP_STREAMLIT_UI"] = "1"

import main


class ReadingCardViewTests(unittest.TestCase):
    def test_view_has_close_without_export(self) -> None:
        record = {
            "file_name": "paper.pdf",
            "created_at": None,
            "updated_at": None,
            "card": {},
        }
        with patch.object(main, "st") as streamlit:
            streamlit.button.return_value = True
            main.view_reading_card_dialog.__wrapped__(record)

        streamlit.download_button.assert_not_called()
        streamlit.button.assert_called_once_with("Close")
        streamlit.rerun.assert_called_once_with()

    def test_independent_export_page_still_exists(self) -> None:
        source = inspect.getsource(main.app)
        self.assertIn('"Compare Papers", "Export"', source)
        self.assertIn('st.subheader("Export")', source)


if __name__ == "__main__":
    unittest.main()
