from __future__ import annotations

import unittest
import zipfile
from io import BytesIO

from paper_reader.exporting.exporters import markdown_to_docx


class ExportingTests(unittest.TestCase):
    def test_docx_removes_invalid_xml_control_characters(self) -> None:
        document = markdown_to_docx("valid\x00text\x0bmore")

        with zipfile.ZipFile(BytesIO(document)) as archive:
            xml = archive.read("word/document.xml")

        self.assertIn(b"validtextmore", xml)
        self.assertNotIn(b"\x00", xml)
        self.assertNotIn(b"\x0b", xml)


if __name__ == "__main__":
    unittest.main()
