"""Structural checks on docs/index.html.

These do not execute the JavaScript (no browser or JS engine here, standard
library only). They pin the things a hand-edit of the page could break without
any Python test noticing: the function names TestJavaScriptParity's fixture
note implicitly relies on staying put in the JS, and two house-style rules
that apply to every page in this research estate (no em dash, no
externally-hosted script beyond the Google Fonts stylesheet).
"""

import re
import unittest
from pathlib import Path

DOCS_PAGE = Path(__file__).resolve().parents[1] / "docs" / "index.html"

# Every function name tests/parity_cases.json's arithmetic (evaluate) or the
# calculator's own state management depends on the JS keeping.
REQUIRED_FUNCTION_NAMES = [
    "evaluate",
    "addRow",
    "groupTotal",
    "loadPreset",
    "loadExample",
    "readInputs",
    "renderCards",
    "render",
]


class TestDocsPageStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = DOCS_PAGE.read_text(encoding="utf-8")

    def test_page_exists(self):
        self.assertTrue(DOCS_PAGE.is_file())

    def test_every_parity_relevant_function_name_is_present(self):
        for name in REQUIRED_FUNCTION_NAMES:
            with self.subTest(function=name):
                self.assertRegex(
                    self.html,
                    r"function\s+" + re.escape(name) + r"\s*\(",
                    f"docs/index.html no longer defines function {name}(); "
                    "the Python parity fixture assumes this arithmetic entry "
                    "point still exists under this name.",
                )

    def test_no_em_dash(self):
        self.assertNotIn("—", self.html)

    def test_no_external_script(self):
        # A <script src="..."> pulling code from anywhere would break the
        # "no build step, nothing but Google Fonts" promise the page makes.
        for match in re.finditer(r"<script\b[^>]*>", self.html, flags=re.IGNORECASE):
            self.assertNotIn("src=", match.group(0).lower())

    def test_only_google_fonts_is_an_external_stylesheet(self):
        hrefs = re.findall(
            r'<link[^>]+rel="stylesheet"[^>]+href="([^"]+)"', self.html
        )
        for href in hrefs:
            self.assertTrue(
                href.startswith("https://fonts.googleapis.com/"),
                f"unexpected external stylesheet: {href}",
            )

    def test_story_section_present(self):
        self.assertIn('id="story"', self.html)
        for anchor in ("Step 1", "Step 2", "Step 3"):
            self.assertIn(anchor, self.html)

    def test_shareable_state_functions_present(self):
        for name in ("encodeState", "decodeState", "syncHash", "applyState"):
            self.assertRegex(self.html, r"function\s+" + re.escape(name) + r"\s*\(")

    def test_copy_link_control_present(self):
        self.assertIn('id="copy-link"', self.html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
