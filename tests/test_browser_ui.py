import unittest

from fastapi.testclient import TestClient

from app import app


class BrowserUiTests(unittest.TestCase):
    def test_root_page_serves_browser_ui(self) -> None:
        client = TestClient(app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertTrue("Retail POS" in response.text)
        self.assertIn("Learning", response.text)
        self.assertIn("Catalog", response.text)
        self.assertIn("Train model", response.text)


if __name__ == "__main__":
    unittest.main()
