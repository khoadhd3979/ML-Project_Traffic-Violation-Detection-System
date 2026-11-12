import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.database_manager import DatabaseManager
from src.api.main import health_check, get_stats_overview, get_violations, lookup_vehicle
from app.services.api_client import APIClient


class TestDatabaseAndAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DatabaseManager()

    def test_database_stats(self):
        stats = self.db.get_statistics()
        self.assertIn("total_violations", stats)
        self.assertIn("violations_today", stats)
        self.assertGreaterEqual(stats["total_violations"], 0)

    def test_database_query_violations(self):
        violations = self.db.get_violations(limit=5)
        self.assertIsInstance(violations, list)

    def test_api_health(self):
        h = health_check()
        self.assertEqual(h["status"], "healthy")
        self.assertTrue(h["database"])

    def test_api_stats(self):
        s = get_stats_overview()
        self.assertTrue(s.success)
        self.assertGreaterEqual(s.total_violations, 0)

    def test_api_violations(self):
        v = get_violations(page=1, page_size=5)
        self.assertTrue(v.success)
        self.assertIsInstance(v.data, list)

    def test_api_client_fetch(self):
        client = APIClient()
        df, source, err = client.fetch_violations(page_size=10)
        self.assertIsNone(err)
        self.assertFalse(df.empty)


if __name__ == "__main__":
    unittest.main()
