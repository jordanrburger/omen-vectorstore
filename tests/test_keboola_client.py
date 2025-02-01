import unittest
from unittest.mock import patch, MagicMock
from app.keboola_client import KeboolaClient


class TestKeboolaClient(unittest.TestCase):
    
    @patch('app.keboola_client.Client')
    def test_list_buckets_success(self, MockClient):
        # Setup mock for the client
        mock_instance = MagicMock()
        mock_instance.buckets.list.return_value = [{'id': 'bucket1'}, {'id': 'bucket2'}]
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        buckets = client.list_buckets()
        self.assertEqual(len(buckets), 2)
        mock_instance.buckets.list.assert_called_once()

    @patch('app.keboola_client.Client')
    def test_list_buckets_failure(self, MockClient):
        # Setup mock to raise exception
        mock_instance = MagicMock()
        mock_instance.buckets.list.side_effect = Exception("Error fetching buckets")
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        buckets = client.list_buckets()
        self.assertIsNone(buckets)
        mock_instance.buckets.list.assert_called_once()

    @patch('app.keboola_client.Client')
    def test_list_tables_success(self, MockClient):
        # Setup mock for the client
        mock_instance = MagicMock()
        mock_instance.buckets.list_tables.return_value = [{'id': 'table1'}, {'id': 'table2'}]
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        tables = client.list_tables("bucket1")
        self.assertEqual(len(tables), 2)
        mock_instance.buckets.list_tables.assert_called_once_with("bucket1")

    @patch('app.keboola_client.Client')
    def test_list_tables_failure(self, MockClient):
        # Setup mock to raise exception
        mock_instance = MagicMock()
        mock_instance.buckets.list_tables.side_effect = Exception("Error fetching tables")
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        tables = client.list_tables("bucket1")
        self.assertIsNone(tables)
        mock_instance.buckets.list_tables.assert_called_once_with("bucket1")

    @patch('app.keboola_client.Client')
    def test_get_table_details_success(self, MockClient):
        # Setup mock for the client
        mock_instance = MagicMock()
        mock_instance.tables.detail.return_value = {'id': 'table1', 'name': 'Test Table'}
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        details = client.get_table_details("table1")
        self.assertEqual(details['id'], 'table1')
        mock_instance.tables.detail.assert_called_once_with("table1")

    @patch('app.keboola_client.Client')
    def test_get_table_details_failure(self, MockClient):
        # Setup mock to raise exception
        mock_instance = MagicMock()
        mock_instance.tables.detail.side_effect = Exception("Error fetching table details")
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        details = client.get_table_details("table1")
        self.assertIsNone(details)
        mock_instance.tables.detail.assert_called_once_with("table1")


if __name__ == '__main__':
    unittest.main()
