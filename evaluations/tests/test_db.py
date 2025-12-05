# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for database utilities."""

import pytest
from pathlib import Path

from evaluations.core.db import (
    DBConfig,
    OrderRecord,
    TestDatabase,
    CSV_TO_DB_COLUMNS,
)


# Path to the CSV file
CSV_PATH = Path(__file__).parent.parent.parent / "data" / "orders.csv"


class TestDBConfig:
    """Tests for DBConfig model."""

    def test_default_config(self):
        """Default config uses environment variables or defaults."""
        config = DBConfig()
        assert config.host is not None
        assert config.port == 5432 or isinstance(config.port, int)
        assert config.dbname is not None

    def test_connection_params(self):
        """connection_params returns correct dict."""
        config = DBConfig(
            host="testhost",
            port=5433,
            user="testuser",
            password="testpass",
            dbname="testdb",
            test_dbname="testdb_test",
        )

        params = config.connection_params(use_test_db=False)
        assert params["host"] == "testhost"
        assert params["dbname"] == "testdb"

        test_params = config.connection_params(use_test_db=True)
        assert test_params["dbname"] == "testdb_test"


class TestOrderRecord:
    """Tests for OrderRecord model."""

    def test_from_db_row_full(self):
        """Create OrderRecord from a complete database row."""
        row = {
            "customer_id": 4165,
            "order_id": 52768,
            "product_name": "JETSON NANO DEVELOPER KIT",
            "product_description": "A small powerful computer...",
            "order_date": "2024-10-05",
            "quantity": 2,
            "order_amount": 298.0,
            "order_status": "Delivered",
            "return_status": None,
            "return_start_date": None,
            "return_received_date": None,
            "return_completed_date": None,
            "return_reason": None,
            "notes": None,
        }
        record = OrderRecord.from_db_row(row)

        assert record.customer_id == 4165
        assert record.order_id == 52768
        assert record.product_name == "JETSON NANO DEVELOPER KIT"
        assert record.order_status == "Delivered"
        assert record.return_status is None

    def test_from_db_row_with_return(self):
        """Create OrderRecord from a row with return data."""
        row = {
            "customer_id": 4165,
            "order_id": 4065,
            "product_name": "NVIDIA® GEFORCE RTX™ 4090",
            "product_description": "The ultimate GeForce GPU",
            "order_date": "2024-10-10",
            "quantity": 1,
            "order_amount": 1599.0,
            "order_status": "Return Requested",
            "return_status": "Requested",
            "return_start_date": "2024-10-12",
            "return_received_date": None,
            "return_completed_date": None,
            "return_reason": "Received a faulty unit",
            "notes": None,
        }
        record = OrderRecord.from_db_row(row)

        assert record.order_status == "Return Requested"
        assert record.return_status == "Requested"
        assert record.return_reason == "Received a faulty unit"

    def test_from_db_row_handles_missing_fields(self):
        """OrderRecord handles missing optional fields gracefully."""
        row = {
            "customer_id": 1234,
            "order_id": 5678,
            "product_name": "Test Product",
        }
        record = OrderRecord.from_db_row(row)

        assert record.customer_id == 1234
        assert record.product_description == ""
        assert record.quantity == 0
        assert record.return_status is None


class TestCSVColumnMapping:
    """Tests for CSV to DB column mapping."""

    def test_all_expected_columns_mapped(self):
        """Verify all expected CSV columns are mapped."""
        expected_csv_cols = {
            "CID", "OrderID", "product_name", "product_description",
            "OrderDate", "Quantity", "OrderAmount", "OrderStatus",
            "ReturnStatus", "ReturnStartDate", "ReturnReceivedDate",
            "ReturnCompletedDate", "ReturnReason", "Notes",
        }
        assert set(CSV_TO_DB_COLUMNS.keys()) == expected_csv_cols

    def test_column_mappings_are_lowercase(self):
        """DB column names should be lowercase."""
        for db_col in CSV_TO_DB_COLUMNS.values():
            assert db_col == db_col.lower()


class TestTestDatabase:
    """Tests for TestDatabase class (CSV operations only, no DB connection)."""

    def test_init_with_default_config(self):
        """TestDatabase initializes with default config."""
        db = TestDatabase()
        assert db.config is not None
        assert db._baseline_data is None

    def test_init_with_custom_config(self):
        """TestDatabase accepts custom config."""
        config = DBConfig(host="customhost", port=5433)
        db = TestDatabase(config=config)
        assert db.config.host == "customhost"
        assert db.config.port == 5433

    @pytest.mark.skipif(not CSV_PATH.exists(), reason="CSV file not found")
    def test_load_baseline_from_csv(self):
        """Load baseline data from CSV file."""
        db = TestDatabase()
        db.load_baseline_from_csv(CSV_PATH)

        assert db._baseline_data is not None
        assert len(db._baseline_data) > 0

    @pytest.mark.skipif(not CSV_PATH.exists(), reason="CSV file not found")
    def test_baseline_data_has_correct_columns(self):
        """Loaded baseline data has correct column names."""
        db = TestDatabase()
        db.load_baseline_from_csv(CSV_PATH)

        first_record = db._baseline_data[0]
        expected_cols = set(CSV_TO_DB_COLUMNS.values())
        actual_cols = set(first_record.keys())
        assert actual_cols == expected_cols

    @pytest.mark.skipif(not CSV_PATH.exists(), reason="CSV file not found")
    def test_baseline_data_for_user_4165(self):
        """Verify we can find test user 4165's data."""
        db = TestDatabase()
        db.load_baseline_from_csv(CSV_PATH)

        user_records = [r for r in db._baseline_data if r.get("customer_id") == "4165"]
        assert len(user_records) >= 1

        # Find the Jetson Nano order
        jetson_orders = [r for r in user_records if "JETSON" in (r.get("product_name") or "").upper()]
        assert len(jetson_orders) >= 1
        assert jetson_orders[0]["order_status"] == "Delivered"

    def test_load_csv_file_not_found(self):
        """Raise error when CSV file doesn't exist."""
        db = TestDatabase()
        with pytest.raises(FileNotFoundError):
            db.load_baseline_from_csv("/nonexistent/path/orders.csv")

    def test_reset_without_baseline_raises_error(self):
        """reset_to_baseline raises error if baseline not loaded."""
        db = TestDatabase()
        with pytest.raises(RuntimeError, match="Baseline data not loaded"):
            db.reset_to_baseline()


class TestTestDatabaseIntegration:
    """
    Integration tests that require a live database connection.

    These tests are skipped by default. To run them:
    1. Ensure PostgreSQL is running
    2. Set appropriate environment variables
    3. Run with: pytest -m integration
    """

    @pytest.mark.integration
    @pytest.mark.skipif(not CSV_PATH.exists(), reason="CSV file not found")
    def test_full_setup_and_query(self):
        """Full integration test: setup, reset, and query."""
        db = TestDatabase()

        try:
            db.setup(CSV_PATH)

            # Query for user 4165
            orders = db.get_orders_by_customer("4165")
            assert len(orders) > 0

            # Find specific order
            order = db.get_order("4165", 52768)
            assert order is not None
            assert order.product_name == "JETSON NANO DEVELOPER KIT"

        finally:
            # Clean up
            db.drop_test_database()

    @pytest.mark.integration
    def test_verify_return_status(self):
        """Test return status verification."""
        db = TestDatabase()

        try:
            db.setup(CSV_PATH)

            # Verify a known return status
            result = db.verify_return_status("4165", 4065, "Requested")
            assert result["passed"] is True

            # Verify wrong status fails
            result = db.verify_return_status("4165", 4065, "Approved")
            assert result["passed"] is False

        finally:
            db.drop_test_database()
