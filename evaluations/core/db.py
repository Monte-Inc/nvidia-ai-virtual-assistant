# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test database management for evaluations.

Provides utilities for cloning the production database schema,
resetting data between tests, and querying/verifying database state.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Column mappings from CSV headers to database columns
CSV_TO_DB_COLUMNS = {
    "CID": "customer_id",
    "OrderID": "order_id",
    "product_name": "product_name",
    "product_description": "product_description",
    "OrderDate": "order_date",
    "Quantity": "quantity",
    "OrderAmount": "order_amount",
    "OrderStatus": "order_status",
    "ReturnStatus": "return_status",
    "ReturnStartDate": "return_start_date",
    "ReturnReceivedDate": "return_received_date",
    "ReturnCompletedDate": "return_completed_date",
    "ReturnReason": "return_reason",
    "Notes": "notes",
}


class DBConfig(BaseModel):
    """Database connection configuration."""

    host: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    password: str = Field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "password")
    )
    dbname: str = Field(
        default_factory=lambda: os.getenv("CUSTOMER_DATA_DB", "customer_data")
    )
    test_dbname: str = Field(
        default_factory=lambda: os.getenv("TEST_CUSTOMER_DATA_DB", "customer_data_test")
    )

    def connection_params(self, use_test_db: bool = False) -> dict:
        """Get connection parameters as a dict."""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "dbname": self.test_dbname if use_test_db else self.dbname,
        }


class OrderRecord(BaseModel):
    """Represents an order/return record from the customer_data table."""

    customer_id: int
    order_id: int
    product_name: str
    product_description: str = ""
    order_date: str = ""
    quantity: int = 0
    order_amount: float = 0.0
    order_status: str = ""
    return_status: str | None = None
    return_start_date: str | None = None
    return_received_date: str | None = None
    return_completed_date: str | None = None
    return_reason: str | None = None
    notes: str | None = None

    @classmethod
    def from_db_row(cls, row: dict) -> "OrderRecord":
        """Create an OrderRecord from a database row dict."""
        return cls(
            customer_id=row["customer_id"],
            order_id=row["order_id"],
            product_name=row["product_name"],
            product_description=row.get("product_description") or "",
            order_date=str(row["order_date"]) if row.get("order_date") else "",
            quantity=row.get("quantity") or 0,
            order_amount=float(row["order_amount"]) if row.get("order_amount") else 0.0,
            order_status=row.get("order_status") or "",
            return_status=row.get("return_status"),
            return_start_date=(
                str(row["return_start_date"]) if row.get("return_start_date") else None
            ),
            return_received_date=(
                str(row["return_received_date"])
                if row.get("return_received_date")
                else None
            ),
            return_completed_date=(
                str(row["return_completed_date"])
                if row.get("return_completed_date")
                else None
            ),
            return_reason=row.get("return_reason"),
            notes=row.get("notes"),
        )


class TestDatabase:
    """
    Manages a test database for evaluation isolation.

    Supports:
    - Creating/cloning the test database schema
    - Loading baseline data from CSV
    - Resetting to baseline state between tests
    - Querying and verifying database state
    """

    # Schema for the customer_data table
    CUSTOMER_DATA_SCHEMA = """
    CREATE TABLE IF NOT EXISTS customer_data (
        customer_id INTEGER NOT NULL,
        order_id INTEGER NOT NULL,
        product_name VARCHAR(255) NOT NULL,
        product_description TEXT,
        order_date TIMESTAMP,
        quantity INTEGER,
        order_amount DECIMAL(10, 2),
        order_status VARCHAR(50),
        return_status VARCHAR(50),
        return_start_date TIMESTAMP,
        return_received_date TIMESTAMP,
        return_completed_date TIMESTAMP,
        return_reason TEXT,
        notes TEXT,
        PRIMARY KEY (customer_id, order_id)
    );
    """

    def __init__(self, config: DBConfig | None = None):
        """Initialize with database configuration."""
        self.config = config or DBConfig()
        self._baseline_data: list[dict] | None = None

    def _get_connection(self, use_test_db: bool = True):
        """Get a database connection."""
        return psycopg2.connect(**self.config.connection_params(use_test_db))

    def _get_admin_connection(self):
        """Get connection to default 'postgres' database for admin operations."""
        params = self.config.connection_params(use_test_db=False)
        params["dbname"] = "postgres"
        return psycopg2.connect(**params)

    def create_test_database(self) -> None:
        """Create the test database if it doesn't exist."""
        conn = self._get_admin_connection()
        conn.autocommit = True

        try:
            with conn.cursor() as cur:
                # Check if database exists
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (self.config.test_dbname,),
                )
                exists = cur.fetchone()

                if not exists:
                    logger.info(f"Creating test database: {self.config.test_dbname}")
                    cur.execute(f'CREATE DATABASE "{self.config.test_dbname}"')
                else:
                    logger.info(
                        f"Test database already exists: {self.config.test_dbname}"
                    )
        finally:
            conn.close()

    def drop_test_database(self) -> None:
        """Drop the test database."""
        conn = self._get_admin_connection()
        conn.autocommit = True

        try:
            with conn.cursor() as cur:
                # Terminate existing connections
                cur.execute(
                    """
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = %s
                    AND pid <> pg_backend_pid()
                    """,
                    (self.config.test_dbname,),
                )
                cur.execute(f'DROP DATABASE IF EXISTS "{self.config.test_dbname}"')
                logger.info(f"Dropped test database: {self.config.test_dbname}")
        finally:
            conn.close()

    def create_schema(self) -> None:
        """Create the customer_data table schema in the test database."""
        conn = self._get_connection(use_test_db=True)
        try:
            with conn.cursor() as cur:
                cur.execute(self.CUSTOMER_DATA_SCHEMA)
            conn.commit()
            logger.info("Created customer_data table schema")
        finally:
            conn.close()

    def load_baseline_from_csv(self, csv_path: str | Path) -> None:
        """
        Load baseline data from a CSV file.

        Args:
            csv_path: Path to the orders.csv file
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self._baseline_data = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Map CSV columns to database columns
                db_row = {}
                for csv_col, db_col in CSV_TO_DB_COLUMNS.items():
                    value = row.get(csv_col, "").strip()
                    # Handle empty values
                    if value == "" or value == '""':
                        value = None
                    db_row[db_col] = value
                self._baseline_data.append(db_row)

        logger.info(f"Loaded {len(self._baseline_data)} records from {csv_path}")

    def reset_to_baseline(self) -> None:
        """
        Reset the test database to baseline state.

        Truncates the customer_data table and reloads from baseline data.
        """
        if self._baseline_data is None:
            raise RuntimeError(
                "Baseline data not loaded. Call load_baseline_from_csv first."
            )

        conn = self._get_connection(use_test_db=True)
        try:
            with conn.cursor() as cur:
                # Truncate and reset
                cur.execute("TRUNCATE TABLE customer_data")

                # Insert baseline data
                columns = list(CSV_TO_DB_COLUMNS.values())
                placeholders = ", ".join(["%s"] * len(columns))
                insert_sql = f"""
                    INSERT INTO customer_data ({", ".join(columns)})
                    VALUES ({placeholders})
                """

                for row in self._baseline_data:
                    values = [row.get(col) for col in columns]
                    cur.execute(insert_sql, values)

            conn.commit()
            logger.info(
                f"Reset test database with {len(self._baseline_data)} records"
            )
        finally:
            conn.close()

    def setup(self, csv_path: str | Path | None = None) -> None:
        """
        Full setup: create database, schema, and load baseline data.

        Args:
            csv_path: Path to orders.csv. If None, uses default location.
        """
        if csv_path is None:
            # Default to the data directory relative to the project
            csv_path = Path(__file__).parent.parent.parent / "data" / "orders.csv"

        self.create_test_database()
        self.create_schema()
        self.load_baseline_from_csv(csv_path)
        self.reset_to_baseline()

    def get_order(
        self, customer_id: int | str, order_id: int | str
    ) -> OrderRecord | None:
        """
        Get a specific order by customer_id and order_id.

        Args:
            customer_id: The customer ID
            order_id: The order ID

        Returns:
            OrderRecord if found, None otherwise
        """
        conn = self._get_connection(use_test_db=True)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM customer_data
                    WHERE customer_id = %s AND order_id = %s
                    """,
                    (customer_id, order_id),
                )
                row = cur.fetchone()
                return OrderRecord.from_db_row(dict(row)) if row else None
        finally:
            conn.close()

    def get_orders_by_customer(self, customer_id: int | str) -> list[OrderRecord]:
        """
        Get all orders for a customer.

        Args:
            customer_id: The customer ID

        Returns:
            List of OrderRecords
        """
        conn = self._get_connection(use_test_db=True)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM customer_data
                    WHERE customer_id = %s
                    ORDER BY order_date DESC
                    """,
                    (customer_id,),
                )
                rows = cur.fetchall()
                return [OrderRecord.from_db_row(dict(row)) for row in rows]
        finally:
            conn.close()

    def get_orders_by_product(
        self, customer_id: int | str, product_name: str
    ) -> list[OrderRecord]:
        """
        Get orders for a customer matching a product name (case-insensitive partial match).

        Args:
            customer_id: The customer ID
            product_name: Product name to search for

        Returns:
            List of matching OrderRecords
        """
        conn = self._get_connection(use_test_db=True)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM customer_data
                    WHERE customer_id = %s
                    AND LOWER(product_name) LIKE LOWER(%s)
                    ORDER BY order_date DESC
                    """,
                    (customer_id, f"%{product_name}%"),
                )
                rows = cur.fetchall()
                return [OrderRecord.from_db_row(dict(row)) for row in rows]
        finally:
            conn.close()

    def verify_return_status(
        self,
        customer_id: int | str,
        order_id: int | str,
        expected_status: str,
    ) -> dict[str, Any]:
        """
        Verify the return status of an order.

        Args:
            customer_id: The customer ID
            order_id: The order ID
            expected_status: The expected return_status value

        Returns:
            Dict with 'passed', 'expected', 'actual' keys
        """
        order = self.get_order(customer_id, order_id)
        if order is None:
            return {
                "passed": False,
                "expected": expected_status,
                "actual": None,
                "error": f"Order not found: customer_id={customer_id}, order_id={order_id}",
            }

        actual_status = order.return_status or ""
        passed = actual_status.lower() == expected_status.lower()

        return {
            "passed": passed,
            "expected": expected_status,
            "actual": actual_status,
        }

    def get_unique_customer_ids(self) -> list[int]:
        """Get all unique customer IDs in the database."""
        conn = self._get_connection(use_test_db=True)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT customer_id FROM customer_data ORDER BY customer_id")
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    def get_customer_summary(self, customer_id: int | str) -> dict[str, Any]:
        """
        Get a summary of a customer's orders for task generation.

        Returns counts of orders by status, return status, and products.
        """
        orders = self.get_orders_by_customer(customer_id)

        order_statuses = {}
        return_statuses = {}
        products = []

        for order in orders:
            # Count order statuses
            status = order.order_status or "Unknown"
            order_statuses[status] = order_statuses.get(status, 0) + 1

            # Count return statuses
            ret_status = order.return_status or "None"
            return_statuses[ret_status] = return_statuses.get(ret_status, 0) + 1

            # Collect products
            products.append(
                {
                    "product_name": order.product_name,
                    "order_id": order.order_id,
                    "order_status": order.order_status,
                    "return_status": order.return_status,
                }
            )

        return {
            "customer_id": customer_id,
            "total_orders": len(orders),
            "order_statuses": order_statuses,
            "return_statuses": return_statuses,
            "products": products,
        }
