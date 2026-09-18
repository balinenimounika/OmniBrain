import sqlite3
import os


# Create database directory
os.makedirs("data/database", exist_ok=True)

# Database path
db_path = "data/database/omni.db"

# Connect to SQLite
connection = sqlite3.connect(db_path)

cursor = connection.cursor()


# Create companies table
cursor.execute("""
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT NOT NULL,
    stock_symbol TEXT NOT NULL,
    stock_price REAL,
    revenue_2024 REAL
)
""")


# Insert sample data
companies = [
    ("Apple", "AAPL", 227.16, 391035.0),
    ("Microsoft", "MSFT", 506.69, 245122.0),
    ("NVIDIA", "NVDA", 181.57, 130497.0),
    ("Google", "GOOGL", 202.09, 350018.0)
]


cursor.executemany("""
INSERT INTO companies
(company_name, stock_symbol, stock_price, revenue_2024)
VALUES (?, ?, ?, ?)
""", companies)


connection.commit()
connection.close()


print("Database created successfully!")
print(f"Location: {db_path}")