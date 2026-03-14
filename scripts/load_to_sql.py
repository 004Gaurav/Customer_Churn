import argparse
from pathlib import Path
import sqlite3
import pandas as pd


def load_csv_to_sqlite(csv_path: Path, db_path: Path, table_name: str) -> None:
    df = pd.read_csv(csv_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists='replace', index=False)

    print(f"Loaded {len(df)} rows into '{table_name}' at {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load churn CSV into SQLite")
    parser.add_argument("--csv", default="data/churn.csv", help="Path to CSV file")
    parser.add_argument("--db", default="sql/churn.db", help="Path to SQLite DB file")
    parser.add_argument("--table", default="customer_churn", help="Target table name")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    db_path = Path(args.db)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    load_csv_to_sqlite(csv_path, db_path, args.table)


if __name__ == "__main__":
    main()
