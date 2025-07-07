import sqlite3
def get_random_row(cursor, table_name):
    """Original helper function to get a random row from a table."""
    try:
        # First, check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """, (table_name,))
        if not cursor.fetchone():
            print(f"Table '{table_name}' does not exist.")
            return None

        # Get total row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        if row_count == 0:
            print(f"Table '{table_name}' is empty.")
            return None

        # Get random row using RANDOM()
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1")
        random_row = cursor.fetchone()

        # Get column names for better output
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in cursor.fetchall()]

        return {
            'row': random_row,
            'columns': columns,
            'row_dict': dict(zip(columns, random_row)) if random_row else None
        }

    except Exception as e:
        print(f"Error selecting random row: {e}")
        return None

def get_all_string_values(cursor, table_name):
    """Helper function to get all possible string values from VARCHAR columns (excluding VARCHAR(255))."""
    try:
        # First, check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """, (table_name,))
        if not cursor.fetchone():
            print(f"Table '{table_name}' does not exist.")
            return None

        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()

        # Look for VARCHAR columns, but exclude VARCHAR(255)
        varchar_columns = []

        for column in columns_info:
            column_name = column[1]
            column_type = column[2].strip() if column[2] else ''
            column_type_upper = column_type.upper()

            # Check if it's a VARCHAR type
            if column_type_upper.startswith('VARCHAR'):
                # Accept plain VARCHAR
                if column_type_upper == 'VARCHAR':
                    varchar_columns.append(column_name)
                # Accept VARCHAR(n) where n != 255
                elif '(' in column_type_upper and ')' in column_type_upper:
                    # Extract the number from VARCHAR(n)
                    try:
                        start = column_type_upper.index('(') + 1
                        end = column_type_upper.index(')')
                        size = column_type_upper[start:end].strip()

                        # Include if the size is not 255
                        if size != '255':
                            varchar_columns.append(column_name)
                    except (ValueError, IndexError):
                        # If we can't parse it properly, skip it
                        pass

        if not varchar_columns:
            print(f"No suitable VARCHAR columns found in table '{table_name}'.")
            return None

        # Get all unique values for each VARCHAR column
        result = {}
        for column in varchar_columns:
            try:
                # Use quote to handle column names with special characters
                cursor.execute(f"""
                    SELECT DISTINCT "{column}"
                    FROM {table_name}
                    WHERE "{column}" IS NOT NULL
                    AND "{column}" != ''
                    ORDER BY "{column}"
                """)
                values = [row[0] for row in cursor.fetchall() if isinstance(row[0], str)]
                result[column] = values
            except Exception as e:
                print(f"Warning: Could not retrieve values for column '{column}': {e}")
                result[column] = []

        return {
            'table_name': table_name,
            'varchar_columns': varchar_columns,
            'all_string_values': result,
            'total_columns': len(varchar_columns),
            'total_unique_values': sum(len(vals) for vals in result.values())
        }

    except Exception as e:
        print(f"Error retrieving string values: {e}")
        return None


def get_random_row_with_all_strings(cursor, table_name):
    """Modified version that gets a random row AND all possible values from VARCHAR columns."""
    try:
        # First, check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """, (table_name,))
        if not cursor.fetchone():
            print(f"Table '{table_name}' does not exist.")
            return None

        # Get total row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        if row_count == 0:
            print(f"Table '{table_name}' is empty.")
            return None

        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        columns = [column[1] for column in columns_info]

        # Get random row
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1")
        random_row = cursor.fetchone()

        # Get all values from VARCHAR columns only
        varchar_values_info = get_all_string_values(cursor, table_name)

        return {
            'random_row': random_row,
            'columns': columns,
            'row_dict': dict(zip(columns, random_row)) if random_row else None,
            'all_string_values': varchar_values_info['all_string_values'] if varchar_values_info else {},
            'varchar_columns': varchar_values_info['varchar_columns'] if varchar_values_info else [],
            'total_unique_values': varchar_values_info['total_unique_values'] if varchar_values_info else 0
        }

    except Exception as e:
        print(f"Error selecting random row with string values: {e}")
        return None


def database_value(table_name,
                  file_path: str = "spider_data/database/department_store/department_store.sqlite",
                  get_all_strings: bool = True):
    """
    Access database and get random row data with optional string value collection.
    Only collects values from columns with exactly VARCHAR type (not VARCHAR(255) etc).

    Args:
        table_name: Name of the table to query
        file_path: Path to the SQLite database file
        get_all_strings: If True, also collect all possible values from VARCHAR columns

    Returns:
        Dictionary containing row data and optionally all string values
    """
    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()

        if get_all_strings:
            result = get_random_row_with_all_strings(cursor, table_name)
        else:
            result = get_random_row(cursor, table_name)

        conn.close()
        return result

    except Exception as e:
        print(f"Error accessing database: {e}")
        return None