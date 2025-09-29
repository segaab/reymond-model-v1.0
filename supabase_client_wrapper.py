supabase_client_wrapper.py

from supabase import create_client, Client from typing import List, Dict, Any, Optional import logging

--------------------------

Logging

--------------------------

logging.basicConfig(level=logging.INFO) logger = logging.getLogger("supabase_client_wrapper")

--------------------------

White-labeled Supabase Client

--------------------------

class SupabaseClientWrapper: def init(self, url: str, service_role_key: str): """ Initialize the Supabase client with given credentials. """ self.supabase: Client = create_client(url, service_role_key) logger.info("Supabase client initialized.")

def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Insert a list of records into a Supabase table.
    """
    if not data:
        return {"success": False, "message": "No data provided for insert."}

    resp = self.supabase.table(table_name).insert(data).execute()
    if resp.error:
        logger.error("Insert error on table %s: %s", table_name, resp.error)
        return {"success": False, "error": resp.error}

    count = len(resp.data) if resp.data else 0
    logger.info("Inserted %d rows into %s.", count, table_name)
    return {"success": True, "inserted_count": count}

def fetch_records(
    self, table_name: str, filters: Dict[str, Any] = None, order_by: str = None,
    descending: bool = True, limit: int = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch records from a table with optional filters, order, and limit.

    :param table_name: Name of the table
    :param filters: Dict of column-value pairs to filter on
    :param order_by: Column name to order by
    :param descending: True for descending order
    :param limit: Max number of records to return
    :return: List of records or empty list if none
    """
    try:
        query = self.supabase.table(table_name).select("*")
        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)
        if order_by:
            query = query.order(order_by, desc=descending)
        if limit:
            query = query.limit(limit)

        response = query.execute()
        data = response.data
        if isinstance(data, list):
            logger.info("Fetched %d records from %s", len(data), table_name)
            return data
        else:
            logger.warning("Fetch returned unexpected data type: %s", type(data))
            return []
    except Exception as e:
        logger.exception("Error fetching records from %s: %s", table_name, e)
        return None

--------------------------

Example usage

--------------------------

if name == "main": # Use hardcoded values if needed for plug-and-play SUPABASE_URL = "https://dzddytphimhoxeccxqsw.supabase.co" SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR6ZGR5dHBoaW1ob3hlY2N4cXN3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTM2Njc5NCwiZXhwIjoyMDY2OTQyNzk0fQ.ng0ST7-V-cDBD0Jc80_0DFWXylzE-gte2I9MCX7qb0Q"

client = SupabaseClientWrapper(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Example: Insert some test data
insert_resp = client.insert_data("example_table", [{"name": "Alice", "score": 95}])
print("Insert response:", insert_resp)

# Example: Fetch latest 5 records for a filter
records = client.fetch_records("example_table", filters={"name": "Alice"}, order_by="score", descending=True, limit=5)
print("Fetched records:", records)

