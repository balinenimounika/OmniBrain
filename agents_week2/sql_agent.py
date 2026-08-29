import sqlite3


DB_PATH = "data/database/omni.db"


def sql_agent(state):

    query = state["query"]

    print("\n[SQL AGENT]")
    print("Handling structured/database query...")
    print("User query:", query)

    query_lower = query.lower()

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    try:

        # -----------------------------------------
        # Stock price query
        # -----------------------------------------

        if "stock price" in query_lower:

            for company in ["apple", "microsoft", "nvidia", "google"]:

                if company in query_lower:

                    cursor.execute("""
                    SELECT company_name, stock_symbol, stock_price
                    FROM companies
                    WHERE LOWER(company_name) = ?
                    """, (company,))

                    result = cursor.fetchone()

                    if result:

                        company_name, symbol, price = result

                        response = (
                            f"{company_name} ({symbol}) "
                            f"stock price: ${price}"
                        )

                        print("SQL Result:", response)

                        return {
                            "response": response
                        }

        # -----------------------------------------
        # Revenue query
        # -----------------------------------------

        if "revenue" in query_lower and "2024" in query_lower:

            for company in ["apple", "microsoft", "nvidia", "google"]:

                if company in query_lower:

                    cursor.execute("""
                    SELECT company_name, revenue_2024
                    FROM companies
                    WHERE LOWER(company_name) = ?
                    """, (company,))

                    result = cursor.fetchone()

                    if result:

                        company_name, revenue = result

                        response = (
                            f"{company_name} revenue in 2024: "
                            f"${revenue} million"
                        )

                        print("SQL Result:", response)

                        return {
                            "response": response
                        }

        return {
            "response": (
                "I could not find matching structured data "
                "in the database."
            )
        }

    finally:

        connection.close()