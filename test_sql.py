from agents_week2.sql_agent import sql_agent


queries = [
    "What is Apple's stock price?",
    "What is Microsoft's stock price?",
    "What was Apple's revenue in 2024?",
    "What is NVIDIA's stock price?"
]


for query in queries:

    print("\n" + "=" * 60)
    print("QUERY:", query)
    print("=" * 60)

    result = sql_agent({
        "query": query
    })

    print("FINAL RESPONSE:")
    print(result["response"])