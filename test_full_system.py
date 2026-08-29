from graph_week2.workflow import graph


queries = [
    # TEXT → SELF-RAG
    "What is the Week 2 development plan?",

    # SQL
    "What is Apple's stock price?",

    # VISION
    "What information is shown in the chart image?"
]


for query in queries:

    print("\n" + "=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    try:
        result = graph.invoke({
            "query": query
        })

        print("\nFINAL RESPONSE:")
        print(result.get("response"))

    except Exception as e:
        print("\nERROR:")
        print(e)