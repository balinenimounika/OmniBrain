from graph_week2.workflow import graph


queries = [
    "What is the Week 2 development plan?",
    "What is the company's revenue in 2024?",
    "What does the document say about OmniBrain?"
]


for query in queries:

    print("\n" + "=" * 60)
    print("QUERY:", query)
    print("=" * 60)

    result = graph.invoke({
        "query": query
    })

    print("\nFINAL RESPONSE:")
    print(result.get("response"))