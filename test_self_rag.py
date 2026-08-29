from graph_week3.self_rag_workflow import graph


query = "What is the company's revenue in 2024?"


print("=" * 60)
print("SELF-RAG TEST")
print("=" * 60)

result = graph.invoke({
    "query": query,
    "attempt": 1
})


print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(result)