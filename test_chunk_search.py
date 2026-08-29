from agents_week3.search_agent import search_agent


query = "What is the Week 2 development plan?"


state = {
    "query": query,
    "attempt": 1
}


result = search_agent(state)


print("\n==============================")
print("RETRIEVED CHUNKS")
print("==============================")

for doc in result["retrieved_docs"]:

    print(
        f"\nChunk ID: {doc['chunk_id']}"
    )

    print(
        f"Page: {doc['page']}"
    )

    print(
        f"Score: {doc['score']}"
    )

    print(
        f"Text: {doc['text']}"
    )