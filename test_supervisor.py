from graph.workflow import graph


def test_query(query):

    print("\n" + "=" * 50)
    print("QUERY:", query)
    print("=" * 50)

    result = graph.invoke({
        "query": query
    })

    print("RESULT:")
    print(result)


if __name__ == "__main__":

    test_query(
        "What does the document say about revenue?"
    )

    test_query(
        "What was the stock price in 2024?"
    )

    test_query(
        "What does this bar chart show?"
    )