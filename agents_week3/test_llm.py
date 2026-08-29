from agents_week3.llm import llm


response = llm.invoke(
    "Explain OmniBrain in one sentence."
)

print("\nLLM RESPONSE:")
print(response.content)