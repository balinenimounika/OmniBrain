import sys
import os

# Add root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.guardrails.guardrails import should_answer_question

def test():
    print("Testing In-Scope")
    state = {
        "user_query": "What is the company's revenue?",
        "retrieved_context": "The company's revenue increased by 25 percent in 2025 due to strong market growth.",
        "similarity_scores": [0.8],
        "retrieval_results": [{"score": 0.8}],
        "retrieval_status": "success"
    }
    print(should_answer_question(state))

    print("\nTesting Out-of-Scope (General)")
    state2 = {
        "user_query": "What is the capital of France?",
        "retrieved_context": "The company's revenue increased by 25 percent in 2025 due to strong market growth.",
        "similarity_scores": [0.3],
        "retrieval_results": [{"score": 0.3}],
        "retrieval_status": "success"
    }
    print(should_answer_question(state2))

    print("\nTesting Insufficient Context")
    state3 = {
        "user_query": "Who is the CEO?",
        "retrieved_context": "The company's revenue increased by 25 percent in 2025 due to strong market growth.",
        "similarity_scores": [0.5],
        "retrieval_results": [{"score": 0.5}],
        "retrieval_status": "success"
    }
    print(should_answer_question(state3))

if __name__ == "__main__":
    test()
