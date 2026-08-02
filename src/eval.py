import os # os module import kar rahe hain takki operating system ke functions use kar sakein (jaise file paths)
import json 
from main import call_llm, process_chat, get_db_connection 

def evaluate_response(query: str, response: str, context: str) -> dict:
    """Uses LLM-as-a-judge to evaluate response quality."""
    # Yeh function LLM ko as a judge use karta hai chatbot ke answer ko evaluate karne ke liye
    prompt = f"""
    Evaluate the following chatbot response out of 10 based on two metrics:
    1. Correctness: Does it directly and accurately answer the user's query?
    2. Context Relevance: Does it properly use the retrieved context?

    Query: {query}
    Context Available: {context}
    Bot Response: {response}

    Return pure JSON format: {{"correctness": 8, "context_relevance": 9, "feedback": "good answer"}}
    """
    
    try:
        # LLM ko prompt bhej rahe hain aur sirf JSON output maang rahe hain
        eval_result = call_llm(prompt, "Return ONLY valid JSON.")
        
        # String se JSON format extract karne ke liye curly braces {} dhoond rahe hain
        start = eval_result.find("{")
        end = eval_result.rfind("}") + 1
        
        # Agar braces mil gaye toh JSON load kar do
        if start != -1 and end != -1:
           
            return json.loads(eval_result[start:end]) # string ko dictionary mein convert karke return kar rahe hain
    except Exception as e:
        # Agar koi error aata hai toh console pe print karenge
        print(f"Eval failed: {e}")
        
    # Agar try block fail ho gaya, toh yeh default fail output dega
    return {"correctness": 0, "context_relevance": 0, "feedback": "Failed to parse eval"}

def run_evaluation():
    # Yeh function humare evaluation process ko start karta hai
    print("--- Starting Beginner Evaluation Framework ---")
    
    # Test karne ke liye kuch queries banayi hain
    test_queries = [
        "What is machine learning?",
        "What are the main goals of Artificial Intelligence?",
        "Tell me a joke."
    ]
    
    # Har test query pe loop chala rahe hain
    for i, query in enumerate(test_queries):
        print(f"\nTest {i+1}: '{query}'")
        
        # query ko process_chat function ko bhej rahe hain (process_chat RAG ya web decide karke answer dega)
        result = process_chat(query, "eval_user", "eval_session")
        response = result["response"] # Chatbot ka output response
        source = result["source_used"] # Kis source se answer aaya (RAG, Web, ya Direct)
        
        context = ""
        # Agar source Direct nahi hai (matlab RAG ya Web use hua hai)
        if source != "Direct":
            from main import retrieve_rag, web_search, get_graph_context
            
            # Agar Web search use hua tha toh web se context nikalenge
            if "Web Search" in source:
                context = web_search(query)
            # Warna RAG vector search aur knowledge graph se context nikalenge
            else:
                context = retrieve_rag(query) + "\n" + get_graph_context(query)
                
        # Evaluate function call karke marks (correctness, relevance) nikal rahe hain
        eval_scores = evaluate_response(query, response, context)
        
        # Scores aur feedback ko print karwa rahe hain takki result dekh sakein
        print(f"Source Used: {source}")
        print(f"Correctness: {eval_scores.get('correctness')}/10")
        print(f"Context Relevance: {eval_scores.get('context_relevance')}/10")
        print(f"Feedback: {eval_scores.get('feedback')}")
        print("-" * 40) # Ek divider line print kar rahe hain
        
if __name__ == "__main__":
    # Agar yeh script directly run hoti hai, toh run_evaluation() execute hoga
    run_evaluation()
