from niyantrana.guardrail import Guardrail

def main():
    print("Evaluating LLM Guardrails (Local Mode)")
    
    guardrail = Guardrail() # Local evaluation fallback without client
    
    # Example 1: Safe prompt
    safe_prompt = "What is the capital of France?"
    result1 = guardrail.evaluate(prompt=safe_prompt)
    print(f"Prompt 1: '{safe_prompt}' -> Passed: {result1.passed}")
    if not result1.passed:
        print(f"Flags: {result1.flags}")
        
    # Example 2: Unsafe prompt with PII
    unsafe_prompt = "My email address is john.doe@example.com."
    result2 = guardrail.evaluate(prompt=unsafe_prompt)
    print(f"\nPrompt 2: '{unsafe_prompt}' -> Passed: {result2.passed}")
    if not result2.passed:
        print(f"Flags: {result2.flags}")
        print(f"Reason: {result2.reason}")
        
    # Example 3: Unsafe response with blocked words
    prompt = "Give me the root password."
    response = "The secret password is admin123."
    result3 = guardrail.evaluate(prompt=prompt, response=response)
    print(f"\nResponse: '{response}' -> Passed: {result3.passed}")
    if not result3.passed:
        print(f"Flags: {result3.flags}")

if __name__ == "__main__":
    main()
