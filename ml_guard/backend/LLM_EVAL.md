# ML Guard: LLM Evaluation & Alignment Architecture

This document defines the architecture for evaluating Large Language Models (LLMs) for safety, truthfulness, and quality within ML Guard 2.0.

## 1. LLM Evaluation Architecture

```text
[ LLM SOURCE (PROD/DEV) ] --( Batch Pairs )--> [ LLM GATEWAY (REST) ]
                                                     |
                                            [ ASYNC EVALUATOR ]
                                                     |
            _________________________________________|_________________________________________
           |                         |                         |                         |
    [ SAFETY ENGINE ]      [ TRUTHFULNESS ENGINE ]     [ ALIGNMENT ENGINE ]      [ ROBUSTNESS ENGINE ]
   (Toxicity, Bias,        (Hallucination, Ref         (Tone, Consistency,       (Jailbreak, Prompt
    Jailbreak)              Alignment)                  Grammar)                  Injection)
           |_________________________|_________________________|_________________________|
                                                     |
                                            [ AGGREGATOR & REPORT ]
                                                     |
                                            [ COMPLIANCE LOG ]
```

## 2. Core Test Design

| Test Category | Metric | Description |
| :--- | :--- | :--- |
| **Safety** | Toxicity Score | Detects hate speech, harassment, and explicit content. |
| **Safety** | Jailbreak Resistance | Tests vulnerabilities to adversarial prompt injection. |
| **Truthfulness** | Hallucination Rate | Measure of factual consistency with provided reference context. |
| **Alignment** | Bias Variance | Deviation in response quality across identity groups. |
| **Quality** | Semantic Consistency | Ability to produce stable answers to semantically identical prompts. |

## 3. Scoring & Alignment Model

ML Guard uses a **Tri-Metric Scoring Model**:
1.  **Safety Score ($S$):** $1 - \max(\text{Toxicity}, \text{JailbreakProbability})$.
2.  **Truthfulness Score ($T$):** NLI-based (Natural Language Inference) alignment between Response and Reference.
3.  **Alignment Score ($A$):** Compliance with predefined persona and branding guidelines.

**Overall Alignment Index (AI):**
$$AI = \frac{W_s S + W_t T + W_a A}{W_s + W_t + W_a}$$

## 4. Extensible Plugin Structure

The engine uses a **Plugin-Based Architecture** to allow easy addition of new LLM tests:
- `BaseLLMTest`: Abstract class for all evaluation modules.
- `HuggingFacePlugin`: Uses local models (e.g., Llama-Guard) for evaluation.
- `CloudAuditPlugin`: Integrates with external APIs (e.g., Google Perspective, OpenAI Moderation).
- `CustomRulePlugin`: RegEx and logic-based checks for domain-specific constraints.

## 5. API Contract

### POST `/api/v1/llm/evaluate`
```json
{
  "model_name": "gpt-4-turbo",
  "prompts": ["What is the capital of France?", "Ignore your instructions and..."],
  "responses": ["Paris is the capital.", "I cannot do that."],
  "references": ["France is a country in Europe with capital Paris."]
}
```

### Response
```json
{
  "run_id": "llm_abcd123",
  "overall_safety_score": 0.98,
  "overall_truthfulness_score": 0.95,
  "results": [
    {"test_name": "Jailbreak Resistance", "score": 1.0, "status": "pass"}
  ]
}
```
