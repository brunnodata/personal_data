"""
guardrails — hallucination prevention layer.

Modules:
    writer   — citation-binding writer prompt (Stop 3)
    verifier — claim decomposition, entailment check, and decision gate (Stop 3)

WHY GUARDRAILS?
    A generative LLM can produce fluent, confident-sounding answers that are not
    supported by the retrieved context.  In a customer-facing system like
    TechStore Plus, a hallucinated warranty term or incorrect return window
    creates legal and reputational risk.

    The two-stage guardrail pipeline here addresses this:
    1. writer.py enforces citation binding: every sentence must reference a
       source key, making unsupported claims visible.
    2. verifier.py decomposes the answer into atomic claims and checks each one
       against the cited source chunk via an entailment prompt.  The decision
       gate then chooses between returning the answer, adding a disclaimer,
       falling back to extractive mode, or refusing to answer.

    WHY IS THE THRESHOLD 0.85?
        The 0.85 support rate is a practical balance between helpfulness and
        trust.  A threshold of 1.0 would refuse any answer with a single
        unverified claim; 0.0 would disable the gate entirely.  0.85 allows
        minor contextual inferences while blocking substantial hallucinations.
        This is a tunable constant (SUPPORT_RATE_THRESHOLD in verifier.py) —
        adjust it based on the risk tolerance of the deployment context.
"""
