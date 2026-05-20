# Vetto AI Task Calibration Rules

## Target Metrics (The Success Zone)
- **Oracle Solution (`solve.sh`)**: Must ALWAYS pass with a score of 1000 (100% success).
- **Fast Tier Models**: Must fail consistently, achieving a **0% pass rate**.
- **Smart Tier Models**: Must land in the optimal calibration zone, scoring between **30% and 50% pass rate**.

## Enforcement & Compliance (Anti-Cheat Rules)
- Never create artificial blockers or hardcoded string assertions that reject alternative but valid engineering solutions.
- The active live production configuration path must always remain valid, accurate, and discoverable through proper environment tracing.
- Shift the discovery burden away from the instruction text and into runtime validation logs and profile configurations.
