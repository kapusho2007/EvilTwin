"""Evil Twin / Rogue AP detection pipeline (Milestone 3, Step 7).

    ingestion -> preprocess (clean + fold-fitted normalizer) -> features -> train -> evaluate

Only `ingestion` knows the raw dataset schema. Every later stage reads the unified names.
"""
