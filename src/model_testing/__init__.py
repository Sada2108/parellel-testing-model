"""Parallel model testing module.

Lets someone register any model by pasting an API key (no code changes),
then run the same datasheet chunk through every registered model at once
and compare token usage, cost, latency and a quality score for the
enhancement/summary call site (``src/ingestion/enrich.py``).
"""
