"""Routing instrumentation: record, align, force, and compare MoE routing.

Submodules:

* :mod:`rr.routing.record`  -- capture per-token top-k expert choices (Phase 0).
* :mod:`rr.routing.align`   -- map target-question tokens across prompts (Phase 2).
* :mod:`rr.routing.metrics` -- Jaccard overlap of expert sets (Phase 2).
* :mod:`rr.routing.force`   -- override routing on chosen tokens (Phase 3).
"""
