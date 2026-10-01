"""Prompt classifier and router: token estimate, task type, complexity score, policy.

Each module is pure (config in, result out) so it can be unit-tested on its own;
engine.py wires them together for the API.
"""
