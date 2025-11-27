#!/usr/bin/env python3
"""Lightweight (no DB) test documenting booking identity precedence logic.
Run manually: python test_identity_precedence.py
This does NOT hit the database; it just simulates the branch ordering we enforce.
"""

from pprint import pprint

TESTS = [
    {
        "label": "Explicit person_id wins",
        "inputs": {
            "person_id": 42,
            "client_name": "Fluffy",
            "client_email": None,
            "selector": "fluffy",
        },
        "expected_notes": "Keeps provided name; fills email later from person if missing",
    },
    {
        "label": "Single-person account before core resolver",
        "inputs": {
            "person_id": None,
            "client_name": None,
            "client_email": None,
            "selector": "householdA",
        },
        "expected_notes": "We attach that sole person instead of creating a shadow",
    },
    {
        "label": "Email direct match before shadow creation",
        "inputs": {
            "person_id": None,
            "client_name": None,
            "client_email": "kid@example.com",
            "selector": "kid@example.com",
        },
        "expected_notes": "Existing person via email selected",
    },
    {
        "label": "Fallback to core resolver (shadow) only when no earlier path matched",
        "inputs": {
            "person_id": None,
            "client_name": "New Name",
            "client_email": None,
            "selector": "New Name",
        },
        "expected_notes": "Core resolver may create shadow person",
    },
]

if __name__ == "__main__":
    print("=== Identity Precedence (Documentation Harness) ===")
    for t in TESTS:
        print(f"\nCase: {t['label']}")
        pprint(t["inputs"])  # This is illustrative; actual DB logic enriches identity.
        print("Note:", t["expected_notes"])
    print("\n(If precedence changes, update this file to reflect new intended rules.)")
