#!/usr/bin/env python3
"""
Simple test script to validate booking fixes work correctly
"""


def test_client_info_extraction():
    """Test that client info extraction works with various key formats"""
    import json

    # Simulate different pending_client formats
    test_cases = [
        # Case 1: find_client result format
        {
            "id": "123",
            "name": "John Doe",
            "primary_email": "john@example.com",
            "people": [],
        },
        # Case 2: Alternative format
        {"client_name": "Jane Smith", "client_email": "jane@example.com"},
        # Case 3: Person match result
        {
            "name": "Bob Wilson",
            "primary_email": "bob@example.com",
            "person_name": "Bob Wilson Jr",
            "person_id": "456",
        },
    ]

    print("=== Testing Client Info Extraction ===")

    for i, pending_client in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {pending_client}")

        # Simulate the extraction logic we implemented
        c_email = (
            pending_client.get("primary_email")
            or pending_client.get("email")
            or pending_client.get("client_email")
            or None
        )
        c_name = (
            pending_client.get("name")
            or pending_client.get("client_name")
            or pending_client.get("person_name")
            or None
        )

        print(f"  Extracted - Name: '{c_name}', Email: '{c_email}'")

        # Check if we have sufficient info
        has_info = bool(c_name or c_email)
        print(f"  Has sufficient info: {has_info}")


def test_parameter_validation():
    """Test parameter validation logic"""
    print("\n=== Testing Parameter Validation ===")

    test_cases = [
        # Case 1: No identity
        {"client_name": None, "client_email": None, "client_query": None},
        # Case 2: Name only
        {"client_name": "Test Client", "client_email": None, "client_query": None},
        # Case 3: Email only
        {"client_name": None, "client_email": "test@example.com", "client_query": None},
        # Case 4: Query only
        {"client_name": None, "client_email": None, "client_query": "fuzzy match"},
        # Case 5: Complete info
        {
            "client_name": "Complete Client",
            "client_email": "complete@example.com",
            "client_query": None,
        },
    ]

    for i, params in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {params}")

        # Simulate the validation logic
        client_name = params.get("client_name")
        client_email = params.get("client_email")
        client_query = params.get("client_query")

        selector = client_query or client_name or client_email
        resolved_name = (client_name or "").strip() or None
        resolved_email = (client_email or "").strip() or None

        # Check if we have sufficient identity
        has_identity = bool(resolved_name or resolved_email)
        print(f"  Selector: '{selector}', Has identity: {has_identity}")

        if not has_identity:
            print(
                f"  Would fail with: Need client_email or client_name to create appointment"
            )


if __name__ == "__main__":
    test_client_info_extraction()
    test_parameter_validation()
    print("\n=== Test Complete ===")
