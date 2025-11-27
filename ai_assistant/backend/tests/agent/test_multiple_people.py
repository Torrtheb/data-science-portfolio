#!/usr/bin/env python3
"""
Test the enhanced booking system with multiple people scenarios
"""


def test_multiple_people_logic():
    """Test the logic for handling clients with multiple people"""
    import json

    # Simulate a client with multiple people (like "fluffy" with Thor and Bunny)
    pending_client = {
        "id": "6",
        "name": "fluffy",
        "primary_email": "bethtorrance55@hotmail.com",
        "people": [
            {"person_id": "3", "full_name": "Thor", "email": "thor@gmail.com"},
            {"person_id": "6", "full_name": "Bunny", "email": "bunny@gmail.com"},
        ],
    }

    print("=== Test Multiple People Handling ===")
    print(f"Client: {pending_client['name']}")
    print(f"People: {[p['full_name'] for p in pending_client['people']]}")

    # Test 1: Should detect multiple people
    people = pending_client.get("people", [])
    has_multiple = len(people) > 1
    print(f"Has multiple people: {has_multiple}")

    if has_multiple:
        people_names = [p.get("full_name", "") for p in people if p.get("full_name")]
        print(f"Should ask: Who is this for? Choose from: {', '.join(people_names)}")

    # Test 2: User chooses "Thor"
    user_choice = "Thor"
    print(f"\nUser chooses: {user_choice}")

    chosen_person = None
    for person in people:
        if person.get("full_name", "").lower() == user_choice.lower():
            chosen_person = person
            break

    if chosen_person:
        print(f"Found match: {chosen_person}")

        # Test enhanced client data
        enhanced_client = dict(pending_client)
        enhanced_client["chosen_person"] = chosen_person
        enhanced_client["chosen_person_id"] = chosen_person.get("person_id")
        enhanced_client["chosen_person_name"] = chosen_person.get("full_name")
        enhanced_client["chosen_person_email"] = chosen_person.get("email")

        print(f"Enhanced client data: {json.dumps(enhanced_client, indent=2)}")

        # Test client info extraction with chosen person
        c_name = None
        c_email = None
        if enhanced_client.get("chosen_person_name"):
            c_name = enhanced_client.get("chosen_person_name")
            c_email = enhanced_client.get("chosen_person_email")
        else:
            c_name = enhanced_client.get("name")
            c_email = enhanced_client.get("primary_email")

        print(f"\nExtracted for booking:")
        print(f"  Name: {c_name}")
        print(f"  Email: {c_email}")
        print(f"  Person ID: {enhanced_client.get('chosen_person_id')}")
    else:
        print(f"No match found for '{user_choice}'")


def test_booking_parameters():
    """Test the booking parameters that would be generated"""
    import json

    print("\n=== Test Booking Parameters ===")

    # Simulate enhanced client with chosen person
    enhanced_client = {
        "id": "6",
        "name": "fluffy",
        "primary_email": "bethtorrance55@hotmail.com",
        "chosen_person_id": "3",
        "chosen_person_name": "Thor",
        "chosen_person_email": "thor@gmail.com",
    }

    # Test parameter generation
    start_local = "2025-09-27T10:00"
    duration_min = 30

    booking_params = {
        "start_local": start_local,
        "duration_min": duration_min,
        "person_id": (
            int(enhanced_client.get("chosen_person_id", 0))
            if enhanced_client.get("chosen_person_id")
            else None
        ),
        "client_name": enhanced_client.get("chosen_person_name"),
        "client_email": enhanced_client.get("chosen_person_email"),
    }

    print(f"Booking parameters: {json.dumps(booking_params, indent=2)}")
    print("✓ This should successfully book for Thor specifically")


if __name__ == "__main__":
    test_multiple_people_logic()
    test_booking_parameters()
    print("\n=== Tests Complete ===")
