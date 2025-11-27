from langchain_core.messages import ToolMessage

# Import the post_tools normalizer directly
from backend.agent.graph import post_tools


def test_email_draft_ui_marker_is_last():
    # Simulate a tool result from create_email_draft_tool that returns
    # the standard marker payload captured by post_tools
    tool_output = {
        "marker": "email_draft",
        "payload": {
            "draft_id": "00000000-0000-0000-0000-000000000000",
            "to": "client@example.com",
            "to_name": "Client",
            "subject": "Hello",
            "text": "Body",
            "html": "<p>Body</p>",
            "status": "pending",
            "recipients": [],
        },
    }

    # The graph’s post_tools expects the last message to be a ToolMessage
    last = ToolMessage(
        name="create_email_draft",
        tool_call_id="call-create-email-draft",
        content=tool_output,
    )

    state = {"messages": [last]}
    out = post_tools(state, config={})

    msgs = out.get("messages") or []
    # Expect at least three AI messages (brief, pending, UI) plus a JSON fallback
    assert len(msgs) >= 3

    brief = msgs[0]
    assert (
        isinstance(getattr(brief, "content", None), str)
        and "drafted an email" in brief.content.lower()
    )

    # Must include a pending marker and a UI marker in order
    contents = [getattr(m, "content", "") for m in msgs]
    assert any(
        isinstance(c, str) and c.startswith("PENDING_EMAIL_SEND:") for c in contents
    )
    assert any(isinstance(c, str) and c.startswith("UI:EMAIL_DRAFT:") for c in contents)

    # The last message should be either the UI marker or a JSON fallback with marker=email_draft
    last = contents[-1]
    is_ui_last = isinstance(last, str) and last.startswith("UI:EMAIL_DRAFT:")
    is_json_last = False
    if isinstance(last, str) and last.strip().startswith("{"):
        import json as _json

        try:
            j = _json.loads(last)
            is_json_last = isinstance(j, dict) and j.get("marker") == "email_draft"
        except Exception:
            is_json_last = False
    assert is_ui_last or is_json_last
