from __future__ import annotations
from email.message import EmailMessage
import os


os.environ.setdefault("BACKEND_DATABASE_URL", "sqlite:///:memory:")

import services.emailer as emailer  # module under test


def test_render_basic_html_with_logo_and_escape(monkeypatch):
    monkeypatch.setattr(
        emailer, "EMAIL_BRAND_LOGO_URL", "https://example.com/logo.png", raising=False
    )
    html = emailer.render_basic_html("Welcome", ["Hello <b>World</b>", "", "Thanks!"])
    assert "Welcome" in html
    assert "https://example.com/logo.png" in html
    # Ensure HTML is escaped inside paragraph
    assert "&lt;b&gt;World&lt;/b&gt;" in html
    # Blank line becomes spacer paragraph
    assert "&nbsp;" in html


def test_render_basic_html_without_logo(monkeypatch):
    monkeypatch.setattr(emailer, "EMAIL_BRAND_LOGO_URL", "", raising=False)
    html = emailer.render_basic_html("Notice", ["Line 1", "Line 2"])
    assert "img src" not in html


def test_send_email_missing_config_prints_notice(capsys, monkeypatch):
    # Clear SMTP config so function early-returns with a log line
    monkeypatch.setattr(emailer, "SMTP_HOST", "", raising=False)
    monkeypatch.setattr(emailer, "SMTP_PORT", 587, raising=False)
    monkeypatch.setattr(emailer, "SMTP_USER", "", raising=False)
    monkeypatch.setattr(emailer, "SMTP_PASS", "", raising=False)
    monkeypatch.setattr(emailer, "MAIL_FROM", "", raising=False)

    emailer.send_email("to@example.com", "Subj", "Body")
    out = capsys.readouterr().out
    assert "Missing SMTP config" in out and "to=to@example.com" in out


def test_send_email_starttls_with_html_and_ics(monkeypatch):
    calls = {"smtp": 0, "ssl": 0}
    send_args = {}

    class DummySMTP:
        def __init__(self, host, port, **kwargs):
            calls["smtp"] += 1
            self.host, self.port, self.kwargs = host, port, kwargs
            self.started_tls = False
            self.logged_in = None
            self.sent = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self, context=None):
            self.started_tls = True

        def login(self, user, pw):
            self.logged_in = (user, pw)

        def send_message(self, msg: EmailMessage):
            self.sent = msg
            send_args["msg"] = msg

    class DummySMTP_SSL(DummySMTP):
        def __init__(self, host, port, **kwargs):
            calls["ssl"] += 1
            super().__init__(host, port, **kwargs)

    # Use STARTTLS on port 587
    monkeypatch.setattr(emailer, "SMTP_HOST", "smtp.test", raising=False)
    monkeypatch.setattr(emailer, "SMTP_PORT", 587, raising=False)
    monkeypatch.setattr(emailer, "SMTP_USER", "user", raising=False)
    monkeypatch.setattr(emailer, "SMTP_PASS", "pass", raising=False)
    monkeypatch.setattr(
        emailer, "MAIL_FROM", "Scheduler <no-reply@test>", raising=False
    )
    monkeypatch.setattr(emailer, "SMTP_FROM_EMAIL", "reply@test", raising=False)
    monkeypatch.setattr(emailer, "SMTP_FROM_NAME", "Support", raising=False)
    monkeypatch.setattr(emailer, "SMTP_STARTTLS", True, raising=False)
    monkeypatch.setattr(emailer, "SMTP_TIMEOUT", "5", raising=False)

    monkeypatch.setattr(emailer.smtplib, "SMTP", DummySMTP)
    monkeypatch.setattr(emailer.smtplib, "SMTP_SSL", DummySMTP_SSL)

    emailer.send_email(
        to="alice@example.com",
        subject="Hello",
        text="Plain body",
        html="<b>HTML</b>",
        ics_text="BEGIN:VCALENDAR\nEND:VCALENDAR\n",
    )

    # Ensure SMTP (not SSL) used and STARTTLS performed
    assert calls == {"smtp": 1, "ssl": 0}
    msg = send_args["msg"]
    assert isinstance(msg, EmailMessage)

    # Headers
    assert msg["From"] == "Scheduler <no-reply@test>"
    assert msg["To"] == "alice@example.com"
    assert msg["Subject"] == "Hello"
    assert msg["Reply-To"] == "Support <reply@test>"

    # Parts: should include text/html and text/calendar attachment
    content_types = [(p.get_content_type(), p.get_filename()) for p in msg.walk()]
    assert ("text/html", None) in content_types
    assert ("text/calendar", "appointment.ics") in content_types


def test_send_email_ssl_when_no_starttls_and_465(monkeypatch):
    counters = {"smtp": 0, "ssl": 0}

    class DummySMTP:
        def __init__(self, *a, **k):
            counters["smtp"] += 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def login(self, *a, **k):
            pass

        def send_message(self, msg):
            pass

    class DummySMTP_SSL(DummySMTP):
        def __init__(self, *a, **k):
            counters["ssl"] += 1
            # Intentionally avoid calling super().__init__ to prevent
            # incrementing the 'smtp' counter when SSL path is chosen.

    # Force SSL path
    monkeypatch.setattr(emailer, "SMTP_HOST", "smtp.test", raising=False)
    monkeypatch.setattr(emailer, "SMTP_PORT", 465, raising=False)
    monkeypatch.setattr(emailer, "SMTP_USER", "user", raising=False)
    monkeypatch.setattr(emailer, "SMTP_PASS", "pass", raising=False)
    monkeypatch.setattr(emailer, "MAIL_FROM", "no-reply@test", raising=False)
    monkeypatch.setattr(emailer, "SMTP_STARTTLS", False, raising=False)

    monkeypatch.setattr(emailer.smtplib, "SMTP", DummySMTP)
    monkeypatch.setattr(emailer.smtplib, "SMTP_SSL", DummySMTP_SSL)

    emailer.send_email("x@y", "Test", "Body")
    assert counters == {"smtp": 0, "ssl": 1}
