from __future__ import annotations
import os
import smtplib
import ssl
from email.message import EmailMessage
from html import escape

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
MAIL_FROM = os.getenv("MAIL_FROM", "")
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "").strip()
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "").strip()
SMTP_TIMEOUT = os.getenv("SMTP_TIMEOUT")
EMAIL_BRAND_LOGO_URL = os.getenv("EMAIL_BRAND_LOGO_URL", "").strip()


BRAND_GREEN = "#16a34a"


def render_basic_html(
    title: str, lines: list[str], brand_color: str = BRAND_GREEN
) -> str:
    """Render a simple branded HTML email body.

    The function wraps plain text lines in styled paragraph tags and adds a
    colored header section. Any provided brand logo URL is included if present
    in 'EMAIL_BRAND_LOGO_URL'.

    Args:
        title: Heading text displayed in the email header bar.
        lines: List of paragraph lines; blank entries render as spacers.
        brand_color: Hex color string for the header background.

    Returns:
        A complete HTML document as a string suitable for use as an email
        alternative part.
    """
    # Convert line list to <p> blocks; preserve blank lines as spacers.
    parts = []
    for ln in lines:
        if ln.strip() == "":
            parts.append("<p style='margin:12px 0'>&nbsp;</p>")
        else:
            safe_ln = escape(ln)
            parts.append(
                f"<p style='margin:8px 0; color:#111827; font-size:14px; line-height:1.5'>{safe_ln}</p>"
            )
    body = "\n".join(parts)
    logo_html = (
        f'<img src="{EMAIL_BRAND_LOGO_URL}" alt="" style="height:22px;vertical-align:middle;margin-right:10px;border-radius:4px"/>'
        if EMAIL_BRAND_LOGO_URL
        else ""
    )
    return f"""
<!doctype html>
<html>
  <body style="margin:0;background:#f8fafc;font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#111827">
    <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
      <tr><td align="center" style="padding:28px 16px">
        <table width="100%" style="max-width:640px;background:#ffffff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,0.06)" role="presentation">
          <tr>
            <td style="padding:16px 20px;border-bottom:1px solid #eef2f7;background:{brand_color};color:#fff;border-top-left-radius:12px;border-top-right-radius:12px">
              {logo_html}<h1 style=\"display:inline-block;margin:0;font-size:16px;font-weight:600;vertical-align:middle\">{title}</h1>
            </td>
          </tr>
          <tr>
            <td style="padding:20px">{body}</td>
          </tr>
          <tr>
            <td style="padding:12px 20px;border-top:1px solid #eef2f7;color:#6b7280;font-size:12px">
              Sent by Scheduler
            </td>
          </tr>
        </table>
      </td></tr>
    </table>
  </body>
</html>
"""


def send_email(
    to: str,
    subject: str,
    text: str,
    html: str | None = None,
    ics_text: str | None = None,
) -> None:
    """Send an email via SMTP with optional HTML and calendar attachment.

    Uses STARTTLS when enabled or implicit SSL when connecting to port 465
    without STARTTLS. If mandatory SMTP environment variables are missing, the
    function logs a message and returns without raising, to avoid breaking core
    flows in non-email environments.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        text: Plaintext body; always included. Used as fallback for HTML.
        html: Optional HTML alternative body.
        ics_text: Optional iCalendar content to attach as 'appointment.ics'.

    Returns:
        None. Raises on SMTP errors only when configuration is present.
    """
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS and MAIL_FROM):
        print(f"[EMAILER] Missing SMTP config; WOULD send to={to} subject={subject}")
        return

    msg = EmailMessage()
    msg["From"] = MAIL_FROM
    msg["To"] = to
    msg["Subject"] = subject

    if html:
        msg.set_content(text)
        msg.add_alternative(html, subtype="html")
    else:
        msg.set_content(text)

    if SMTP_FROM_EMAIL:
        reply_to = SMTP_FROM_EMAIL
        if SMTP_FROM_NAME:
            reply_to = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
        msg["Reply-To"] = reply_to

    if ics_text:
        msg.add_attachment(
            ics_text.encode("utf-8"),
            maintype="text",
            subtype="calendar",
            filename="appointment.ics",
        )

    smtp_kwargs = {}
    if SMTP_TIMEOUT:
        try:
            smtp_kwargs["timeout"] = float(SMTP_TIMEOUT)
        except ValueError:
            pass

    smtp_cls = (
        smtplib.SMTP_SSL if (not SMTP_STARTTLS and SMTP_PORT == 465) else smtplib.SMTP
    )

    with smtp_cls(SMTP_HOST, SMTP_PORT, **smtp_kwargs) as server:
        if SMTP_STARTTLS and isinstance(server, smtplib.SMTP):
            server.starttls(context=ssl.create_default_context())
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
