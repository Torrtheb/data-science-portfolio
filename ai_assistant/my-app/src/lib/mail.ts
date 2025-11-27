// src/lib/mail.ts
import nodemailer from "nodemailer"

declare global {
   
  var __mailTransporterPromise: Promise<nodemailer.Transporter> | undefined
}

const APP_NAME = process.env.APP_NAME || "Your App"
const MAIL_FROM = process.env.MAIL_FROM || `"${APP_NAME}" <no-reply@example.com>`
const CONTACT_EMAIL = process.env.CONTACT_EMAIL

async function buildTransporter() {
  // Prefer single URL if provided
  if (process.env.SMTP_URL) {
    return nodemailer.createTransport(process.env.SMTP_URL)
  }

  const { SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS } = process.env
  if (SMTP_HOST && SMTP_PORT && SMTP_USER && SMTP_PASS) {
    return nodemailer.createTransport({
      host: SMTP_HOST,
      port: Number(SMTP_PORT),
      secure: Number(SMTP_PORT) === 465,
      auth: { user: SMTP_USER, pass: SMTP_PASS },
    })
  }

  // Dev preview inbox (Ethereal)
  const testAcc = await nodemailer.createTestAccount()
  return nodemailer.createTransport({
    host: "smtp.ethereal.email",
    port: 587,
    auth: { user: testAcc.user, pass: testAcc.pass },
  })
}

async function getTransporter() {
  if (!global.__mailTransporterPromise) {
    global.__mailTransporterPromise = buildTransporter()
  }
  return global.__mailTransporterPromise
}

export async function sendMail(opts: {
  to: string
  subject: string
  html: string
  text?: string
}) {
  const t = await getTransporter()
  const info = await t.sendMail({
    from: MAIL_FROM,
    to: opts.to,
    subject: opts.subject,
    html: opts.html,
    text: opts.text,
  })
  const preview = nodemailer.getTestMessageUrl(info)
  if (preview) {
    console.log("Ethereal preview URL:", preview)
  }
}

// Optional helpers
export async function sendPasswordResetEmail(to: string, url: string) {
  const subject = "Welcome to my studio";
  const heading = "Welcome to my studio";
  const html = `<!doctype html>
<html>
  <body style="margin:0;background:#f8fafc;font-family:system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#111827">
    <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
      <tr><td align="center" style="padding:28px 16px">
        <table width="100%" style="max-width:640px;background:#ffffff;border-radius:12px;box-shadow:0 1px 3px rgba(15,23,42,0.08)" role="presentation">
          <tr>
            <td style="padding:20px 24px;border-bottom:1px solid #e2e8f0;background:#065f46;color:#ecfdf5;border-top-left-radius:12px;border-top-right-radius:12px">
              <h1 style="margin:0;font-size:20px;font-weight:600;">${heading}</h1>
            </td>
          </tr>
          <tr>
            <td style="padding:24px">
              <p style="margin:0 0 16px;font-size:16px;line-height:1.6;">Hi there,</p>
              <p style="margin:0 0 16px;font-size:16px;line-height:1.6;">I'm excited to have you at my music studio. Whether you're joining for the first time or resetting your password, click the button below to choose a secure password. For your protection, the link will expire in 24 hours.</p>
              <p style="margin:0 0 16px;font-size:14px;line-height:1.6;color:#475569;">PS: To make sure you never miss a lesson update, add <span style="color:#0f172a;">${CONTACT_EMAIL || 'no-reply@example.com'}</span> to your contacts.</p>
              <p style="margin:32px 0;text-align:center;">
                <a href="${url}" style="display:inline-block;padding:14px 28px;background:#047857;color:#fff;font-size:16px;font-weight:600;text-decoration:none;border-radius:999px;">Choose your password</a>
              </p>
              <p style="margin:0 0 16px;font-size:14px;line-height:1.6;color:#475569;">If the button doesn’t work, copy and paste this address into your browser:<br/><span style="word-break:break-all;color:#0f172a;">${url}</span></p>
              <p style="margin:24px 0 0;font-size:16px;line-height:1.6;">Looking forward to seeing you soon,<br/>Elizabeth</p>
            </td>
          </tr>
          <tr>
            <td style="padding:16px 24px;border-top:1px solid #e2e8f0;font-size:12px;color:#64748b;">If you didn’t request this email, you can safely ignore it.</td>
          </tr>
        </table>
      </td></tr>
    </table>
  </body>
</html>`;

  const text = `Welcome to my studio!

Hi there,

I'm excited to have you at my music studio. Whether you're joining for the first time or resetting your password, open the link below within 24 hours to choose a secure password:
${url}

If you didn't request this, you can ignore this message.

Looking forward to seeing you soon,
Elizabeth

PS: To make sure you never miss a lesson update, add ${CONTACT_EMAIL || 'no-reply@example.com'} to your contacts.
`

  return sendMail({
    to,
    subject,
    html,
    text,
  })
}

export async function sendPasswordChangedNotice(to: string) {
  return sendMail({
    to,
    subject: "Your password was changed",
    html: `<p>This is a confirmation that your password has just been changed.</p><p>If you didn't do this, reset it immediately.</p>`,
    text: `Your password was changed. If this wasn't you, reset it immediately.`,
  })
}
