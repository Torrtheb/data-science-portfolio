# Deployment Hardening Checklist

This checklist captures the operational controls that should be in place before pushing the scheduling stack to production.

## HTTPS, HSTS, and Reverse Proxy
- Terminate TLS in a managed load balancer or reverse proxy (CloudFront, Cloudflare, Nginx, Traefik, etc.).
- Provision certificates through an automated authority (ACME/Let’s Encrypt, AWS Certificate Manager, etc.) and enable auto‑renewal.
- Forward traffic to the FastAPI app only over TLS; set `ENABLE_HSTS=1` once HTTPS is enforced and have the proxy append the `Strict-Transport-Security` header.
- Enable HTTP/2 (or HTTP/3 when supported) at the edge for better security posture and performance.
- Configure reverse-proxy rate limiting and request-size caps. Recommended defaults:
  - 60 requests/minute per IP for authenticated dashboards.
  - 10 requests/minute per IP for anonymous/public endpoints.
  - 1 MB maximum request payload unless file uploads are required.
- Strip hop-by-hop headers and disable TRACE/OPTIONS where not needed.

## Secrets Management
- Store runtime secrets (database URL, API keys, SMTP credentials, NextAuth secret) in a secrets manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault) or encrypted environment variables.
- Rotate credentials quarterly or immediately after personnel changes.
- Scope database users to least privilege (separate read/write roles when possible) and disable superuser access for the application role.

## Network and Firewall Controls
- Place the application servers inside a private subnet; only expose the reverse proxy/load balancer.
- Restrict database ingress to the application security group / VPC only.
- Forbid outbound egress except for required third‑party APIs (OpenAI, SMTP provider, etc.).
- Use security groups / firewall rules to block SSH from the public internet; prefer SSM Session Manager or VPN for admin access.

## Backups and Disaster Recovery
- Ensure managed database snapshots are encrypted at rest (AWS RDS/KMS, GCP CMEK, etc.).
- Limit snapshot access to backup operators; automate snapshot rotation (e.g., 30 day retention).
- Verify that object storage buckets used for backups enforce server-side encryption and block public access.
- Test restoration procedures at least twice a year (restore snapshot into staging and validate migrations).

## Application Configuration
- Set `CORS_ALLOWED_ORIGINS` to a comma-separated list of trusted domains (e.g., `https://app.example.com`).
- For Next.js / NextAuth:
  - Configure `NEXTAUTH_URL` to the production domain.
  - Ensure the `next-auth.session-token` cookie is `Secure` and `SameSite=lax` (default). Avoid embedding secrets in the frontend.
  - Use NextAuth’s built-in CSRF protection for POST callbacks; for custom API routes, validate the `next-auth.csrf-token` token or enforce double-submit cookies.
- Confirm that all backend routes that mutate state require `require_owner`/`require_client` and that `AUTH_DISABLED` is never set to `1` outside local development.

## Monitoring and Logging
- Centralise logs (e.g., CloudWatch, Stackdriver, ELK) and add alerts for:
  - Repeated 401/403 responses (possible credential stuffing).
  - Repeated 429 responses (rate-limit pressure).
  - Prompt-injection warnings emitted by `/api/agent/chat`.
- Enable application performance monitoring (Datadog, OpenTelemetry) to detect anomalies.

## Continuous Integration / Security Testing
- Run `bandit -r backend` and the security-focused pytest suite in CI (see `.github/workflows/security.yml`).
- Consider adding dependency scanning (`pip-audit`) once the toolchain is available in CI.

## Release Checklist (Summary)
1. Update DNS to point to the HTTPS-enabled reverse proxy.
2. Set production `.env` equivalents via your secrets manager (no `.env` files on disk).
3. Enable `ENABLE_HSTS=1` and restrict `CORS_ALLOWED_ORIGINS` to real domains.
4. Verify database/network firewall rules and backup encryption policies.
5. Run the CI security workflow before tagging a release.
6. Document incident response (who rotates keys, where logs live, on-call escalation path).

Keep this document updated as the infrastructure evolves.
