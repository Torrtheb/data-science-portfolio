// next.config.mjs
/** @type {import('next').NextConfig} */

// Prefer API_BASE_URL; fall back to BACKEND_URL; then localhost.
const BACKEND =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.NEXT_PUBLIC_BACKEND_URL ||
  'http://127.0.0.1:8080';

const isProd = process.env.NODE_ENV === 'production';

// Loosened CSP that works with Next.js hydration + your backend
const csp = [
  "default-src 'self'",
  "base-uri 'self'",
  `script-src 'self' 'unsafe-inline'${isProd ? '' : " 'unsafe-eval'"} blob:`,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob: https:",
  "font-src 'self' data:",
  `connect-src 'self' ${BACKEND} https: wss:`,
  "frame-ancestors 'none'",
  "form-action 'self'",
  "object-src 'none'",
  "upgrade-insecure-requests"
].join('; ');

const securityHeaders = [
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'X-Frame-Options', value: 'DENY' },
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
  {
    key: 'Permissions-Policy',
    value: [
      'camera=()',
      'microphone=()',
      'geolocation=()',
      'accelerometer=()',
      'autoplay=()',
      'encrypted-media=()',
      'fullscreen=(self)',
      'payment=()',
    ].join(', '),
  },
  // While stabilizing, keep Report-Only to avoid blocking. Once okay, swap to enforcing:
  { key: 'Content-Security-Policy-Report-Only', value: csp },
  ...(isProd ? [{ key: 'Strict-Transport-Security', value: 'max-age=31536000; includeSubDomains; preload' }] : []),

  { key: 'X-Debug-Backend', value: BACKEND },
];

const nextConfig = {
  reactStrictMode: true,

  experimental: {
    serverActions: { allowedOrigins: ['*'] },
  },

  async headers() {
    return [{ source: '/:path*', headers: securityHeaders }];
  },

  async rewrites() {
    return [
      { source: '/api/:path*', destination: `${BACKEND}/api/:path*` },
      { source: '/api/backend/:path*', destination: `${BACKEND}/api/:path*` },
      { source: '/undefined/api/:path*', destination: `${BACKEND}/api/:path*` },
      { source: '/_debug/:path*',        destination: `${BACKEND}/api/_debug/:path*` }, // << add this

    ];
  },

  compress: false,
};

export default nextConfig;
