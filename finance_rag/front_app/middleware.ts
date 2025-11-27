import { NextResponse, type NextRequest } from 'next/server';

// Toggle with env if you want: set to '1' to enable
const ENABLE = process.env.NEXT_PUBLIC_ENABLE_CLIENT_RATELIMIT === '1';

// Limits (per browser, via cookie)
const WINDOW_MS = 60_000; // 1 minute
const MAX_REQ = 20;   

const PATHS = [
  '/api/chat/stream',
  '/api/price',
  '/api/quote',
  '/api/td/candles',
  '/api/market/candles',
];

export function middleware(req: NextRequest) {
  if (!ENABLE) return NextResponse.next();

  const { pathname } = req.nextUrl;
  if (!PATHS.some((p) => pathname.startsWith(p))) {
    return NextResponse.next();
  }

  const now = Date.now();
  const cookie = req.cookies.get('rate')?.value || '';
  let bucket: { ts: number; count: number } = { ts: now, count: 0 };

  try {
    if (cookie) bucket = JSON.parse(cookie);
  } catch {
    bucket = { ts: now, count: 0 };
  }

  // reset window if expired
  if (now - bucket.ts > WINDOW_MS) {
    bucket = { ts: now, count: 0 };
  }

  bucket.count += 1;

  if (bucket.count > MAX_REQ) {
    const res = new NextResponse(
      JSON.stringify({ error: 'Too many requests. Please slow down.' }),
      { status: 429, headers: { 'content-type': 'application/json', 'retry-after': '60' } }
    );
    res.cookies.set('rate', JSON.stringify(bucket), { httpOnly: true, sameSite: 'lax', maxAge: 60 });
    return res;
  }

  const res = NextResponse.next();
  res.cookies.set('rate', JSON.stringify(bucket), { httpOnly: true, sameSite: 'lax', maxAge: 60 });
  return res;
}

export const config = {
  matcher: ['/api/:path*'],
};
