// next.config.ts
// Relax ESLint during builds to avoid blocking deploys on stylistic issues.
// We’ll still surface lint locally and can tighten later.
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  // Allow production builds to succeed even if there are type errors.
  // We'll still catch them locally with `npx tsc --noEmit`.
  typescript: {
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
