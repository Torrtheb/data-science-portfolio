'use client';

import { useEffect } from 'react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('GlobalError:', { message: error?.message, digest: error?.digest });
  }, [error]);

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <div className="border border-red-200 bg-red-50 rounded-2xl p-4">
        <div className="font-semibold text-red-700 mb-1">Something went wrong</div>
        <div className="text-sm text-red-700/80 break-words">
          {error?.message || 'Unknown error'}
        </div>

        {error?.digest && (
          <details className="text-xs text-red-700/70 mt-2">
            <summary className="cursor-pointer">Details</summary>
            <div className="mt-1">Digest: {error.digest}</div>
          </details>
        )}

        <div className="mt-3 flex gap-2">
          <button
            onClick={reset}
            className="text-sm px-3 py-1.5 rounded-xl border bg-white hover:bg-gray-50"
          >
            Try again
          </button>
          <button
            onClick={() => window.location.reload()}
            className="text-sm px-3 py-1.5 rounded-xl border bg-white hover:bg-gray-50"
          >
            Reload
          </button>
        </div>
      </div>
    </div>
  );
}
