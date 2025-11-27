// front_app/src/lib/stream.ts
export type OnToken = (token: string) => void;
export type OnDone = (payload: any) => void;
export type OnError = (err: any) => void;
export type OnEvent = (payload: any) => void;

export type PostSSEOptions = {
  headers?: Record<string, string>;
  credentials?: RequestCredentials; // "include" if you use cookies
  signal?: AbortSignal;             // cancel in-flight requests
  apiKey?: string;                  // convenience for x-api-key
  sessionId?: string;               // convenience for x-session-id
  onEvent?: OnEvent;                // observe every parsed frame
};

/**
 * Post to an SSE endpoint that returns lines like:
 *   data: {"token":"..."}
 *   data: {"done":true,"sources":[...], ...}
 *
 * Splits events on blank lines, tolerates CRLF and multi-line data.
 * Calls onToken for each token, and onDone once when {done:true} arrives.
 * Throws if initial response is !ok or body missing.
 *
 * Returns a handle with .abort() so callers can cancel the stream.
 */
export async function postSSE(
  url: string,
  body: any,
  onToken: OnToken,
  onDone: OnDone,
  onError?: OnError,
  opts: PostSSEOptions = {}
) {
  const headers: Record<string, string> = {
    "content-type": "application/json",
    ...(opts.headers || {}),
  };
  if (opts.apiKey) headers["x-api-key"] = opts.apiKey;
  if (opts.sessionId) headers["x-session-id"] = opts.sessionId;

  const ctrl = new AbortController();
  const signal = opts.signal ?? ctrl.signal;

  // Build RequestInit without undefined props
  const init: RequestInit = {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal,
  };
  if (opts.credentials) {
    init.credentials = opts.credentials;
  }

  const res = await fetch(url, init);

  if (!res.ok || !res.body) {
    const msg = `SSE request failed: ${res.status} ${res.statusText}`;
    onError?.(msg);
    throw new Error(msg);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";          
  let doneCalled = false;  
  let sawAnyTokens = false;

  const abortHandle = { abort: () => ctrl.abort() };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      let chunk = decoder.decode(value, { stream: true });
      // Strip BOM if present
      if (buffer === "" && chunk.charCodeAt(0) === 0xfeff) {
        chunk = chunk.slice(1);
      }
      // Normalize newlines
      buffer += chunk.replace(/\r\n/g, "\n");

      // Process any full SSE events currently in buffer
      let sepIndex: number;
      while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, sepIndex);
        buffer = buffer.slice(sepIndex + 2);

        // Extract only "data:" lines and join; tolerate multi-line payloads
        const dataLines = rawEvent
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim());

        if (!dataLines.length) continue;

        const dataStr = dataLines.join("\n");
        if (!dataStr) continue;

        let payload: any;
        try {
          payload = JSON.parse(dataStr);
        } catch {
          onError?.(`Bad JSON in SSE frame: ${dataStr}`);
          continue;
        }

        opts.onEvent?.(payload);

        // --- TOKEN FRAME ----------------------------------------------------
        if ("token" in payload) {
          const t = (payload.token ?? "") as string;
          if (t !== "") {
            sawAnyTokens = true;
            onToken(t);
          }
          continue;
        }

        // --- ERROR FRAME ----------------------------------------------------
        if (payload?.error) {
          onError?.(payload.error);
          continue;
        }

        // --- DONE FRAME -----------------------------------------------------
        if (payload?.done) {
          doneCalled = true;
          const finalText =
            (payload.text as string) ??
            (payload.final as string) ??
            "";

          if (!sawAnyTokens && finalText) {
            onToken(finalText);
            sawAnyTokens = true;
          }
          onDone(payload);
        }
      }
    }

    // If the stream ended without an explicit {done:true}, finalize anyway.
    if (!doneCalled) {
      onDone({ text: sawAnyTokens ? undefined : "" });
    }
  } catch (e: any) {
    if (e?.name === "AbortError") {
      onError?.("stream_aborted");
      return abortHandle;
    }
    onError?.(e);
    throw e;
  }

  return abortHandle;
}
