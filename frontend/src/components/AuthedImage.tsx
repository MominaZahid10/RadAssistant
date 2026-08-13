"use client";

/**
 * An <img> for a route that requires authentication (Phase 6, Step 6).
 *
 * ⚠️  A PLAIN <img src> CANNOT SEND AN Authorization HEADER.
 * The browser issues that request itself. It carries cookies but no bearer
 * token, so the moment /images/{id}/file started requiring auth, every
 * thumbnail broke — and broke silently, as a broken-image icon rather than an
 * error anyone would report.
 *
 * The tempting fix is `?token=...` in the URL. That is precisely what the auth
 * design already rejects: a token in a URL leaks through server logs, browser
 * history, referrer headers and screenshots — the same reasoning that made a
 * bare UUID unacceptable as an access control for these images.
 *
 * So this fetches the bytes with the header and renders them from a blob.
 */

import { useEffect, useState } from "react";
import { fetchImageObjectUrl } from "@/lib/api";

interface Props {
  /** API path, e.g. "/api/v1/images/{id}/thumbnail". */
  path: string | null;
  alt: string;
  className?: string;
}

export default function AuthedImage({ path, alt, className }: Props) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!path) return;

    let cancelled = false;
    let created: string | null = null;

    fetchImageObjectUrl(path)
      .then((url) => {
        if (cancelled) {
          // Arrived after unmount. Release it immediately or it leaks — the
          // component that would have revoked it is already gone.
          if (url) URL.revokeObjectURL(url);
          return;
        }
        created = url;
        if (url) setObjectUrl(url);
        else setFailed(true);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });

    return () => {
      cancelled = true;
      // ⚠️  An object URL pins the entire image in memory until revoked.
      // A list of thumbnails that creates them per render and never releases
      // them grows without bound for as long as the tab stays open.
      if (created) URL.revokeObjectURL(created);
    };
  }, [path]);

  if (failed) {
    return <span className={className} title={alt} aria-label={alt}>🩻</span>;
  }

  if (!objectUrl) {
    // Reserves the same space, so a grid of thumbnails does not reflow as
    // each one arrives.
    return <span className={className} aria-hidden />;
  }

  // eslint-disable-next-line @next/next/no-img-element
  return <img src={objectUrl} alt={alt} className={className} />;
}
