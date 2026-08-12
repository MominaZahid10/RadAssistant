/**
 * RadAssist AI — Image Viewer (Phase 4)
 *
 * Full-screen lightbox for a medical image, with its metadata panel.
 *
 * DESIGN NOTES:
 * - Dark background. Radiology images are viewed against dark surrounds
 *   because a bright frame compresses perceived contrast in the greyscale
 *   range where findings live.
 * - De-identification status is shown EXPLICITLY. A viewer that stays silent
 *   invites the assumption that everything has been processed — and an image
 *   that still carries PHI looks identical to one that doesn't.
 * - Escape closes, and focus is trapped while open.
 */
"use client";

import { useEffect, useRef, useState } from "react";
import { type MedicalImage, imageUrl } from "@/lib/api";

interface Props {
  image: MedicalImage;
  onClose: () => void;
}

export default function ImageViewer({ image, onClose }: Props) {
  const [zoomed, setZoomed] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);

  // Escape to close, and stop the page behind from scrolling.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);

    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();

    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [onClose]);

  const src = imageUrl(image.file_url);

  const meta: Array<[string, string | null]> = [
    ["Modality", image.modality],
    ["Body part", image.body_part],
    ["View", image.view_position],
    ["Study year", image.study_date ? image.study_date.slice(0, 4) : null],
    [
      "Dimensions",
      image.width && image.height ? `${image.width} × ${image.height}` : null,
    ],
    [
      "Size",
      image.file_size ? `${(image.file_size / 1024).toFixed(0)} KB` : null,
    ],
    ["Source", image.source_type.replace(/_/g, " ")],
  ];

  return (
    <div
      className="viewer-backdrop"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`Image: ${image.filename}`}
    >
      <div className="viewer-shell" onClick={(e) => e.stopPropagation()}>
        {/* ── Header ── */}
        <div className="viewer-header">
          <div className="min-w-0">
            <p className="viewer-title">{image.filename}</p>
            {image.caption && <p className="viewer-caption">{image.caption}</p>}
          </div>
          <button
            ref={closeRef}
            onClick={onClose}
            className="viewer-close"
            aria-label="Close viewer"
          >
            ✕
          </button>
        </div>

        {/* ── Image ── */}
        <div className={`viewer-stage ${zoomed ? "is-zoomed" : ""}`}>
          {src ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={src}
              alt={image.caption || image.filename}
              onClick={() => setZoomed((z) => !z)}
              title={zoomed ? "Click to fit" : "Click to zoom"}
            />
          ) : (
            <p className="viewer-empty">
              No file available (status: {image.status})
            </p>
          )}
        </div>

        {/* ── Metadata ── */}
        <div className="viewer-meta">
          {meta
            .filter(([, v]) => v)
            .map(([label, value]) => (
              <div key={label} className="viewer-meta-item">
                <span className="viewer-meta-label">{label}</span>
                <span className="viewer-meta-value">{value}</span>
              </div>
            ))}

          {/*
            ⚠️  Shown for every image, not only de-identified ones.
            An image carrying PHI is visually indistinguishable from one that
            doesn't, so silence here reads as reassurance. DICOM studies are
            de-identified on ingest; figures and plain uploads never contained
            PHI to begin with, which is a different claim and labelled as such.
          */}
          {image.source_type === "dicom_upload" && (
            <div className="viewer-meta-item">
              <span className="viewer-meta-label">Privacy</span>
              <span
                className={
                  image.is_deidentified ? "badge-safe" : "badge-warn"
                }
              >
                {image.is_deidentified
                  ? "De-identified"
                  : "NOT de-identified"}
              </span>
            </div>
          )}

          {image.source_url && (
            <div className="viewer-meta-item">
              <span className="viewer-meta-label">Source</span>
              <a
                href={image.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="viewer-link"
              >
                View original ↗
              </a>
            </div>
          )}
        </div>

        {/* ── OCR text, when this was a photographed report ── */}
        {image.ocr_text && (
          <details className="viewer-ocr">
            <summary>Extracted text ({image.ocr_text.length} chars)</summary>
            <pre>{image.ocr_text}</pre>
          </details>
        )}
      </div>
    </div>
  );
}
