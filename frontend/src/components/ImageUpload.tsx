/**
 * RadAssist AI — Image Upload (Phase 4)
 *
 * Drag-and-drop upload for DICOM studies, photographed reports, and images.
 *
 * WHY A SOURCE-TYPE SELECTOR RATHER THAN PURE AUTO-DETECTION:
 * The backend detects DICOM from its magic bytes, so that needs no help. But
 * it cannot tell a photograph of a printed report from any other photograph —
 * and that distinction decides whether the file gets OCR'd into the
 * searchable knowledge base. Only the person uploading knows.
 */
"use client";

import { useCallback, useRef, useState } from "react";
import {
  imageApi,
  type ImageSourceType,
  type MedicalImage,
} from "@/lib/api";

interface Props {
  onUploaded?: (image: MedicalImage) => void;
  documentId?: string;
}

type Phase = "idle" | "uploading" | "processing" | "done" | "error";

export default function ImageUpload({ onUploaded, documentId }: Props) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [message, setMessage] = useState("");
  const [dragging, setDragging] = useState(false);
  const [sourceType, setSourceType] = useState<ImageSourceType>("image_upload");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setPhase("uploading");
      setMessage(`Uploading ${file.name}...`);

      try {
        const accepted = await imageApi.upload(file, {
          sourceType,
          documentId,
        });

        // Upload returns immediately; the work happens in the background.
        setPhase("processing");
        setMessage(
          accepted.detected_type === "dicom"
            ? "DICOM detected — de-identifying and rendering..."
            : accepted.detected_type === "report"
            ? "Reading text from the report..."
            : "Processing image..."
        );

        const done = await imageApi.waitForProcessing(accepted.id);

        if (done.status === "failed") {
          setPhase("error");
          setMessage(done.error_message || "Processing failed.");
          return;
        }

        setPhase("done");
        setMessage(
          done.source_type === "report_upload" && done.ocr_text
            ? `Done — extracted ${done.ocr_text.length} characters of text.`
            : "Done."
        );
        onUploaded?.(done);

        // Clear the input so the same file can be re-selected.
        if (inputRef.current) inputRef.current.value = "";
      } catch (e) {
        setPhase("error");
        setMessage(e instanceof Error ? e.message : "Upload failed.");
      }
    },
    [sourceType, documentId, onUploaded]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const busy = phase === "uploading" || phase === "processing";

  return (
    <div className="upload-panel">
      {/* ── What kind of file is this? ── */}
      <div className="upload-types" role="radiogroup" aria-label="Image type">
        {(
          [
            ["image_upload", "Image", "X-ray, CT slice, or any picture"],
            ["report_upload", "Report photo", "Text will be extracted by OCR"],
            ["dicom_upload", "DICOM study", "Will be de-identified on upload"],
          ] as const
        ).map(([value, label, hint]) => (
          <button
            key={value}
            role="radio"
            aria-checked={sourceType === value}
            disabled={busy}
            onClick={() => setSourceType(value)}
            className={`upload-type ${sourceType === value ? "active" : ""}`}
            title={hint}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── Drop zone ── */}
      <div
        className={`dropzone ${dragging ? "is-dragging" : ""} ${busy ? "is-busy" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!busy) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={busy ? (e) => e.preventDefault() : onDrop}
        onClick={() => !busy && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          hidden
          // .dcm and no-extension DICOM files both need to be selectable —
          // PACS exports are often named IM000001 with no suffix at all.
          accept=".dcm,.dicom,image/*,application/octet-stream"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />

        {busy ? (
          <>
            <div className="dropzone-spinner" />
            <p className="dropzone-text">{message}</p>
          </>
        ) : (
          <>
            <div className="dropzone-icon">🩻</div>
            <p className="dropzone-text">
              Drop a file here, or <span className="underline">browse</span>
            </p>
            <p className="dropzone-hint">
              DICOM · PNG · JPG · TIFF — up to 200 MB
            </p>
          </>
        )}
      </div>

      {/* ── Result ── */}
      {phase === "done" && <p className="upload-ok">✅ {message}</p>}
      {phase === "error" && <p className="upload-err">⚠️ {message}</p>}

      {sourceType === "dicom_upload" && !busy && (
        <p className="upload-note">
          Patient name, ID, dates, institution and all unrecognised tags are
          discarded on upload. Only an allowlist of clinical tags is kept.
        </p>
      )}
    </div>
  );
}
