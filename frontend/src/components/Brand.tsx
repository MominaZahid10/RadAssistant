/**
 * RadAssistant — brand mark.
 *
 * A scanner aperture: segmented outer ring, soft inner ring, solid focal
 * point. Deliberately not a literal organ — an anatomical glyph dates fast,
 * fixes the product to one body part, and turns to mush below 24px. This
 * reads as imaging at 26px in the sidebar and at 52px on the empty state,
 * and needs no container tile behind it.
 *
 * `currentColor` throughout, so one component serves the white-on-azure
 * placement on the sign-in panel and the azure-on-white placement in the app.
 */
export default function Brand({
  size = 26,
  className,
}: {
  size?: number;
  className?: string;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      className={className}
      aria-hidden="true"
      focusable="false"
      style={{ flexShrink: 0 }}
    >
      {/* Outer aperture. The dash pattern is computed against the real
          circumference (2π × 13 ≈ 81.68) so the four gaps land evenly —
          eyeballed values leave a visibly short final segment. */}
      <circle
        cx="16"
        cy="16"
        r="13"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeDasharray="16.4 4"
      />
      <circle
        cx="16"
        cy="16"
        r="5.5"
        stroke="currentColor"
        strokeWidth="2"
        opacity="0.45"
      />
      <circle cx="16" cy="16" r="2" fill="currentColor" />
    </svg>
  );
}
