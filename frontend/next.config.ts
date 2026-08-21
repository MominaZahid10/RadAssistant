import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /**
   * ⚠️  REQUIRED BY Dockerfile.prod.
   *
   * Standalone output traces the exact files the server needs and writes a
   * self-contained bundle to .next/standalone, including a minimal server.js.
   * The production image copies that instead of the whole project plus
   * node_modules — a runtime layer with no source, no devDependencies and no
   * build tooling in it.
   *
   * Without this, .next/standalone is never produced and the production image
   * fails at COPY with a confusing "not found" rather than anything that
   * points at the real cause.
   *
   * No effect on `next dev`, so development is unchanged.
   */
  output: "standalone",

  /**
   * ⚠️  PINS WHERE INSIDE .next/standalone THE SERVER LANDS.
   *
   * Next infers a "workspace root" by walking UP from this directory looking
   * for a package.json or lockfile, and mirrors the path from that root into
   * the output. With nothing above frontend/ you get
   * .next/standalone/server.js. Let a package.json appear one level up — a
   * monorepo, a stray npm install at the repo root, a tool that drops one —
   * and the same build silently produces
   * .next/standalone/frontend/server.js instead.
   *
   * That is not hypothetical: it happened while this Dockerfile was being
   * tested, and the failure surfaces as a COPY that finds nothing, or an
   * image that builds cleanly and then exits with "Cannot find module
   * /app/server.js" at runtime.
   *
   * Pinning the root to this directory makes the layout the same everywhere —
   * in Docker, on a laptop, and in whatever CI runs it later.
   */
  outputFileTracingRoot: path.join(__dirname),
};

export default nextConfig;
