# ══════════════════════════════════════════════════════════════
# RadAssist AI — Frontend Dockerfile
# ══════════════════════════════════════════════════════════════
# For DEVELOPMENT, we use Node.js with hot reload (--turbopack)
# so changes appear instantly in the browser without rebuilding.
# ══════════════════════════════════════════════════════════════

FROM node:20-alpine

WORKDIR /app

# Copy package files first (Docker layer caching)
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install

# Copy the rest of the app
COPY . .

# Expose the dev server port
EXPOSE 3000

# Start Next.js dev server with Turbopack (fast refresh)
CMD ["npm", "run", "dev"]
