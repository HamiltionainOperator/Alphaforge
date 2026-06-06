import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// FastAPI always runs on localhost regardless of whether we're in LAN mode —
// the Vite dev server is what binds to 0.0.0.0 and proxies through.
const API_TARGET = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
      "/health": { target: API_TARGET, changeOrigin: true },
      "/ws": { target: API_TARGET.replace("http", "ws"), changeOrigin: true, ws: true },
    },
  },
});

