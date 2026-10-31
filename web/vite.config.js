import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API (FastAPI, uvicorn server.app:app --port 8000) serves saved runs; the dev server proxies to it.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
});
