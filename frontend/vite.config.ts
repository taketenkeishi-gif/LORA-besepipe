import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const apiTarget = process.env.VITE_API_TARGET || "http://127.0.0.1:5175";

export default defineConfig({
  // Keep the test instance's dependency prebundle separate from the live app.
  cacheDir: process.env.VITE_API_TARGET ? "node_modules/.vite-gallery-test" : "node_modules/.vite-live",
  resolve: { dedupe: ["react", "react-dom"] },
  plugins: [tailwindcss(), react()],
  server: {
    host: "127.0.0.1",
    port: 5176,
    strictPort: true,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: false,
      },
    },
  },
});
