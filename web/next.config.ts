import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // The FastAPI workbench is commonly opened through 127.0.0.1 while Next
  // serves HMR. Explicitly allow both loopback host spellings in development.
  allowedDevOrigins: ["127.0.0.1", "localhost"],
};

export default nextConfig;
