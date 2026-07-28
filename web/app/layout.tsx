import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PKB Evidence Workbench",
  description: "A visual workbench for evidence-grounded PKB-Agent runs.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
