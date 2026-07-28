import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host =
    requestHeaders.get("x-forwarded-host") ??
    requestHeaders.get("host") ??
    "localhost:3000";
  const protocol =
    requestHeaders.get("x-forwarded-proto") ??
    (host.includes("localhost") ? "http" : "https");

  return {
    metadataBase: new URL(`${protocol}://${host}`),
    title: "Agent Forge · 可导入数据的实验看板",
    description: "在浏览器本机导入实验 JSON，对比 Agent 训练的效果、趋势与成本。",
    openGraph: {
      title: "Agent Forge · 可导入数据的实验看板",
      description: "在浏览器本机导入实验 JSON，对比 Agent 训练的效果、趋势与成本",
      type: "website",
    },
    twitter: {
      card: "summary",
      title: "Agent Forge · 可导入数据的实验看板",
      description: "在浏览器本机导入实验 JSON，对比 Agent 训练的效果、趋势与成本",
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        {children}
      </body>
    </html>
  );
}
