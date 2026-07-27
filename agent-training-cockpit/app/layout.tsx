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
    title: "Agent Forge · 训练驾驶舱",
    description: "看清 Agent 每一次行动、错误与进步的训练工作台。",
    openGraph: {
      title: "Agent Forge · 训练驾驶舱",
      description: "看清 Agent 每一次行动、错误与进步",
      type: "website",
      images: [{ url: "/og.png", width: 1792, height: 909, alt: "Agent Forge 训练驾驶舱" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "Agent Forge · 训练驾驶舱",
      description: "看清 Agent 每一次行动、错误与进步",
      images: ["/og.png"],
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
