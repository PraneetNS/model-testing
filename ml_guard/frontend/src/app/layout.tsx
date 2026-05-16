import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { CustomCursor } from '@/components/ui/CustomCursor';
import { RouteLoader } from '@/components/ui/RouteLoader';

const inter = Inter({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700', '800'],
  variable: '--font-inter',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['400'],
  variable: '--font-jetbrains',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Niyantrana — AI Governance Platform',
  description:
    'Behavioral contracts, drift detection, cryptographic audit certificates. The enterprise AI governance platform.',
  keywords: [
    'AI governance',
    'machine learning',
    'behavioral contracts',
    'drift detection',
    'AI compliance',
    'AIBOM',
    'governance-as-code',
  ],
  openGraph: {
    title: 'Niyantrana — AI Governance Platform',
    description: 'Behavioral contracts, drift detection, cryptographic audit certificates.',
    type: 'website',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'Niyantrana AI Governance Platform',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Niyantrana — AI Governance Platform',
    description: 'Behavioral contracts, drift detection, cryptographic audit certificates.',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className="scroll-smooth"
      style={{ scrollBehavior: 'smooth' }}
    >
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} antialiased`}
        style={{
          fontFamily: 'Inter, system-ui, sans-serif',
          backgroundColor: '#F7F6F2',
          color: '#3D3D3A',
        }}
      >
        <CustomCursor />
        <RouteLoader>{children}</RouteLoader>
      </body>
    </html>
  );
}
