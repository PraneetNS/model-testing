import ModelDetailClient from '@/components/dashboard/ModelDetailClient';
import type { Metadata } from 'next';

export const metadata: Metadata = { title: 'Model Detail — Niyantrana Dashboard' };

export default async function ModelDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <ModelDetailClient modelId={id} />;
}
