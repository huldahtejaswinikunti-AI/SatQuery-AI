import type { AnalysisResult, ObservationDomain, Scenario } from '../types';

const rawBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || '';
const API_BASE = rawBase ? (rawBase.endsWith('/api') ? rawBase : `${rawBase.replace(/\/$/, '')}/api`) : '/api';

export async function checkHealth(): Promise<{ status: string; active_models: string[] }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchScenarios(domain: ObservationDomain): Promise<Scenario[]> {
  const res = await fetch(`${API_BASE}/scenarios?domain=${domain}`);
  if (!res.ok) throw new Error(`Failed to load scenarios: ${res.statusText}`);
  return res.json();
}

export function getImageUrl(filePath: string, domain: ObservationDomain): string {
  return `${API_BASE}/image/${encodeURIComponent(filePath)}?domain=${domain}`;
}

export async function runAnalysis(
  mode: ObservationDomain,
  files: string[],
  query: string
): Promise<AnalysisResult> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, files, query }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Analysis request failed');
  }
  return res.json();
}

export async function exportPdfReport(
  result: AnalysisResult,
  query: string,
  imagesMeta?: any[]
): Promise<Blob> {
  const res = await fetch(`${API_BASE}/export-pdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ result, query, images_meta: imagesMeta }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'PDF generation request failed');
  }
  return res.blob();
}

export async function uploadCustomRasters(
  files: File[],
  domain: ObservationDomain
): Promise<{ status: string; count: number; items: any[] }> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });
  const res = await fetch(`${API_BASE}/upload?domain=${domain}`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to upload raster images');
  }
  return res.json();
}

