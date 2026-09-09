import React, { useState } from 'react';
import type { ObservationItem, AnalysisResult, ObservationDomain } from '../../types';
import { getImageUrl } from '../../api/client';
import { ZoomIn, ZoomOut, RotateCcw, Layers } from 'lucide-react';

interface ImageViewerProps {
  domain: ObservationDomain;
  item: ObservationItem | null;
  analysisResult: AnalysisResult | null;
  isLoading?: boolean;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  domain,
  item,
  analysisResult,
  isLoading = false,
}) => {
  const [zoom, setZoom] = useState(1);
  const [showMask, setShowMask] = useState(true);
  const [maskOpacity, setMaskOpacity] = useState(0.75);

  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.25, 3.0));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.25, 0.75));
  const handleResetZoom = () => setZoom(1);

  if (!item) {
    return (
      <div className="w-full aspect-[4/3] rounded border border-[var(--border-hairline)] bg-[var(--bg-panel-elevated)] flex flex-col items-center justify-center p-8 text-center">
        <p className="font-serif text-base text-[var(--text-secondary)] mb-1">
          No Satellite Raster Loaded
        </p>
        <span className="font-mono text-xs text-[var(--text-secondary)]">
          SELECT AN ARCHIVAL SCENE FROM THE GALLERY ABOVE
        </span>
      </div>
    );
  }

  const filePath = item.file || (item.files && item.files[0]) || '';
  const imageUrl = filePath ? getImageUrl(filePath, domain) : '';
  const maskUrl = analysisResult?.overlay;

  return (
    <div className="w-full rounded border border-[var(--border-hairline)] bg-[var(--bg-panel-elevated)] overflow-hidden shadow-2xl transition-all">
      {/* Telemetry Header Bar */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2.5 border-b border-[var(--border-hairline)] bg-[var(--bg-panel)] font-mono text-[11px] text-[var(--text-secondary)] gap-2">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">
            <span className="text-[var(--accent-verified)]">SENSOR:</span>
            <span className="text-[var(--text-primary)] font-bold">{item.sensor || 'SAT-EO'}</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="text-[var(--accent-verified)]">GSD:</span>
            <span className="text-[var(--text-primary)] font-bold">{item.resolution || '0.5m / px'}</span>
          </span>
          <span className="hidden sm:flex items-center gap-1">
            <span className="text-[var(--accent-verified)]">ACQUIRED:</span>
            <span className="text-[var(--text-primary)]">{item.date || 'ARCHIVE'}</span>
          </span>
        </div>

        {/* Layer & Zoom Toolbar */}
        <div className="flex items-center gap-2">
          {maskUrl && (
            <div className="flex items-center gap-2 mr-2 pr-2 border-r border-[var(--border-hairline)]">
              <button
                onClick={() => setShowMask(!showMask)}
                className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-sans font-medium transition-all ${
                  showMask
                    ? 'bg-[var(--accent-verified)] text-[var(--bg-void)]'
                    : 'bg-[var(--bg-panel)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                <Layers className="w-3 h-3" />
                {showMask ? 'MASK ON' : 'MASK OFF'}
              </button>

              {showMask && (
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={maskOpacity}
                  onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                  className="w-16 accent-[var(--accent-verified)] cursor-pointer"
                  title="Mask Opacity"
                />
              )}
            </div>
          )}

          <div className="flex items-center gap-1">
            <button
              onClick={handleZoomOut}
              className="p-1 rounded bg-[rgba(255,255,255,0.05)] hover:bg-[rgba(255,255,255,0.1)] text-[var(--text-primary)] transition-colors cursor-pointer"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="w-10 text-center text-[10px]">{Math.round(zoom * 100)}%</span>
            <button
              onClick={handleZoomIn}
              className="p-1 rounded bg-[rgba(255,255,255,0.05)] hover:bg-[rgba(255,255,255,0.1)] text-[var(--text-primary)] transition-colors cursor-pointer"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleResetZoom}
              className="p-1 rounded bg-[rgba(255,255,255,0.05)] hover:bg-[rgba(255,255,255,0.1)] text-[var(--text-primary)] transition-colors cursor-pointer"
              title="Reset Zoom"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Raster Canvas (never white while loading) */}
      <div className="relative w-full aspect-[4/3] bg-[var(--bg-panel-elevated)] overflow-hidden flex items-center justify-center select-none">
        <div
          className="relative transition-transform duration-150 ease-out origin-center"
          style={{ transform: `scale(${zoom})` }}
        >
          {/* Base Satellite Raster */}
          {imageUrl && (
            <img
              src={imageUrl}
              alt={item.name}
              className="max-h-[560px] w-auto object-contain rounded transition-opacity duration-300"
              style={{ opacity: 1 }}
            />
          )}

          {/* Grounding Segmentation Mask Overlay */}
          {maskUrl && showMask && (
            <img
              src={maskUrl}
              alt="Analytical Grounding Overlay"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-200"
              style={{ opacity: maskOpacity }}
            />
          )}
        </div>

        {/* High-Contrast Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 bg-[rgba(5,7,13,0.78)] backdrop-blur-sm flex flex-col items-center justify-center gap-3 z-20">
            <div className="w-9 h-9 rounded-full border-2 border-[var(--accent-verified)] border-t-transparent animate-spin" />
            <span className="font-mono text-xs tracking-wider text-[var(--text-primary)] font-bold">
              RUNNING MULTIMODAL VERIFICATION PIPELINE...
            </span>
          </div>
        )}
      </div>

      {/* Footer Raster Provenance */}
      <div className="px-4 py-2 border-t border-[var(--border-hairline)] bg-[var(--bg-panel)] flex items-center justify-between font-mono text-[10px] text-[var(--text-secondary)]">
        <span>PROVENANCE: {item.provenance || 'ISRO BHOOVAN / PRADAN / COPERNICUS'}</span>
        <span>LOCATION: {item.location || 'GLOBAL REGISTRATION'}</span>
      </div>
    </div>
  );
};
