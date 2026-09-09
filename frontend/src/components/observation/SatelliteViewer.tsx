import React, { useState } from 'react';
import type { ObservationDomain, ObservationItem } from '../../types';
import { getImageUrl } from '../../api/client';
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

interface SatelliteViewerProps {
  domain: ObservationDomain;
  activeItem?: ObservationItem;
  overlayUri?: string | null;
  confidenceTag?: string;
}

export const SatelliteViewer: React.FC<SatelliteViewerProps> = ({
  domain,
  activeItem,
  overlayUri,
  confidenceTag: _confidenceTag,
}) => {
  const [zoom, setZoom] = useState(1);
  const [layer, setLayer] = useState<'base' | 'overlay'>('base');
  const [overlayOpacity, setOverlayOpacity] = useState(0.85);

  const targetFile = activeItem?.file || (activeItem?.files && activeItem?.files[0]) || '';
  const imgUrl = targetFile ? getImageUrl(targetFile, domain) : '';

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.25, 3));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.25, 0.75));
  const handleReset = () => setZoom(1);

  return (
    <div className="flex flex-col h-full">
      {/* Top Controls Bar */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-white/10">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Primary Satellite Raster
          </span>
          {activeItem?.sensor && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              {activeItem.sensor}
            </span>
          )}
        </div>

        {/* Layer Toggle Switch */}
        <div className="flex items-center gap-3">
          {overlayUri && (
            <div className="flex items-center gap-1.5 bg-space-900 border border-white/10 p-1 rounded-lg text-xs font-semibold">
              <button
                onClick={() => setLayer('base')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  layer === 'base' ? 'bg-teal-500/20 text-teal-300 font-bold' : 'text-slate-400 hover:text-white'
                }`}
              >
                Base Raster
              </button>
              <button
                onClick={() => setLayer('overlay')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  layer === 'overlay' ? 'bg-teal-400 text-space-950 font-extrabold shadow' : 'text-slate-400 hover:text-white'
                }`}
              >
                Grounding Mask
              </button>
            </div>
          )}

          {/* Zoom Actions */}
          <div className="flex items-center gap-1 bg-space-900 border border-white/10 rounded-lg p-0.5 text-slate-300">
            <button
              onClick={handleZoomIn}
              className="p-1.5 hover:text-teal-400 hover:bg-space-800 rounded"
              title="Zoom In"
            >
              <ZoomIn size={14} />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-1.5 hover:text-teal-400 hover:bg-space-800 rounded"
              title="Zoom Out"
            >
              <ZoomOut size={14} />
            </button>
            <button
              onClick={handleReset}
              className="p-1.5 hover:text-teal-400 hover:bg-space-800 rounded"
              title="Reset View"
            >
              <RotateCcw size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Main Image Stage */}
      <div className="relative flex-1 min-h-[360px] max-h-[460px] bg-space-950 rounded-lg overflow-hidden border border-white/10 flex items-center justify-center">
        {imgUrl ? (
          <div
            className="relative w-full h-full flex items-center justify-center overflow-hidden transition-transform duration-200"
            style={{ transform: `scale(${zoom})` }}
          >
            {/* Base Satellite Raster */}
            <img
              src={imgUrl}
              alt={activeItem?.name || 'Satellite observation'}
              className="max-w-full max-h-full object-contain select-none"
              draggable={false}
            />

            {/* Overlaid Grounding/Detection Mask */}
            {overlayUri && layer === 'overlay' && (
              <img
                src={overlayUri}
                alt="Detection Overlay"
                className="absolute inset-0 max-w-full max-h-full m-auto object-contain pointer-events-none"
                style={{ opacity: overlayOpacity }}
              />
            )}
          </div>
        ) : (
          <div className="text-sm font-mono text-slate-500">No active observation loaded</div>
        )}

        {/* Floating Layer Opacity Slider when overlay active */}
        {overlayUri && layer === 'overlay' && (
          <div className="absolute bottom-3 right-3 bg-space-900/90 backdrop-blur-md border border-white/10 px-3 py-1.5 rounded-lg flex items-center gap-2 text-xs font-mono text-slate-300">
            <span>Mask Opacity:</span>
            <input
              type="range"
              min="0.2"
              max="1"
              step="0.05"
              value={overlayOpacity}
              onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
              className="w-20 accent-teal-400 cursor-pointer"
            />
          </div>
        )}
      </div>

      {/* Telemetry Metadata Chips Bar */}
      <div className="mt-3 flex flex-wrap gap-2 text-xs font-mono">
        <div className="bg-space-900/80 border border-white/10 px-3 py-1.5 rounded text-slate-300">
          <span className="text-slate-500">Sensor:</span> {activeItem?.sensor || 'Optical / SAR'}
        </div>
        <div className="bg-space-900/80 border border-white/10 px-3 py-1.5 rounded text-slate-300">
          <span className="text-slate-500">GSD:</span> {activeItem?.resolution || 'Native GSD'}
        </div>
        <div className="bg-space-900/80 border border-white/10 px-3 py-1.5 rounded text-slate-300">
          <span className="text-slate-500">Acquisition:</span> {activeItem?.date || 'Archived Observation'}
        </div>
        <div className="bg-space-900/80 border border-white/10 px-3 py-1.5 rounded text-slate-300">
          <span className="text-slate-500">Location:</span> {activeItem?.location || 'Target Coordinates'}
        </div>
        <div className="bg-space-900/80 border border-white/10 px-3 py-1.5 rounded text-slate-300">
          <span className="text-slate-500">Archive:</span> {activeItem?.provenance || 'ISRO / Copernicus'}
        </div>
      </div>
    </div>
  );
};
