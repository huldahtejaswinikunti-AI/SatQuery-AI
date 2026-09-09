import React from 'react';
import type { ObservationDomain, ObservationItem, Scenario } from '../../types';
import { getImageUrl } from '../../api/client';

interface ObservationGalleryProps {
  domain: ObservationDomain;
  scenarios: Scenario[];
  selectedScenarioIdx: number;
  onSelectScenario: (idx: number) => void;
  selectedItemIdx: number;
  onSelectItem: (idx: number) => void;
}

export const ObservationGallery: React.FC<ObservationGalleryProps> = ({
  domain,
  scenarios,
  selectedScenarioIdx,
  onSelectScenario,
  selectedItemIdx,
  onSelectItem,
}) => {
  const currentScenario = scenarios[selectedScenarioIdx];
  const items: ObservationItem[] = currentScenario?.sample_items || [];

  return (
    <div className="glass-panel p-5 mb-6">
      {/* Scenario Header & Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-white/10 mb-5">
        <div>
          <div className="text-[11px] font-mono uppercase tracking-widest text-teal-400 font-bold mb-1">
            {domain === 'lunar' ? 'CHANDRAYAAN-2 LUNAR EXPLORATION' : 'EARTH OBSERVATION PROGRAM'}
          </div>
          <h2 className="text-xl font-bold text-white tracking-wide">
            {domain === 'lunar' ? 'Lunar Surface Scenarios & Target Formations' : 'Terrestrial Scenarios & Surface Reconnaissance'}
          </h2>
        </div>

        {/* Scenario dropdown selector */}
        <div className="min-w-[280px]">
          <select
            value={selectedScenarioIdx}
            onChange={(e) => {
              onSelectScenario(Number(e.target.value));
              onSelectItem(0);
            }}
            className="w-full bg-space-900 border border-cyan-500/30 text-slate-100 text-xs rounded-lg px-3 py-2.5 focus:outline-none focus:border-teal-400 font-medium"
          >
            {scenarios.map((scen, idx) => (
              <option key={idx} value={idx}>
                {idx + 1}. {scen.label || scen.query.slice(0, 45)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ISRO Context Note */}
      {(currentScenario?.why_this_matters_to_isro || currentScenario?.why_it_matters) && (
        <div className="text-xs text-slate-400 font-mono mb-5 flex items-start gap-2 bg-space-900/50 p-3 rounded-lg border border-white/5">
          <span className="text-teal-400 font-bold">ISRO CONTEXT:</span>
          <span>{currentScenario.why_this_matters_to_isro || currentScenario.why_it_matters}</span>
        </div>
      )}

      {/* Available Observations Grid */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-300">
            Available Observations{' '}
            <span className="text-slate-500 font-normal">({items.length} Multi-Image Options)</span>
          </div>
          <span className="text-[11px] font-mono text-teal-400">Click any observation to inspect</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
          {items.map((item, idx) => {
            const isSelected = idx === selectedItemIdx;
            const targetFile = item.file || (item.files && item.files[0]) || '';
            const imgUrl = targetFile ? getImageUrl(targetFile, domain) : '';

            return (
              <div
                key={item.id || idx}
                onClick={() => onSelectItem(idx)}
                className={`cursor-pointer rounded-lg overflow-hidden border transition-all duration-200 flex flex-col justify-between ${
                  isSelected
                    ? 'bg-space-900 border-teal-400 shadow-glow-teal scale-[1.02]'
                    : 'bg-space-900/60 border-white/10 hover:border-cyan-400/40 hover:bg-space-900/90'
                }`}
              >
                {/* Header Meta */}
                <div className="p-2.5 pb-1">
                  <div className="flex items-center justify-between gap-1 mb-1">
                    <span className="text-[10px] font-mono font-bold text-cyan-400 uppercase truncate">
                      {item.sensor || 'Raster'}
                    </span>
                    {item.resolution && (
                      <span className="text-[10px] font-mono text-slate-400">{item.resolution}</span>
                    )}
                  </div>
                  <h3
                    className="text-xs font-semibold text-slate-100 truncate"
                    title={item.name || `Observation ${idx + 1}`}
                  >
                    {item.name || `Observation ${idx + 1}`}
                  </h3>
                </div>

                {/* Thumbnail Preview */}
                <div className="relative w-full h-24 bg-space-950 overflow-hidden my-1">
                  {imgUrl ? (
                    <img
                      src={imgUrl}
                      alt={item.name}
                      className="w-full h-full object-cover"
                      loading="lazy"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-[10px] text-slate-600 font-mono">
                      PREVIEW
                    </div>
                  )}
                  {isSelected && (
                    <div className="absolute top-1 right-1 bg-teal-400 text-space-950 text-[9px] font-extrabold px-1.5 py-0.5 rounded shadow">
                      ACTIVE
                    </div>
                  )}
                </div>

                {/* Footer Info */}
                <div className="px-2.5 py-1.5 border-t border-white/5 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                  <span>{item.date || 'Archival'}</span>
                  <span className={isSelected ? 'text-teal-400 font-bold' : ''}>
                    {isSelected ? '✓ Selected' : `Select #${idx + 1}`}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
