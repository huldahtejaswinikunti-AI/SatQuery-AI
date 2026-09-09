import React from 'react';
import type { ObservationDomain } from '../../types';
import { Loader2 } from 'lucide-react';

interface QueryConsoleProps {
  domain: ObservationDomain;
  query: string;
  onChangeQuery: (query: string) => void;
  onExecute: () => void;
  isLoading: boolean;
}

export const QueryConsole: React.FC<QueryConsoleProps> = ({
  domain,
  query,
  onChangeQuery,
  onExecute,
  isLoading,
}) => {
  const isLunar = domain === 'lunar';

  // Domain-specific contextual query presets
  const earthChips = [
    { label: '+ Locate Buildings', q: 'Locate all buildings and built-up structures in this scene.' },
    { label: '+ Find Water', q: 'Identify flooded water bodies, rivers, and coastal drainage.' },
    { label: '+ Describe Scene', q: 'Describe the land use and dominant terrain features in this scene.' },
    { label: '+ Detect Changes', q: 'What changed between these observation dates? Has new construction occurred?' },
    { label: '+ Analyze Veg.', q: 'Assess vegetation density and compute physical spectral indices.' },
    { label: '+ Identify Roads', q: 'Identify major transportation roads, highways, and logistics corridors.' },
  ];

  const lunarChips = [
    { label: '+ Find Craters', q: 'Detect prominent impact crater candidates across the lunar scene.' },
    { label: '+ Crater Morphology', q: 'Analyze crater rim morphology, slope gradients, and rim degradation.' },
    { label: '+ Analyze Terrain', q: 'Describe the lunar surface morphology and regolith micro-relief texture.' },
    { label: '+ Detect Shadows', q: 'Identify deep shadow zones and potential permanently shadowed regions (PSR).' },
    { label: '+ Find Ejecta', q: 'Identify high-albedo ejecta blanket patterns and radial impact rays.' },
    { label: '+ Compare Regions', q: 'Compare surface roughness, crater density, and geological formations.' },
  ];

  const chips = isLunar ? lunarChips : earthChips;

  return (
    <div className="mt-4 pt-4 border-t border-white/10">
      <div className="text-[11px] font-mono font-bold uppercase tracking-wider text-slate-400 mb-2">
        Contextual Quick Query Commands ({isLunar ? 'Lunar Mode' : 'Earth Mode'}):
      </div>

      {/* Query Chips */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-3">
        {chips.map((chip, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => onChangeQuery(chip.q)}
            className="text-left text-xs bg-space-900/80 hover:bg-space-800 border border-white/10 hover:border-teal-400/50 text-slate-300 hover:text-white px-2.5 py-1.5 rounded transition-all truncate"
            title={chip.q}
          >
            {chip.label}
          </button>
        ))}
      </div>

      {/* Query Textarea */}
      <div className="relative mb-3">
        <textarea
          value={query}
          onChange={(e) => onChangeQuery(e.target.value)}
          placeholder={
            isLunar
              ? 'Enter lunar natural language query (e.g., Identify impact craters and ejecta rays)...'
              : 'Enter remote sensing natural language query (e.g., Locate industrial buildings and assess land cover)...'
          }
          rows={3}
          className="w-full bg-space-950 border border-cyan-500/25 rounded-lg p-3 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-teal-400 font-sans resize-none"
        />
      </div>

      {/* Execution Call to Action */}
      <button
        onClick={onExecute}
        disabled={isLoading || !query.trim()}
        className={`w-full py-3 px-4 rounded-lg font-extrabold text-xs tracking-wider uppercase flex items-center justify-center gap-2 transition-all duration-200 ${
          isLoading || !query.trim()
            ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
            : 'bg-gradient-to-r from-teal-400 to-sky-500 hover:from-teal-300 hover:to-sky-400 text-space-950 shadow-glow-teal hover:shadow-glow-cyan transform hover:-translate-y-0.5'
        }`}
      >
        {isLoading ? (
          <>
            <Loader2 size={16} className="animate-spin text-space-950" />
            <span>EXECUTING AUTONOMOUS PIPELINE & CROSS-VERIFICATION...</span>
          </>
        ) : (
          <span>RUN SATQUERY ANALYSIS &rarr;</span>
        )}
      </button>
      <div className="text-[10px] font-mono text-slate-500 text-center mt-2">
        Autonomous Router &rarr; Specialist Pipeline &rarr; Deterministic Signal Cross-Verification
      </div>
    </div>
  );
};
