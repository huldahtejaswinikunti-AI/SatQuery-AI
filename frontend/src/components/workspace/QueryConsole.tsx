import React from 'react';
import type { ObservationDomain } from '../../types';
import { Loader2, ArrowRight } from 'lucide-react';

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
    <div className="w-full space-y-3 pt-3">
      {/* Quick Command Chips */}
      <div>
        <span className="font-mono text-[10px] tracking-wider text-[var(--text-secondary)] uppercase block mb-2">
          CONTEXTUAL COMMAND PRESETS ({isLunar ? 'LUNAR MODE' : 'EARTH MODE'}):
        </span>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {chips.map((chip, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => onChangeQuery(chip.q)}
              className="text-left font-sans text-xs bg-[var(--bg-panel-elevated)] hover:bg-[rgba(77,184,255,0.08)] border border-[var(--border-hairline)] hover:border-[var(--accent-verified)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] px-2.5 py-1.5 rounded transition-all truncate focus:outline-none focus:ring-1 focus:ring-[var(--accent-verified)]"
              title={chip.q}
            >
              {chip.label}
            </button>
          ))}
        </div>
      </div>

      {/* Query Textarea */}
      <div className="space-y-1">
        <textarea
          value={query}
          onChange={(e) => onChangeQuery(e.target.value)}
          placeholder={
            isLunar
              ? 'Enter lunar natural language query (e.g., Identify impact craters and ejecta rays)...'
              : 'Enter remote sensing natural language query (e.g., Locate industrial buildings and assess land cover)...'
          }
          rows={3}
          className="w-full bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] rounded p-3 text-sm text-[var(--text-primary)] placeholder-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent-verified)] font-sans resize-none transition-all"
        />
      </div>

      {/* Run Button */}
      <button
        onClick={onExecute}
        disabled={isLoading || !query.trim()}
        className={`w-full py-3 px-4 rounded font-sans font-bold text-xs tracking-wider uppercase flex items-center justify-center gap-2 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-[var(--accent-verified)] ${
          isLoading || !query.trim()
            ? 'bg-[var(--bg-panel-elevated)] text-[var(--text-secondary)] border border-[var(--border-hairline)] cursor-not-allowed opacity-50'
            : 'bg-[var(--accent-verified)] text-[var(--bg-void)] hover:bg-[#68c6ff] shadow-[0_0_20px_rgba(77,184,255,0.35)] cursor-pointer'
        }`}
      >
        {isLoading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin text-[var(--bg-void)]" />
            <span>EXECUTING AUTONOMOUS PIPELINE...</span>
          </>
        ) : (
          <>
            <span>RUN SATQUERY ANALYSIS</span>
            <ArrowRight className="w-4 h-4" />
          </>
        )}
      </button>

      <div className="text-[10px] font-mono text-[var(--text-secondary)] text-center">
        Autonomous Router &rarr; Specialist VLM &rarr; Deterministic Signal Verification
      </div>
    </div>
  );
};
