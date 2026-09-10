import React from 'react';
import type { ObservationDomain } from '../../types';
import { Loader2, ArrowRight, Terminal, Sparkles, X } from 'lucide-react';

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
    { label: '+ Find Water Bodies', q: 'Identify flooded water bodies, rivers, and coastal drainage.' },
    { label: '+ Describe Scene', q: 'Describe the land use and dominant terrain features in this scene.' },
    { label: '+ Detect Changes', q: 'What changed between these observation dates? Has new construction occurred?' },
    { label: '+ Spectral Vegetation', q: 'Assess vegetation density and compute physical spectral indices.' },
    { label: '+ Identify Roads', q: 'Identify major transportation roads, highways, and logistics corridors.' },
  ];

  const lunarChips = [
    { label: '+ Find Craters', q: 'Detect prominent impact crater candidates across the lunar scene.' },
    { label: '+ Crater Morphology', q: 'Analyze crater rim morphology, slope gradients, and rim degradation.' },
    { label: '+ Analyze Regolith', q: 'Describe the lunar surface morphology and regolith micro-relief texture.' },
    { label: '+ Shadow & PSRs', q: 'Identify deep shadow zones and potential permanently shadowed regions (PSR).' },
    { label: '+ Ejecta Blankets', q: 'Identify high-albedo ejecta blanket patterns and radial impact rays.' },
    { label: '+ Compare Terrains', q: 'Compare surface roughness, crater density, and geological formations.' },
  ];

  const chips = isLunar ? lunarChips : earthChips;

  return (
    <div className="w-full rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-panel)] p-4 sm:p-5 shadow-xl space-y-4">
      {/* Console Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-1 border-b border-[var(--border-hairline)]">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded flex items-center justify-center bg-[rgba(77,184,255,0.12)] text-[var(--accent-verified)]">
            <Terminal className="w-3.5 h-3.5" />
          </div>
          <span className="font-mono text-xs font-bold tracking-wider text-[var(--text-primary)] uppercase flex items-center gap-2">
            MISSION QUERY CONSOLE
            <span className="hidden sm:inline-block text-[10px] px-2 py-0.5 rounded bg-[rgba(77,184,255,0.1)] text-[var(--accent-verified)] border border-[rgba(77,184,255,0.25)]">
              CUSTOM USER INPUT
            </span>
          </span>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono text-[var(--text-secondary)]">
          <span className="flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-[var(--accent-verified)]" />
            <span>Type any custom question or click preset</span>
          </span>
          {query.trim() && (
            <button
              type="button"
              onClick={() => onChangeQuery('')}
              className="flex items-center gap-1 text-[var(--text-secondary)] hover:text-[var(--accent-conflict)] transition-colors cursor-pointer"
              title="Clear input text"
            >
              <X className="w-3 h-3" />
              <span>Clear</span>
            </button>
          )}
        </div>
      </div>

      {/* Contextual Command Presets */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-[10px] tracking-wider text-[var(--text-secondary)] uppercase">
            QUICK PRESETS ({isLunar ? 'LUNAR SCIENTIFIC PROMPTS' : 'EARTH OBSERVATION PROMPTS'}):
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {chips.map((chip, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => onChangeQuery(chip.q)}
              className="text-left font-sans text-xs bg-[var(--bg-panel-elevated)] hover:bg-[rgba(77,184,255,0.1)] border border-[var(--border-hairline)] hover:border-[var(--accent-verified)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] px-2.5 py-1.5 rounded transition-all truncate focus:outline-none focus:ring-1 focus:ring-[var(--accent-verified)] cursor-pointer"
              title={chip.q}
            >
              {chip.label}
            </button>
          ))}
        </div>
      </div>

      {/* Textarea + Execution Controls Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 items-stretch">
        {/* Natural Language Query Textarea */}
        <div className="lg:col-span-9 relative">
          <textarea
            value={query}
            onChange={(e) => onChangeQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                if (!isLoading && query.trim()) onExecute();
              }
            }}
            placeholder={
              isLunar
                ? 'Type custom lunar question (e.g. Detect impact craters, quantify ejecta blanket distribution, or measure rim degradation)...'
                : 'Type custom remote sensing question (e.g. Locate industrial buildings, quantify flood extent, or compare NDVI spectral vegetation)...'
            }
            rows={3}
            className="w-full h-full min-h-[88px] bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] focus:border-[var(--accent-verified)] rounded-lg p-3 text-sm text-[var(--text-primary)] placeholder-[var(--text-secondary)] focus:outline-none font-sans resize-none transition-all focus:ring-1 focus:ring-[var(--accent-verified)] shadow-inner"
          />
          <div className="absolute bottom-2 right-2 text-[10px] font-mono text-[var(--text-secondary)] opacity-60 pointer-events-none hidden sm:block">
            Ctrl+Enter to Run
          </div>
        </div>

        {/* Action Trigger Button & Telemetry */}
        <div className="lg:col-span-3 flex flex-col justify-between gap-2">
          <button
            onClick={onExecute}
            disabled={isLoading || !query.trim()}
            className={`w-full h-full min-h-[50px] py-3 px-4 rounded-lg font-sans font-bold text-xs tracking-wider uppercase flex items-center justify-center gap-2 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-[var(--accent-verified)] ${
              isLoading || !query.trim()
                ? 'bg-[var(--bg-panel-elevated)] text-[var(--text-secondary)] border border-[var(--border-hairline)] cursor-not-allowed opacity-50'
                : 'bg-[var(--accent-verified)] text-[var(--bg-void)] hover:bg-[#68c6ff] shadow-[0_0_25px_rgba(77,184,255,0.4)] hover:shadow-[0_0_35px_rgba(77,184,255,0.6)] cursor-pointer active:scale-[0.98]'
            }`}
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-[var(--bg-void)]" />
                <span>ANALYZING...</span>
              </>
            ) : (
              <>
                <span>RUN SATQUERY</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
          <div className="text-[10px] font-mono text-[var(--text-secondary)] text-center">
            Autonomous Router &middot; Specialist VLM &middot; Physics Verification
          </div>
        </div>
      </div>
    </div>
  );
};
