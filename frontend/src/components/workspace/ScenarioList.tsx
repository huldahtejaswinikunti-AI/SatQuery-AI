import React from 'react';
import type { Scenario, ObservationDomain } from '../../types';
import { getImageUrl } from '../../api/client';

interface ScenarioListProps {
  domain: ObservationDomain;
  scenarios: Scenario[];
  selectedScenarioIdx: number;
  selectedItemIdx: number;
  onSelectScenario: (idx: number) => void;
  onSelectItem: (idx: number) => void;
}

export const ScenarioList: React.FC<ScenarioListProps> = ({
  domain,
  scenarios,
  selectedScenarioIdx,
  selectedItemIdx,
  onSelectScenario,
  onSelectItem,
}) => {
  if (scenarios.length === 0) {
    return (
      <div className="p-6 rounded border border-[var(--border-hairline)] bg-[var(--bg-panel)] text-center">
        <p className="font-sans text-sm text-[var(--text-secondary)]">
          Select a scenario to begin mission observation.
        </p>
      </div>
    );
  }

  const currentScenario = scenarios[selectedScenarioIdx] || scenarios[0];
  const items = currentScenario?.sample_items || [];

  return (
    <div className="w-full space-y-4">
      {/* Scenario Header & Selection Row */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[var(--border-hairline)] pb-3">
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs tracking-widest text-[var(--accent-verified)] uppercase">
            ACTIVE MISSION
          </span>
          <select
            value={selectedScenarioIdx}
            onChange={(e) => onSelectScenario(Number(e.target.value))}
            className="bg-[var(--bg-panel)] text-[var(--text-primary)] border border-[var(--border-hairline)] rounded px-3 py-1.5 font-serif text-sm focus:outline-none focus:ring-1 focus:ring-[var(--accent-verified)]"
          >
            {scenarios.map((s, idx) => (
              <option key={idx} value={idx} className="bg-[var(--bg-panel)]">
                {s.label} ({s.task.toUpperCase()})
              </option>
            ))}
          </select>
        </div>

        {currentScenario && (
          <div className="flex items-center gap-4 text-xs text-[var(--text-secondary)]">
            <span className="font-mono">
              TASK: <span className="text-[var(--text-primary)] font-bold">{currentScenario.task.toUpperCase()}</span>
            </span>
            <span className="font-mono">
              FRAMES: <span className="text-[var(--text-primary)] font-bold">{items.length} ARCHIVAL SCENES</span>
            </span>
          </div>
        )}
      </div>

      {/* Multi-Image Observation Strip */}
      {items.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="font-mono text-[11px] tracking-wider text-[var(--text-secondary)] uppercase">
              OBSERVATION SCENE TILES ({items.length} ARCHIVAL FRAMES)
            </span>
            <span className="font-sans text-[11px] text-[var(--text-secondary)]">
              Click frame to load in high-resolution viewer
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {items.map((item, idx) => {
              const isSelected = selectedItemIdx === idx;
              const imgPath = item.file || (item.files && item.files[0]) || '';
              const imgUrl = imgPath ? getImageUrl(imgPath, domain) : '';

              return (
                <button
                  key={idx}
                  onClick={() => onSelectItem(idx)}
                  className={`group relative text-left rounded overflow-hidden border transition-all duration-200 bg-[var(--bg-panel-elevated)] focus:outline-none focus:ring-1 focus:ring-[var(--accent-verified)] ${
                    isSelected
                      ? 'border-[var(--accent-verified)] shadow-[0_0_15px_rgba(77,184,255,0.25)]'
                      : 'border-[var(--border-hairline)] hover:border-[rgba(255,255,255,0.2)] opacity-80 hover:opacity-100'
                  }`}
                >
                  {/* Thumbnail Raster (never white while loading) */}
                  <div className="aspect-video w-full overflow-hidden bg-[var(--bg-panel-elevated)] relative">
                    {imgUrl ? (
                      <img
                        src={imgUrl}
                        alt={item.name}
                        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                        loading="lazy"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-[var(--bg-panel-elevated)] text-[var(--text-secondary)] text-[10px] font-mono">
                        N/A
                      </div>
                    )}

                    {isSelected && (
                      <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-[var(--accent-verified)]" />
                    )}
                  </div>

                  {/* Tile Metadata */}
                  <div className="p-2 space-y-1">
                    <p className="font-serif text-xs font-semibold text-[var(--text-primary)] truncate">
                      {item.name}
                    </p>
                    <div className="flex items-center justify-between font-mono text-[9px] text-[var(--text-secondary)]">
                      <span>{item.sensor || 'SAT'}</span>
                      <span>{item.resolution || '0.5m'}</span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
