import React, { useState } from 'react';
import type { Scenario, AnalysisResult, ObservationDomain, ObservationItem } from '../../types';
import { ScenarioList } from './ScenarioList';
import { CustomUploadZone } from './CustomUploadZone';
import { ImageViewer } from './ImageViewer';
import { QueryConsole } from './QueryConsole';
import { ReportPanel } from './ReportPanel';
import { SectionDivider } from '../layout/SectionDivider';
import { Columns, Maximize2, Split } from 'lucide-react';

interface ObservationDeckProps {
  domain: ObservationDomain;
  scenarios: Scenario[];
  selectedScenarioIdx: number;
  selectedItemIdx: number;
  onSelectScenario: (idx: number) => void;
  onSelectItem: (idx: number) => void;
  query: string;
  onChangeQuery: (query: string) => void;
  onExecute: () => void;
  isLoading: boolean;
  result: AnalysisResult | null;
  error: string | null;
  sourceMode: 'archival' | 'custom';
  onChangeSourceMode: (mode: 'archival' | 'custom') => void;
  customItems: ObservationItem[];
  onUploadSuccess: (items: ObservationItem[]) => void;
  onClearCustomUploads: () => void;
  selectedCustomIdx: number;
  onSelectCustomItem: (idx: number) => void;
}

export type WorkspaceLayout = 'balanced' | 'expanded-report' | 'expanded-viewer';

export const ObservationDeck: React.FC<ObservationDeckProps> = ({
  domain,
  scenarios,
  selectedScenarioIdx,
  selectedItemIdx,
  onSelectScenario,
  onSelectItem,
  query,
  onChangeQuery,
  onExecute,
  isLoading,
  result,
  error,
  sourceMode,
  onChangeSourceMode,
  customItems,
  onUploadSuccess,
  onClearCustomUploads,
  selectedCustomIdx,
  onSelectCustomItem,
}) => {
  const [layoutMode, setLayoutMode] = useState<WorkspaceLayout>('balanced');
  const currentScenario = scenarios[selectedScenarioIdx] || null;
  const currentItem =
    sourceMode === 'custom'
      ? customItems[selectedCustomIdx] || null
      : currentScenario?.sample_items?.[selectedItemIdx] || null;

  // Toggle report expansion
  const toggleReportExpand = () => {
    setLayoutMode((prev) => (prev === 'expanded-report' ? 'balanced' : 'expanded-report'));
  };

  return (
    <section id="observation-workspace" className="w-full max-w-[1680px] mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Workspace Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2 h-2 rounded-full bg-[var(--accent-verified)]" />
            <span className="font-mono text-xs tracking-widest text-[var(--accent-verified)] uppercase">
              ISRO / SAC MISSION WORKSTATION · PS 26167
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-serif font-bold text-[var(--text-primary)]">
            {domain === 'lunar' ? 'Lunar Surface Exploration Deck' : 'Multimodal Earth Observation Deck'}
          </h2>
        </div>

        {/* Dynamic Workspace Layout Selector */}
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] text-[var(--text-secondary)] uppercase tracking-wider hidden sm:inline">
            LAYOUT RATIO:
          </span>
          <div className="flex items-center rounded border border-[var(--border-hairline)] bg-[var(--bg-panel-elevated)] p-1 gap-1">
            <button
              onClick={() => setLayoutMode('balanced')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded text-[10px] font-mono font-bold transition-all cursor-pointer ${
                layoutMode === 'balanced'
                  ? 'bg-[var(--accent-verified)] text-[var(--bg-void)]'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              title="Balanced 50/50 split"
            >
              <Split className="w-3 h-3" />
              <span>50 / 50</span>
            </button>

            <button
              onClick={() => setLayoutMode('expanded-report')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded text-[10px] font-mono font-bold transition-all cursor-pointer ${
                layoutMode === 'expanded-report'
                  ? 'bg-[var(--accent-verified)] text-[var(--bg-void)]'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              title="Expand report to 65% width"
            >
              <Maximize2 className="w-3 h-3" />
              <span>WIDE REPORT</span>
            </button>

            <button
              onClick={() => setLayoutMode('expanded-viewer')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded text-[10px] font-mono font-bold transition-all cursor-pointer ${
                layoutMode === 'expanded-viewer'
                  ? 'bg-[var(--accent-verified)] text-[var(--bg-void)]'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              title="Expand raster viewer to 65% width"
            >
              <Columns className="w-3 h-3" />
              <span>WIDE VIEWER</span>
            </button>
          </div>
        </div>
      </div>

      {/* Source Selection Mode Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[var(--border-hairline)] pb-4">
        <div className="flex items-center rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-panel-elevated)] p-1 gap-1">
          <button
            onClick={() => onChangeSourceMode('archival')}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-xs font-mono font-bold transition-all cursor-pointer ${
              sourceMode === 'archival'
                ? 'bg-[var(--accent-verified)] text-[var(--bg-void)] shadow-[0_0_15px_rgba(77,184,255,0.25)]'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-panel)]'
            }`}
          >
            <span>🛰️ Archival Mission Scenes</span>
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
              sourceMode === 'archival' ? 'bg-[rgba(0,0,0,0.3)] text-[var(--bg-void)]' : 'bg-[var(--bg-panel)] text-[var(--text-secondary)]'
            }`}>
              {scenarios.length}
            </span>
          </button>

          <button
            onClick={() => onChangeSourceMode('custom')}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-xs font-mono font-bold transition-all cursor-pointer ${
              sourceMode === 'custom'
                ? 'bg-[var(--accent-verified)] text-[var(--bg-void)] shadow-[0_0_15px_rgba(77,184,255,0.25)]'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-panel)]'
            }`}
          >
            <span>📤 Upload Custom Rasters</span>
            {customItems.length > 0 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-void)] text-[var(--accent-verified)] font-bold">
                {customItems.length} ACTIVE
              </span>
            )}
          </button>
        </div>

        <div className="text-[11px] font-mono text-[var(--text-secondary)]">
          {sourceMode === 'archival'
            ? 'Curated ISRO / SAC benchmark scenes & multimodal telemetry'
            : 'Direct GeoTIFF, PNG, or JPEG raster ingestion (Optical, SAR, or Lunar TMC-2)'}
        </div>
      </div>

      {/* 1. Observation Source Panel */}
      {sourceMode === 'archival' ? (
        <ScenarioList
          domain={domain}
          scenarios={scenarios}
          selectedScenarioIdx={selectedScenarioIdx}
          selectedItemIdx={selectedItemIdx}
          onSelectScenario={onSelectScenario}
          onSelectItem={onSelectItem}
        />
      ) : (
        <CustomUploadZone
          domain={domain}
          onUploadSuccess={onUploadSuccess}
          uploadedItems={customItems}
          onClearUploads={onClearCustomUploads}
        />
      )}

      {/* 2. Interactive Mission Query Console */}
      <QueryConsole
        domain={domain}
        query={query}
        onChangeQuery={onChangeQuery}
        onExecute={onExecute}
        isLoading={isLoading}
      />

      <SectionDivider label="ACTIVE OBSERVATION & SCIENTIFIC TELEMETRY" />

      {/* 3. Responsive Flexible Mission Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start transition-all duration-300">
        {/* Left Column (Visual: High-Resolution Raster Viewer) */}
        <div
          className={`space-y-3 transition-all duration-300 ${
            layoutMode === 'expanded-report'
              ? 'lg:col-span-5'
              : layoutMode === 'expanded-viewer'
              ? 'lg:col-span-7'
              : 'lg:col-span-6'
          }`}
        >
          {sourceMode === 'custom' && customItems.length > 1 && (
            <div className="flex flex-wrap items-center gap-2 p-2.5 rounded-lg bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)]">
              <span className="font-mono text-[10px] text-[var(--accent-verified)] uppercase tracking-wider">
                INSPECT RASTER:
              </span>
              {customItems.map((item, idx) => (
                <button
                  key={item.id || idx}
                  onClick={() => onSelectCustomItem(idx)}
                  className={`px-2.5 py-1 rounded text-xs font-mono transition-all cursor-pointer ${
                    selectedCustomIdx === idx
                      ? 'bg-[var(--accent-verified)] text-[var(--bg-void)] font-bold shadow-[0_0_10px_rgba(77,184,255,0.3)]'
                      : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-panel)]'
                  }`}
                >
                  {idx === 0 ? 'Image 1 (Primary)' : 'Image 2 (Secondary)'}: {item.name || item.filename}
                </button>
              ))}
            </div>
          )}

          <ImageViewer
            domain={domain}
            item={currentItem}
            analysisResult={result}
            isLoading={isLoading}
          />
        </div>

        {/* Right Column (Information: Structured Scientific Analysis Report) */}
        <div
          className={`h-full transition-all duration-300 ${
            layoutMode === 'expanded-report'
              ? 'lg:col-span-7'
              : layoutMode === 'expanded-viewer'
              ? 'lg:col-span-5'
              : 'lg:col-span-6'
          }`}
        >
          <ReportPanel
            domain={domain}
            result={result}
            item={currentItem}
            isLoading={isLoading}
            error={error}
            isExpanded={layoutMode === 'expanded-report'}
            onToggleExpand={toggleReportExpand}
          />
        </div>
      </div>
    </section>
  );
};
