import React, { useState } from 'react';
import type { Scenario, AnalysisResult, ObservationDomain } from '../../types';
import { ScenarioList } from './ScenarioList';
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
}) => {
  const [layoutMode, setLayoutMode] = useState<WorkspaceLayout>('balanced');
  const currentScenario = scenarios[selectedScenarioIdx] || null;
  const currentItem = currentScenario?.sample_items?.[selectedItemIdx] || null;

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

      {/* 1. Multi-Image Scenario Tile Selector */}
      <ScenarioList
        domain={domain}
        scenarios={scenarios}
        selectedScenarioIdx={selectedScenarioIdx}
        selectedItemIdx={selectedItemIdx}
        onSelectScenario={onSelectScenario}
        onSelectItem={onSelectItem}
      />

      <SectionDivider label="ACTIVE OBSERVATION & SCIENTIFIC TELEMETRY" />

      {/* 2. Responsive Flexible Mission Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start transition-all duration-300">
        {/* Left Column (Visual: Viewer + Query Console) */}
        <div
          className={`space-y-4 transition-all duration-300 ${
            layoutMode === 'expanded-report'
              ? 'lg:col-span-5'
              : layoutMode === 'expanded-viewer'
              ? 'lg:col-span-7'
              : 'lg:col-span-6'
          }`}
        >
          <ImageViewer
            domain={domain}
            item={currentItem}
            analysisResult={result}
            isLoading={isLoading}
          />

          <QueryConsole
            domain={domain}
            query={query}
            onChangeQuery={onChangeQuery}
            onExecute={onExecute}
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
