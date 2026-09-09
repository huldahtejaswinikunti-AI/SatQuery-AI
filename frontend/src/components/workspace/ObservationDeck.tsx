import React from 'react';
import type { Scenario, AnalysisResult, ObservationDomain } from '../../types';
import { ScenarioList } from './ScenarioList';
import { ImageViewer } from './ImageViewer';
import { QueryConsole } from './QueryConsole';
import { ReportPanel } from './ReportPanel';
import { SectionDivider } from '../layout/SectionDivider';

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
  const currentScenario = scenarios[selectedScenarioIdx] || null;
  const currentItem = currentScenario?.sample_items?.[selectedItemIdx] || null;

  return (
    <section id="observation-workspace" className="w-full max-w-7xl mx-auto px-4 sm:px-6 py-10 space-y-8">
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

        <div className="font-mono text-xs text-[var(--text-secondary)]">
          WORKSPACE RATIO: <span className="text-[var(--text-primary)] font-bold">55% VISUAL / 45% TELEMETRY</span>
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

      {/* 2. Balanced 55/45 Side-by-Side Mission Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column (55% Visual: Viewer + Query Console) */}
        <div className="lg:col-span-7 space-y-4">
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

        {/* Right Column (45% Information: Structured Scientific Analysis Report) */}
        <div className="lg:col-span-5 h-full">
          <ReportPanel
            domain={domain}
            result={result}
            item={currentItem}
            isLoading={isLoading}
            error={error}
          />
        </div>
      </div>
    </section>
  );
};
