import React from 'react';
import type { AnalysisResult, ObservationDomain, ObservationItem, Scenario } from '../../types';
import { ObservationGallery } from '../gallery/ObservationGallery';
import { SatelliteViewer } from './SatelliteViewer';
import { QueryConsole } from './QueryConsole';
import { ScientificReport } from './ScientificReport';

interface ObservationDeckProps {
  domain: ObservationDomain;
  scenarios: Scenario[];
  selectedScenarioIdx: number;
  onSelectScenario: (idx: number) => void;
  selectedItemIdx: number;
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
  onSelectScenario,
  selectedItemIdx,
  onSelectItem,
  query,
  onChangeQuery,
  onExecute,
  isLoading,
  result,
  error,
}) => {
  const currentScenario = scenarios[selectedScenarioIdx];
  const items = currentScenario?.sample_items || [];
  const activeItem: ObservationItem | undefined = items[selectedItemIdx];

  return (
    <section id="observation-workspace" className="max-w-7xl mx-auto px-4 py-8">
      {/* 1. Curated Multi-Image Scenario Observation Selector */}
      <ObservationGallery
        domain={domain}
        scenarios={scenarios}
        selectedScenarioIdx={selectedScenarioIdx}
        onSelectScenario={onSelectScenario}
        selectedItemIdx={selectedItemIdx}
        onSelectItem={onSelectItem}
      />

      {/* 2. Side-by-Side 60/40 Professional Mission Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left Column (58%): Interactive Viewer & Natural Language Console */}
        <div className="lg:col-span-7 flex flex-col justify-between glass-panel p-5">
          <SatelliteViewer
            domain={domain}
            activeItem={activeItem}
            overlayUri={result?.overlay}
            confidenceTag={result?.confidence_tag}
          />
          <QueryConsole
            domain={domain}
            query={query}
            onChangeQuery={onChangeQuery}
            onExecute={onExecute}
            isLoading={isLoading}
          />
        </div>

        {/* Right Column (42%): Structured Scientific Report */}
        <div className="lg:col-span-5 h-full">
          <ScientificReport
            domain={domain}
            activeItem={activeItem}
            result={result}
            error={error}
          />
        </div>
      </div>
    </section>
  );
};
