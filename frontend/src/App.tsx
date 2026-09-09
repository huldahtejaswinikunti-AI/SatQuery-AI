import React, { useEffect, useState, useCallback } from 'react';
import type { AnalysisResult, ObservationDomain, Scenario } from './types';
import { fetchScenarios, runAnalysis } from './api/client';
import { PageBackground } from './components/layout/PageBackground';
import { Nav } from './components/layout/Nav';
import { PlanetaryHero } from './components/space/PlanetaryHero';
import { ObservationDeck } from './components/workspace/ObservationDeck';

export const App: React.FC = () => {
  const [domain, setDomain] = useState<ObservationDomain>('earth');
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selectedScenarioIdx, setSelectedScenarioIdx] = useState(0);
  const [selectedItemIdx, setSelectedItemIdx] = useState(0);
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load scenarios whenever domain changes
  useEffect(() => {
    let isMounted = true;
    fetchScenarios(domain)
      .then((scens) => {
        if (!isMounted) return;
        setScenarios(scens);
        setSelectedScenarioIdx(0);
        setSelectedItemIdx(0);
        setResult(null);
        setError(null);
        if (scens.length > 0) {
          const firstItem = scens[0]?.sample_items?.[0];
          setQuery(firstItem?.query || scens[0]?.query || '');
        }
      })
      .catch((err) => {
        if (isMounted) setError(`Failed to load mission scenarios: ${err.message}`);
      });
    return () => {
      isMounted = false;
    };
  }, [domain]);

  // Sync query when scenario or selected image changes
  useEffect(() => {
    const scen = scenarios[selectedScenarioIdx];
    const item = scen?.sample_items?.[selectedItemIdx];
    if (item?.query) {
      setQuery(item.query);
    } else if (scen?.query) {
      setQuery(scen.query);
    }
  }, [scenarios, selectedScenarioIdx, selectedItemIdx]);

  // Execute Analysis
  const handleExecute = useCallback(async () => {
    const scen = scenarios[selectedScenarioIdx];
    const item = scen?.sample_items?.[selectedItemIdx];
    const files = item?.files || (item?.file ? [item.file] : scen?.sample_files || scen?.files || []);

    if (files.length === 0) {
      setError('No observation file available to analyze for this scene.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const res = await runAnalysis(domain, files, query);
      setResult(res);
      if (res.confidence_tag === 'error') {
        setError(res.validation_failure_reason || res.answer || 'Analysis pipeline returned an error.');
      }
    } catch (err: any) {
      setError(err.message || 'Execution failed. Please retry.');
    } finally {
      setIsLoading(false);
    }
  }, [domain, scenarios, selectedScenarioIdx, selectedItemIdx, query]);

  return (
    <PageBackground>
      {/* 1. Floating Top Navigation Bar */}
      <Nav
        activeDomain={domain}
        onSelectDomain={(newDomain) => {
          setDomain(newDomain);
          setResult(null);
          setError(null);
        }}
      />

      {/* 2. Full-Viewport 3D Multi-Map Planetary Hero */}
      <PlanetaryHero
        activeDomain={domain}
        onSelectDomain={(newDomain) => {
          setDomain(newDomain);
          setResult(null);
          setError(null);
        }}
      />

      {/* 3. Balanced 55/45 Observation Deck Workstation */}
      <main className="relative z-10">
        <ObservationDeck
          domain={domain}
          scenarios={scenarios}
          selectedScenarioIdx={selectedScenarioIdx}
          onSelectScenario={(idx) => {
            setSelectedScenarioIdx(idx);
            setSelectedItemIdx(0);
            setResult(null);
            setError(null);
          }}
          selectedItemIdx={selectedItemIdx}
          onSelectItem={(idx) => {
            setSelectedItemIdx(idx);
            setResult(null);
            setError(null);
          }}
          query={query}
          onChangeQuery={setQuery}
          onExecute={handleExecute}
          isLoading={isLoading}
          result={result}
          error={error}
        />
      </main>

      {/* 4. Footer */}
      <footer className="border-t border-[var(--border-hairline)] bg-[var(--bg-void)] py-6 px-6 text-center text-xs text-[var(--text-secondary)] font-mono">
        SatQuery AI &middot; Interactive Vision-Language Assistant for Multimodal Remote Sensing &middot; SIH 2026 PS 26167 &middot; ISRO / SAC
      </footer>
    </PageBackground>
  );
};
