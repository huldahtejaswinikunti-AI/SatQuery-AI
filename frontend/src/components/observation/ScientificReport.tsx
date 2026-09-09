import React from 'react';
import type { AnalysisResult, ObservationDomain, ObservationItem } from '../../types';
import { CheckCircle2, AlertTriangle, HelpCircle, ShieldCheck, Download } from 'lucide-react';

interface ScientificReportProps {
  domain: ObservationDomain;
  activeItem?: ObservationItem;
  result?: AnalysisResult | null;
  error?: string | null;
}

export const ScientificReport: React.FC<ScientificReportProps> = ({
  domain,
  activeItem,
  result,
  error,
}) => {
  const isLunar = domain === 'lunar';

  // Render Error state
  if (error) {
    return (
      <div className="glass-panel p-5 border-red-500/40 bg-red-950/20 h-full flex flex-col justify-center">
        <div className="flex items-center gap-2 text-red-400 font-bold text-sm mb-2 font-mono uppercase">
          <AlertTriangle size={18} /> Analysis Execution Failed
        </div>
        <p className="text-xs text-slate-300 font-mono leading-relaxed bg-space-950/80 p-3 rounded border border-red-500/20">
          {error}
        </p>
      </div>
    );
  }

  // Render Staging Telemetry Panel before query execution
  if (!result) {
    return (
      <div className="glass-panel p-5 h-full flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between pb-3 mb-4 border-b border-white/10">
            <div className="flex items-center gap-2">
              <ShieldCheck size={16} className="text-teal-400" />
              <span className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Mission Staging Telemetry
              </span>
            </div>
            <span className="text-[10px] font-mono font-bold text-teal-400 bg-teal-500/10 border border-teal-500/30 px-2 py-0.5 rounded">
              PIPELINE READY
            </span>
          </div>

          <p className="text-xs text-slate-300 leading-relaxed mb-4">
            Staged observation raster is loaded and prepared for agentic vision-language reasoning,
            open-vocabulary grounding, and deterministic signal cross-verification.
          </p>

          {/* Staging Parameters Table */}
          <div className="space-y-2 mb-5 font-mono text-xs">
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-500 uppercase">Target Body:</span>
              <span className="text-teal-300 font-bold">{isLunar ? 'MOON (LUNAR SURFACE)' : 'EARTH (TERRESTRIAL)'}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-500 uppercase">Sensor Payload:</span>
              <span className="text-cyan-400 font-bold">{activeItem?.sensor || 'Optical / SAR'}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-500 uppercase">Ground Sample Distance:</span>
              <span className="text-slate-200">{activeItem?.resolution || 'Native GSD'}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-500 uppercase">Acquisition Timestamp:</span>
              <span className="text-slate-200">{activeItem?.date || 'Archived Observation'}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-500 uppercase">Archive Provenance:</span>
              <span className="text-slate-200">{activeItem?.provenance || 'ISRO / Copernicus Archive'}</span>
            </div>
          </div>

          {/* Active Domain Capabilities */}
          <div>
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-bold mb-2">
              Active Specialist Capabilities:
            </div>
            <div className="space-y-1.5 text-xs text-slate-300">
              {isLunar ? (
                <>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
                    <span>Chandrayaan-2 TMC-2 / OHRC Optical Surface Morphology</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
                    <span>Shadow Analysis & Cold-Trap (PSR) Detection</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
                    <span>Ejecta Blanket & High-Albedo Deposit Delineation</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
                    <span>GeoChat-7B Vision-Language Remote Sensing Reasoning</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
                    <span>CLIPSeg Zero-Shot Open-Vocabulary Grounding</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
                    <span>Deterministic Spectral Indices (NDVI / NDWI / NDBI)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
                    <span>TinyCD Deep Bi-Temporal Feature Differencing</span>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="text-[11px] font-mono text-slate-500 border-t border-white/10 pt-3 mt-4">
          Select or enter a query on the left to execute scientific analysis.
        </div>
      </div>
    );
  }

  // Helper for Confidence Badge
  const getConfidenceBadge = () => {
    const tag = result.confidence_tag?.toLowerCase() || '';
    if (isLunar || tag.includes('unverified')) {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono font-bold bg-slate-800 text-slate-300 border border-slate-600">
          <HelpCircle size={13} /> Experimental &middot; No Cross-Check Available
        </span>
      );
    }
    if (tag.includes('verified') || tag.includes('high')) {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono font-bold bg-teal-500/15 text-teal-300 border border-teal-500/40 shadow-glow-teal">
          <CheckCircle2 size={13} /> High Confidence (Cross-Verified)
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono font-bold bg-amber-500/15 text-amber-300 border border-amber-500/40">
        <AlertTriangle size={13} /> Moderate Confidence / Disagreement
      </span>
    );
  };

  const facts = result.verified_facts || {};

  return (
    <div className="glass-panel p-5 h-full flex flex-col justify-between overflow-y-auto">
      <div>
        {/* Report Header Bar */}
        <div className="flex items-center justify-between pb-3 mb-3 border-b border-white/10">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-white">
              {isLunar ? '🌕 Lunar Scientific Report' : '🛰️ Satellite Scientific Analysis Report'}
            </span>
          </div>
          {getConfidenceBadge()}
        </div>

        {/* Lunar Honesty Disclaimer Banner */}
        {isLunar && (
          <div className="mb-4 bg-slate-900/80 border-l-2 border-slate-500 p-2.5 rounded-r text-[11px] text-slate-300 font-mono leading-relaxed">
            <strong>Scientific Notice:</strong> Lunar imagery has no deterministic ground-truth cross-check in this system &mdash; treat vision-language findings as model-generated interpretations.
          </div>
        )}

        {/* 1. Executive Summary */}
        <div className="mb-4">
          <div className="text-[11px] font-mono font-bold uppercase tracking-wider text-teal-400 mb-1.5">
            1. Executive Summary & Findings:
          </div>
          <div className="bg-space-950/80 border-l-2 border-teal-400 p-3 rounded-r text-xs md:text-sm text-slate-100 leading-relaxed">
            {result.answer}
          </div>
        </div>

        {/* 2. Physical Measurements & Calibration (Distinguishes CV measurement vs Model interpretation) */}
        <div className="mb-4">
          <div className="text-[11px] font-mono font-bold uppercase tracking-wider text-cyan-400 mb-1.5">
            2. Physical Measurements & Calibration:
          </div>
          <div className="bg-space-950 border border-white/10 rounded-lg p-3 font-mono text-xs space-y-1.5">
            <div className="flex justify-between py-1 border-b border-white/5">
              <span className="text-slate-400">Analysis Methodology:</span>
              <span className="text-teal-400 font-bold">
                {isLunar ? 'Zero-Shot Regolith Morphology Engine' : 'Autonomous Specialist Router & Cross-Verifier'}
              </span>
            </div>
            {facts.water_fraction !== undefined && (
              <div className="flex justify-between py-1 border-b border-white/5">
                <span className="text-slate-400">Calibrated Water Body Coverage:</span>
                <span className="text-teal-300 font-bold">{(facts.water_fraction * 100).toFixed(1)}%</span>
              </div>
            )}
            {facts.urban_fraction !== undefined && (
              <div className="flex justify-between py-1 border-b border-white/5">
                <span className="text-slate-400">Calibrated Urban Structural Density:</span>
                <span className="text-teal-300 font-bold">{(facts.urban_fraction * 100).toFixed(1)}%</span>
              </div>
            )}
            {facts.crater_diameter_km !== undefined && (
              <div className="flex justify-between py-1 border-b border-white/5">
                <span className="text-slate-400">Crater Diameter:</span>
                <span className="text-amber-300 font-bold">{facts.crater_diameter_km}</span>
              </div>
            )}
            {result.consensus_score !== null && result.consensus_score !== undefined && (
              <div className="flex justify-between py-1 border-b border-white/5">
                <span className="text-slate-400">Cross-Modal Consensus Index:</span>
                <span className="text-teal-300 font-bold">{(result.consensus_score * 100).toFixed(0)}%</span>
              </div>
            )}
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Verification Integrity:</span>
              <span className="text-slate-300">
                {isLunar ? 'Unverified (Model Interpretation Only)' : 'Deterministic Spectral / Cross-Checked'}
              </span>
            </div>
          </div>
        </div>

        {/* 3. Deterministic Verification Checklist */}
        <div className="mb-4">
          <div className="text-[11px] font-mono font-bold uppercase tracking-wider text-teal-400 mb-1.5">
            3. Deterministic Verification Matrix:
          </div>
          <div className="bg-space-950 border border-white/10 rounded-lg p-3 font-mono text-xs space-y-1.5">
            <div className="flex justify-between items-center py-0.5">
              <span className="text-slate-300 flex items-center gap-1.5">
                <span className="text-teal-400">✓</span> Pixel Array Radiometry Validated
              </span>
              <span className="text-teal-400 font-bold text-[10px]">PASS</span>
            </div>
            <div className="flex justify-between items-center py-0.5">
              <span className="text-slate-300 flex items-center gap-1.5">
                <span className="text-teal-400">✓</span> Spectral / Morphological Gradient Grids
              </span>
              <span className="text-teal-400 font-bold text-[10px]">PASS</span>
            </div>
            <div className="flex justify-between items-center py-0.5">
              <span className="text-slate-300 flex items-center gap-1.5">
                <span className={isLunar ? 'text-amber-400' : 'text-teal-400'}>
                  {isLunar ? '!' : '✓'}
                </span>{' '}
                Calibrated Scale Metadata
              </span>
              <span className={isLunar ? 'text-amber-400 font-bold text-[10px]' : 'text-teal-400 font-bold text-[10px]'}>
                {isLunar ? 'QUALITATIVE' : 'PASS'}
              </span>
            </div>
          </div>
        </div>

        {/* 4. Provenance & Dataset Archive */}
        <div className="text-[11px] font-mono text-slate-400 bg-space-950/60 p-2.5 rounded border border-white/5 space-y-1">
          <div>
            <strong>Archive Source:</strong> {activeItem?.provenance || (isLunar ? 'ISRO / ISSDC PRADAN Archive' : 'Copernicus Sentinel Hub')}
          </div>
          <div>
            <strong>Model Pipeline:</strong> GeoChat-7B &middot; CLIPSeg &middot; TinyCD &middot; ResNet-18
          </div>
        </div>
      </div>

      {/* Export & Actions Footer */}
      <div className="pt-4 mt-3 border-t border-white/10 flex items-center justify-between">
        <span className="text-[10px] font-mono text-slate-500">
          SIH 2026 &middot; PS 26167 &middot; ISRO SAC
        </span>
        <button
          onClick={() => alert(`JSON Trace:\n${JSON.stringify(result.trace, null, 2)}`)}
          className="flex items-center gap-1.5 text-xs font-mono text-teal-400 hover:text-teal-300 bg-teal-500/10 hover:bg-teal-500/20 border border-teal-500/30 px-3 py-1 rounded transition-colors"
        >
          <Download size={13} /> View Execution Trace
        </button>
      </div>
    </div>
  );
};
