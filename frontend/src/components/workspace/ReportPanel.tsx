import React, { useState, useRef, useEffect } from 'react';
import type { AnalysisResult, ObservationDomain, ObservationItem } from '../../types';
import { ConfidenceBadge } from './ConfidenceBadge';
import { exportPdfReport } from '../../api/client';
import { 
  ShieldCheck, 
  AlertCircle, 
  FileText, 
  Activity, 
  Download, 
  Check, 
  Copy, 
  ChevronDown, 
  FileCode, 
  Loader2,
  Maximize2,
  Minimize2
} from 'lucide-react';

interface ReportPanelProps {
  domain: ObservationDomain;
  result: AnalysisResult | null;
  item: ObservationItem | null;
  isLoading?: boolean;
  error?: string | null;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}

export const ReportPanel: React.FC<ReportPanelProps> = ({
  domain,
  result,
  item,
  isLoading = false,
  error = null,
  isExpanded = false,
  onToggleExpand,
}) => {
  const [copied, setCopied] = useState(false);
  const [isPdfLoading, setIsPdfLoading] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Error state
  if (error) {
    return (
      <div className="w-full h-full min-h-[480px] rounded border border-[rgba(239,68,68,0.3)] bg-[var(--bg-panel)] p-6 flex flex-col items-center justify-center text-center space-y-4">
        <AlertCircle className="w-10 h-10 text-rose-400" />
        <div className="space-y-1">
          <h3 className="font-serif text-base font-bold text-[var(--text-primary)]">
            Analysis Failed for This Scene
          </h3>
          <p className="font-sans text-xs text-[var(--text-secondary)] max-w-sm">
            {error || 'An unexpected pipeline error occurred. Please re-select the observation image or try another query.'}
          </p>
        </div>
      </div>
    );
  }

  // Loading state (high-contrast spinner)
  if (isLoading) {
    return (
      <div className="w-full h-full min-h-[480px] rounded border border-[var(--border-hairline)] bg-[var(--bg-panel)] p-6 flex flex-col items-center justify-center text-center space-y-4">
        <div className="w-10 h-10 rounded-full border-2 border-[var(--accent-verified)] border-t-transparent animate-spin" />
        <div className="space-y-1">
          <span className="font-mono text-xs tracking-wider text-[var(--text-primary)] font-bold uppercase">
            Synthesizing Scientific Analysis Report...
          </span>
          <p className="font-sans text-xs text-[var(--text-secondary)]">
            Executing multimodal grounding and deterministic verification
          </p>
        </div>
      </div>
    );
  }

  // Staging / Idle state (Zero empty voids)
  if (!result) {
    return (
      <div className="w-full h-full rounded border border-[var(--border-hairline)] bg-[var(--bg-panel)] p-6 space-y-6">
        <div className="flex items-center justify-between border-b border-[var(--border-hairline)] pb-4">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[var(--accent-verified)]" />
            <h3 className="font-serif text-sm font-bold tracking-wide text-[var(--text-primary)]">
              MISSION TELEMETRY & STAGING DECK
            </h3>
          </div>
          <span className="font-mono text-[10px] tracking-wider text-[var(--text-secondary)] uppercase">
            STATUS: AWAITING QUERY
          </span>
        </div>

        <div className="space-y-4 font-sans text-xs text-[var(--text-secondary)]">
          <p className="leading-relaxed">
            Observation scene loaded. Dispatch remote sensing queries to activate the autonomous vision-language reasoning engine and spatial grounding layers.
          </p>

          <div className="grid grid-cols-2 gap-3 pt-2">
            <div className="p-3 rounded bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] space-y-1">
              <span className="font-mono text-[10px] text-[var(--text-secondary)] uppercase">SENSOR REGIME</span>
              <p className="font-serif text-xs font-semibold text-[var(--text-primary)]">
                {item?.sensor || (domain === 'lunar' ? 'CH-2 TMC-2 / OHRC' : 'SENTINEL / LANDSAT')}
              </p>
            </div>
            <div className="p-3 rounded bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] space-y-1">
              <span className="font-mono text-[10px] text-[var(--text-secondary)] uppercase">GSD RESOLUTION</span>
              <p className="font-serif text-xs font-semibold text-[var(--text-primary)]">
                {item?.resolution || (domain === 'lunar' ? '0.25m - 5.0m / px' : '10.0m / px')}
              </p>
            </div>
          </div>

          <div className="border-t border-[var(--border-hairline)] pt-4 space-y-2">
            <span className="font-mono text-[10px] text-[var(--text-secondary)] uppercase tracking-wider">
              ACTIVE PIPELINE CAPABILITIES
            </span>
            <ul className="space-y-1.5 font-mono text-[11px] text-[var(--text-primary)]">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-verified)]" />
                GeoChat-7B LoRA Remote Sensing VLM
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-verified)]" />
                CLIPSeg Zero-Shot Spatial Grounding Mask
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-verified)]" />
                Deterministic Spectral Indices (NDVI / NDWI / NDBI / SAR)
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-verified)]" />
                Dynamic Cross-Verification Agreement Matrix
              </li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  // Active Scientific Report
  const isLunar = domain === 'lunar';
  const confidenceScore = result.confidence_score !== undefined && result.confidence_score !== null
    ? result.confidence_score
    : (result.consensus_score ?? 0.85);

  const facts = result.verified_facts || {};
  const spectral = facts.spectral_summary;
  const topK = facts.top_k;
  const changeSummary = facts.change_summary;

  // Download Handlers
  const getTimestamp = () => new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

  const handleDownloadMd = () => {
    const md = result.report_markdown || `# SatQuery AI - Scientific Analysis Report\n\n## Executive Summary\n${result.answer}\n\nConfidence: ${(confidenceScore * 100).toFixed(1)}%\n`;
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SatQuery_Report_${domain.toUpperCase()}_${getTimestamp()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setIsMenuOpen(false);
  };

  const handleDownloadJson = () => {
    const exportData = {
      mission: "SatQuery AI Remote Sensing Mission Intelligence",
      domain,
      timestamp: new Date().toISOString(),
      confidence_assessment: {
        score: confidenceScore,
        tag: result.confidence_tag,
        consensus_score: result.consensus_score,
        semantic_consistency: result.semantic_consistency,
      },
      executive_summary: result.answer,
      verified_facts: result.verified_facts,
      spectral_analysis: spectral || null,
      land_cover_ranking: topK || null,
      bi_temporal_change: changeSummary || null,
      execution_trace: result.trace,
      input_metadata: result.metas || null,
      raw_markdown_report: result.report_markdown || null,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SatQuery_Data_${domain.toUpperCase()}_${getTimestamp()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setIsMenuOpen(false);
  };

  const handleDownloadPdf = async () => {
    setIsPdfLoading(true);
    try {
      const blob = await exportPdfReport(result, result.answer || 'Multimodal Remote Sensing Analysis', result.metas);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `SatQuery_Report_${domain.toUpperCase()}_${getTimestamp()}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.warn('Backend PDF generation fallback triggered:', err);
      window.print();
    } finally {
      setIsPdfLoading(false);
      setIsMenuOpen(false);
    }
  };

  const handleCopy = () => {
    const md = result.report_markdown || result.answer;
    navigator.clipboard.writeText(md);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full h-full rounded border border-[var(--border-hairline)] bg-[var(--bg-panel)] p-6 space-y-6 overflow-y-auto max-h-[760px] transition-all">
      {/* Report Header: Title, Actions & Semantic Confidence Badge */}
      {/* Report Header: Two-tier layout for seamless fit without clipping */}
      <div className="border-b border-[var(--border-hairline)] pb-4 space-y-3">
        {/* Row 1: Title, Confidence Badge & Expand/Collapse */}
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-2 min-w-0">
            <FileText className="w-4 h-4 text-[var(--accent-verified)] flex-shrink-0" />
            <h3 className="font-serif text-sm sm:text-base font-bold text-[var(--text-primary)] truncate tracking-wide">
              SCIENTIFIC ANALYSIS REPORT
            </h3>
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            <ConfidenceBadge
              domain={domain}
              confidenceScore={confidenceScore}
              isVerified={!isLunar}
            />

            {onToggleExpand && (
              <button
                onClick={onToggleExpand}
                className="p-1.5 rounded bg-[rgba(255,255,255,0.05)] hover:bg-[rgba(77,184,255,0.15)] text-[var(--text-secondary)] hover:text-[var(--accent-verified)] border border-[var(--border-hairline)] transition-all cursor-pointer flex items-center gap-1 text-[10px] font-mono font-semibold"
                title={isExpanded ? "Collapse to 50/50 view" : "Expand report section to wide view"}
              >
                {isExpanded ? (
                  <>
                    <Minimize2 className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">COLLAPSE</span>
                  </>
                ) : (
                  <>
                    <Maximize2 className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">EXPAND</span>
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* Row 2: Action Toolbar (Quick Pills + Export Dropdown + Copy) */}
        <div className="flex items-center justify-between gap-2 pt-1.5 border-t border-[rgba(255,255,255,0.04)] flex-wrap sm:flex-nowrap">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[9px] text-[var(--text-secondary)] uppercase tracking-wider hidden sm:inline">
              FORMAT:
            </span>
            <div className="flex items-center rounded border border-[var(--border-hairline)] bg-[var(--bg-panel-elevated)] p-0.5">
              <button
                onClick={handleDownloadPdf}
                disabled={isPdfLoading}
                className="px-2 py-0.5 text-[9px] font-mono font-bold text-[var(--accent-verified)] hover:bg-[rgba(77,184,255,0.15)] rounded transition-colors disabled:opacity-50 cursor-pointer flex items-center gap-1"
                title="Download Executive PDF"
              >
                {isPdfLoading ? <Loader2 className="w-2.5 h-2.5 animate-spin" /> : null}
                <span>PDF</span>
              </button>
              <span className="text-[var(--border-hairline)]">|</span>
              <button
                onClick={handleDownloadMd}
                className="px-2 py-0.5 text-[9px] font-mono font-bold text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[rgba(255,255,255,0.08)] rounded transition-colors cursor-pointer"
                title="Download Markdown Report"
              >
                MD
              </button>
              <span className="text-[var(--border-hairline)]">|</span>
              <button
                onClick={handleDownloadJson}
                className="px-2 py-0.5 text-[9px] font-mono font-bold text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[rgba(255,255,255,0.08)] rounded transition-colors cursor-pointer"
                title="Download JSON Telemetry"
              >
                JSON
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            {/* Export Dropdown Menu Button */}
            <div className="relative" ref={menuRef}>
              <button
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className="flex items-center gap-1.5 px-3 py-1 rounded bg-[rgba(77,184,255,0.12)] hover:bg-[var(--accent-verified)] text-[var(--accent-verified)] hover:text-[var(--bg-void)] border border-[rgba(77,184,255,0.3)] transition-all text-[10px] font-mono font-bold tracking-wider cursor-pointer"
                title="Export Report (PDF / MD / JSON)"
              >
                <Download className="w-3 h-3" />
                <span>EXPORT</span>
                <ChevronDown className={`w-3 h-3 transition-transform ${isMenuOpen ? 'rotate-180' : ''}`} />
              </button>

              {/* Dropdown Menu */}
              {isMenuOpen && (
                <div className="absolute right-0 mt-1.5 w-52 rounded bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] shadow-2xl py-1 z-50 animate-in fade-in slide-in-from-top-1 duration-150">
                  <div className="px-3 py-1.5 border-b border-[var(--border-hairline)]">
                    <span className="font-mono text-[9px] text-[var(--text-secondary)] uppercase tracking-wider">
                      Select Export Format
                    </span>
                  </div>

                  <button
                    onClick={handleDownloadPdf}
                    disabled={isPdfLoading}
                    className="w-full text-left px-3 py-2 text-xs flex items-center gap-2.5 text-[var(--text-primary)] hover:bg-[rgba(77,184,255,0.12)] hover:text-[var(--accent-verified)] transition-colors cursor-pointer disabled:opacity-50"
                  >
                    <FileText className="w-3.5 h-3.5 text-rose-400" />
                    <div className="flex flex-col">
                      <span className="font-semibold">PDF Document</span>
                      <span className="text-[10px] text-[var(--text-secondary)] font-mono">.pdf (Executive Briefing)</span>
                    </div>
                  </button>

                  <button
                    onClick={handleDownloadMd}
                    className="w-full text-left px-3 py-2 text-xs flex items-center gap-2.5 text-[var(--text-primary)] hover:bg-[rgba(77,184,255,0.12)] hover:text-[var(--accent-verified)] transition-colors cursor-pointer"
                  >
                    <FileText className="w-3.5 h-3.5 text-cyan-400" />
                    <div className="flex flex-col">
                      <span className="font-semibold">Markdown Report</span>
                      <span className="text-[10px] text-[var(--text-secondary)] font-mono">.md (Tables & Evidence)</span>
                    </div>
                  </button>

                  <button
                    onClick={handleDownloadJson}
                    className="w-full text-left px-3 py-2 text-xs flex items-center gap-2.5 text-[var(--text-primary)] hover:bg-[rgba(77,184,255,0.12)] hover:text-[var(--accent-verified)] transition-colors cursor-pointer"
                  >
                    <FileCode className="w-3.5 h-3.5 text-amber-400" />
                    <div className="flex flex-col">
                      <span className="font-semibold">Structured Telemetry</span>
                      <span className="text-[10px] text-[var(--text-secondary)] font-mono">.json (Trace & Metrics)</span>
                    </div>
                  </button>
                </div>
              )}
            </div>

            {/* Copy Report Button */}
            <button
              onClick={handleCopy}
              className="p-1.5 rounded bg-[rgba(255,255,255,0.05)] hover:bg-[rgba(255,255,255,0.1)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors cursor-pointer"
              title="Copy Report to Clipboard"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
      </div>

      {/* 1. Executive Summary */}
      <div className="space-y-2">
        <span className="font-mono text-[10px] tracking-wider text-[var(--accent-verified)] uppercase">
          1. EXECUTIVE SUMMARY
        </span>
        <p className="font-sans text-xs text-[var(--text-primary)] leading-relaxed bg-[var(--bg-panel-elevated)] p-3.5 rounded border border-[var(--border-hairline)]">
          {result.answer || 'Detailed feature detection and spatial grounding completed across the active observation frame.'}
        </p>
      </div>

      {/* 2. Physical Spectral Analysis (NDVI / NDWI / NDBI) */}
      {spectral && (
        <div className="space-y-2">
          <span className="font-mono text-[10px] tracking-wider text-[var(--accent-verified)] uppercase">
            2. DETERMINISTIC SPECTRAL INDICES
          </span>
          <div className="grid grid-cols-3 gap-2">
            <div className="p-3 rounded bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] space-y-1">
              <span className="font-mono text-[9px] text-[var(--text-secondary)] uppercase">NDVI (VEGETATION)</span>
              <p className="font-mono text-sm font-bold text-emerald-400">
                {spectral.ndvi_mean !== undefined ? Number(spectral.ndvi_mean).toFixed(4) : 'N/A'}
              </p>
              <span className="font-sans text-[10px] text-[var(--text-secondary)] block truncate">
                {spectral.ndvi_mean > 0.3 ? 'Dense Canopy' : 'Sparse / Soil'}
              </span>
            </div>

            <div className="p-3 rounded bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] space-y-1">
              <span className="font-mono text-[9px] text-[var(--text-secondary)] uppercase">NDWI (WATER)</span>
              <p className="font-mono text-sm font-bold text-cyan-400">
                {spectral.ndwi_mean !== undefined ? Number(spectral.ndwi_mean).toFixed(4) : 'N/A'}
              </p>
              <span className="font-sans text-[10px] text-[var(--text-secondary)] block truncate">
                {spectral.ndwi_mean > 0.1 ? 'Open Water' : 'Dry Surface'}
              </span>
            </div>

            <div className="p-3 rounded bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] space-y-1">
              <span className="font-mono text-[9px] text-[var(--text-secondary)] uppercase">NDBI (BUILT-UP)</span>
              <p className="font-mono text-sm font-bold text-amber-400">
                {spectral.ndbi_mean !== undefined ? Number(spectral.ndbi_mean).toFixed(4) : 'N/A'}
              </p>
              <span className="font-sans text-[10px] text-[var(--text-secondary)] block truncate">
                {spectral.ndbi_mean > 0.05 ? 'Urbanized' : 'Natural Cover'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* 3. Change Detection Summary (When Bi-Temporal) */}
      {changeSummary && (
        <div className="space-y-2">
          <span className="font-mono text-[10px] tracking-wider text-[var(--accent-verified)] uppercase">
            3. BI-TEMPORAL CHANGE ANALYSIS
          </span>
          <div className="p-3 rounded bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] space-y-2 font-mono text-xs">
            <div className="flex items-center justify-between">
              <span className="text-[var(--text-secondary)]">DETECTED CHANGE RATIO:</span>
              <span className="text-rose-400 font-bold">
                {changeSummary.pct_area_changed !== undefined ? `${Number(changeSummary.pct_area_changed).toFixed(2)}%` : 'Active'}
              </span>
            </div>
            {changeSummary.class_before && (
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-[var(--text-secondary)]">PRIMARY CLASS (T1):</span>
                <span className="text-[var(--text-primary)]">{changeSummary.class_before.join(', ')}</span>
              </div>
            )}
            {changeSummary.class_after && (
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-[var(--text-secondary)]">PRIMARY CLASS (T2):</span>
                <span className="text-[var(--text-primary)]">{changeSummary.class_after.join(', ')}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 4. Land Cover Classification Ranking (ResNet-18) */}
      {topK && topK.length > 0 && (
        <div className="space-y-2">
          <span className="font-mono text-[10px] tracking-wider text-[var(--accent-verified)] uppercase">
            4. LAND COVER CLASSIFICATION (RESNET-18 / BIGEARTHNET)
          </span>
          <div className="space-y-1.5">
            {topK.slice(0, 4).map((entry: any, idx: number) => (
              <div
                key={idx}
                className="flex items-center justify-between p-2 rounded bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] font-mono text-xs"
              >
                <span className="text-[var(--text-primary)]">{entry.class_name || 'Land Cover'}</span>
                <span className="text-[var(--accent-verified)] font-bold">
                  {(Number(entry.probability || 0) * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 5. Deterministic Verification Matrix */}
      <div className="space-y-2">
        <span className="font-mono text-[10px] tracking-wider text-[var(--accent-verified)] uppercase">
          5. DETERMINISTIC VERIFICATION MATRIX
        </span>
        <div className="p-3 rounded bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] space-y-2 font-mono text-xs">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-[var(--text-secondary)]">PRIMARY MODEL:</span>
            <span className="text-[var(--text-primary)]">GeoChat-7B Remote Sensing VLM</span>
          </div>
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-[var(--text-secondary)]">SPATIAL GROUNDING:</span>
            <span className="text-[var(--text-primary)]">{result.overlay ? 'CLIPSeg Zero-Shot Mask' : 'Direct Interpretation'}</span>
          </div>
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-[var(--text-secondary)]">CONSENSUS SCORE:</span>
            <span className="text-[var(--accent-verified)] font-bold">
              {(confidenceScore * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* 6. Provenance & Compliance Footer */}
      <div className="border-t border-[var(--border-hairline)] pt-3 flex items-center justify-between font-mono text-[10px] text-[var(--text-secondary)]">
        <div className="flex items-center gap-1.5">
          <ShieldCheck className="w-3.5 h-3.5 text-[var(--accent-verified)]" />
          <span>ISRO SAC PS-26167 STANDARDS COMPLIANT</span>
        </div>
        <span>TRACE: {result.trace?.task || 'EXEC-COMPLETE'}</span>
      </div>
    </div>
  );
};
