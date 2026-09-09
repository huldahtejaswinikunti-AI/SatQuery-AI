export type ObservationDomain = 'earth' | 'lunar';

export interface ObservationItem {
  id: string;
  file?: string;
  files?: string[];
  name: string;
  sensor: string;
  resolution?: string;
  date?: string;
  location?: string;
  query?: string;
  provenance?: string;
}

export interface Scenario {
  id?: string;
  label: string;
  task: string;
  query: string;
  why_this_matters_to_isro?: string;
  why_it_matters?: string;
  sample_items?: ObservationItem[];
  sample_files?: string[];
  files?: string[];
}

export interface TelemetryMeta {
  filename?: string;
  sensor?: string;
  resolution_m?: number | string;
  acquisition_date?: string;
  crs?: string;
  coordinates?: string;
  source_dataset?: string;
  band_count?: number;
  width?: number;
  height?: number;
  [key: string]: any;
}

export interface AnalysisResult {
  answer: string;
  confidence: string;
  confidence_tag: string;
  confidence_score?: number | null;
  overlay?: string | null; // base64 data URI
  verified_facts?: Record<string, any>;
  consensus_score?: number | null;
  change_direction?: string | null;
  semantic_consistency?: number | null;
  validation_failure_reason?: string | null;
  trace?: Record<string, any>;
  metas?: TelemetryMeta[];
  report_markdown?: string;
}
