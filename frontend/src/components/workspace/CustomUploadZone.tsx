import React, { useState, useRef } from 'react';
import type { ObservationDomain, ObservationItem } from '../../types';
import { uploadCustomRasters } from '../../api/client';
import { UploadCloud, FileCheck, AlertCircle, X, Image as ImageIcon, Loader2 } from 'lucide-react';

interface CustomUploadZoneProps {
  domain: ObservationDomain;
  onUploadSuccess: (items: ObservationItem[]) => void;
  uploadedItems: ObservationItem[];
  onClearUploads: () => void;
}

export const CustomUploadZone: React.FC<CustomUploadZoneProps> = ({
  domain,
  onUploadSuccess,
  uploadedItems,
  onClearUploads,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = async (selectedFiles: FileList | File[]) => {
    const validFiles: File[] = [];
    const validExtensions = ['.tif', '.tiff', '.geotiff', '.png', '.jpg', '.jpeg'];

    for (let i = 0; i < selectedFiles.length; i++) {
      const f = selectedFiles[i];
      const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
      if (validExtensions.includes(ext)) {
        validFiles.push(f);
      }
    }

    if (validFiles.length === 0) {
      setUploadError('Unsupported file format. Please upload GeoTIFF (.tif), PNG, or JPEG raster files.');
      return;
    }

    if (validFiles.length > 2) {
      setUploadError('Please select at most 2 raster files (1 for single-scene analysis or 2 for paired fusion/change detection).');
      return;
    }

    setIsUploading(true);
    setUploadError(null);

    try {
      const response = await uploadCustomRasters(validFiles, domain);
      if (response.items && response.items.length > 0) {
        onUploadSuccess(response.items);
      }
    } catch (err: any) {
      setUploadError(err.message || 'Upload failed. Please verify raster format and try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  return (
    <div className="w-full space-y-4">
      {/* Upload Dropzone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-lg p-6 sm:p-8 text-center transition-all duration-200 cursor-pointer ${
          isDragging
            ? 'border-[var(--accent-verified)] bg-[rgba(77,184,255,0.08)] shadow-[0_0_25px_rgba(77,184,255,0.2)]'
            : uploadedItems.length > 0
            ? 'border-[var(--border-hairline)] bg-[var(--bg-panel-elevated)] hover:border-[var(--accent-verified)]'
            : 'border-[var(--border-hairline)] bg-[var(--bg-panel)] hover:bg-[var(--bg-panel-elevated)] hover:border-[rgba(77,184,255,0.5)]'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFiles(e.target.files);
            }
          }}
        />

        <div className="flex flex-col items-center justify-center gap-3">
          <div
            className={`w-12 h-12 rounded-full flex items-center justify-center transition-all ${
              isDragging
                ? 'bg-[var(--accent-verified)] text-[var(--bg-void)] scale-110'
                : 'bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] text-[var(--accent-verified)]'
            }`}
          >
            {isUploading ? (
              <Loader2 className="w-6 h-6 animate-spin text-[var(--accent-verified)]" />
            ) : (
              <UploadCloud className="w-6 h-6" />
            )}
          </div>

          <div className="space-y-1">
            <div className="text-sm font-sans font-bold text-[var(--text-primary)]">
              {isUploading
                ? 'Ingesting & Decoding Raster Imagery...'
                : isDragging
                ? 'Release to upload raster files'
                : 'Click to Browse or Drag & Drop Custom Satellite Rasters'}
            </div>
            <p className="text-xs text-[var(--text-secondary)] font-mono">
              Accepts 1 raster (single analysis) or 2 rasters (Optical+SAR fusion / Bi-temporal change)
            </p>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2 pt-1 text-[11px] font-mono text-[var(--text-secondary)]">
            <span className="px-2 py-0.5 rounded bg-[var(--bg-void)] border border-[var(--border-hairline)]">
              GeoTIFF (.tif, .geotiff)
            </span>
            <span className="px-2 py-0.5 rounded bg-[var(--bg-void)] border border-[var(--border-hairline)]">
              PNG (.png)
            </span>
            <span className="px-2 py-0.5 rounded bg-[var(--bg-void)] border border-[var(--border-hairline)]">
              JPEG (.jpg)
            </span>
            <span className="text-[var(--accent-verified)]">
              &middot; Native Multispectral, SAR & Lunar Rasters
            </span>
          </div>
        </div>
      </div>

      {/* Error Alert */}
      {uploadError && (
        <div className="flex items-center gap-2.5 p-3 rounded bg-[rgba(255,92,122,0.1)] border border-[rgba(255,92,122,0.3)] text-[var(--accent-conflict)] text-xs font-mono">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      {/* Uploaded Files Summary Cards */}
      {uploadedItems.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs text-[var(--accent-verified)] uppercase tracking-wider flex items-center gap-1.5">
              <FileCheck className="w-3.5 h-3.5" />
              <span>ACTIVE USER CUSTOM RASTERS ({uploadedItems.length} LOADED):</span>
            </span>
            <button
              type="button"
              onClick={onClearUploads}
              className="text-[11px] font-mono text-[var(--text-secondary)] hover:text-[var(--accent-conflict)] flex items-center gap-1 transition-colors cursor-pointer"
            >
              <X className="w-3 h-3" />
              <span>Clear Uploads</span>
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {uploadedItems.map((item, idx) => (
              <div
                key={item.id || idx}
                className="flex items-start gap-3 p-3 rounded-lg border border-[var(--accent-verified)] bg-[var(--bg-panel-elevated)] shadow-[0_0_15px_rgba(77,184,255,0.08)]"
              >
                <div className="w-10 h-10 rounded bg-[var(--bg-void)] border border-[var(--border-hairline)] flex items-center justify-center shrink-0 text-[var(--accent-verified)]">
                  <ImageIcon className="w-5 h-5" />
                </div>
                <div className="min-w-0 flex-1 space-y-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-sans text-xs font-bold text-[var(--text-primary)] truncate">
                      {item.name || item.filename}
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--bg-void)] text-[var(--accent-verified)] border border-[var(--border-hairline)]">
                      {item.sensor || 'Raster'}
                    </span>
                  </div>
                  <div className="text-[11px] font-mono text-[var(--text-secondary)] truncate">
                    {item.provenance || 'User Telemetry'}
                  </div>
                  <div className="text-[10px] font-mono text-[var(--accent-verified)] truncate">
                    {item.location || 'Local Coordinates'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
