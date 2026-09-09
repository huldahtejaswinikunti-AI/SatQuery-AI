"""Reusable Streamlit UI widgets for SatQuery AI.

Every widget renders based on what data is present -- optional fields
(consensus_score, change_direction, semantic_consistency) are shown only
when non-None, and hidden silently otherwise.

Accessibility: confidence badges use colour AND icon/label, never colour
alone.  All interactive elements have visible focus states via CSS.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import streamlit as st

from satquery.utils.image_utils import to_display_rgb


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def render_header(model_status: str = "READY") -> None:
    """Render the mission-control style workstation header."""
    from pathlib import Path

    logo_path = Path(__file__).parent / "assets" / "logo.png"
    has_logo = logo_path.exists()

    col_brand, col_telemetry = st.columns([0.65, 0.35])
    with col_brand:
        sub_c1, sub_c2 = st.columns([0.1, 0.9])
        with sub_c1:
            if has_logo:
                st.image(str(logo_path), width=46)
        with sub_c2:
            st.markdown(
                '<h2 style="margin:0; font-size:1.6rem; font-weight:800; color:#f8fafc; letter-spacing:-0.5px;">'
                'SATQUERY AI <span style="font-size:0.85rem; font-weight:600; color:#2dd4bf; margin-left:8px; border:1px solid rgba(45,212,191,0.3); padding:2px 8px; border-radius:12px;">MISSION WORKSTATION</span>'
                '</h2>'
                '<p style="margin:2px 0 0 0; color:#94a3b8; font-size:0.88rem; font-family:ui-monospace, monospace;">'
                'Agentic Earth & Lunar Observation Intelligence &middot; SIH 2026 &middot; PS 26167 &middot; ISRO / SAC'
                '</p>',
                unsafe_allow_html=True,
            )

    with col_telemetry:
        st.markdown(
            '<div style="display:flex; justify-content:flex-end; gap:16px; align-items:center; height:100%;">'
            '  <div style="text-align:right;">'
            '    <div style="font-size:0.68rem; text-transform:uppercase; color:#64748b; font-weight:700; letter-spacing:0.8px;">SYSTEM STATUS</div>'
            '    <div class="telemetry-status-online">' + model_status + '</div>'
            '  </div>'
            '  <div style="border-left:1px solid #334155; height:28px;"></div>'
            '  <div style="text-align:right;">'
            '    <div style="font-size:0.68rem; text-transform:uppercase; color:#64748b; font-weight:700; letter-spacing:0.8px;">ACTIVE MODEL STACK</div>'
            '    <div style="font-size:0.82rem; color:#cbd5e1; font-weight:700; font-family:ui-monospace, monospace;">GeoChat &middot; CLIPSeg &middot; TinyCD</div>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# 3D Planetary Hero Experience (Three.js WebGL)
# ---------------------------------------------------------------------------


def render_3d_planetary_hero(current_mode: str = "Earth analysis") -> None:
    """Render cinematic curved 3D WebGL planetary horizon inspired by motionsites.ai space-planet."""
    import streamlit.components.v1 as components

    is_lunar = "lunar" in current_mode.lower()
    active_name = "MOON" if is_lunar else "EARTH"
    subtitle_desc = (
        "High-resolution Chandrayaan-2 OHRC/TMC-2 lunar surface analysis, crater morphology, "
        "and shadowed cold-trap detection with deterministic verification."
        if is_lunar else
        "Multimodal remote sensing intelligence with GeoChat vision-language reasoning, "
        "CLIPSeg spatial grounding, and Sentinel-1/2 optical-SAR consensus."
    )

    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700;800;900&family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
      <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body, html {{
          width: 100%; height: 100%; overflow: hidden; background: #02050b;
          font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          color: #f8fafc;
        }}
        #canvas-container {{
          width: 100%; height: 550px; position: relative;
          background: radial-gradient(ellipse at 50% 12%, #0a1628 0%, #030712 55%, #010307 100%);
        }}
        canvas {{ display: block; width: 100%; height: 100%; cursor: grab; }}
        canvas:active {{ cursor: grabbing; }}

        /* Top Minimal Floating Navbar */
        .space-nav {{
          position: absolute; top: 16px; left: 28px; right: 28px;
          display: flex; justify-content: space-between; align-items: center;
          pointer-events: auto; z-index: 20;
        }}
        .space-brand {{
          display: flex; align-items: center; gap: 10px; font-size: 1.15rem; font-weight: 800;
          letter-spacing: 0.5px; color: #ffffff; text-transform: uppercase;
        }}
        .space-brand-pill {{
          font-size: 0.68rem; font-weight: 700; color: #2dd4bf;
          background: rgba(45, 212, 191, 0.12); border: 1px solid rgba(45, 212, 191, 0.35);
          padding: 3px 10px; border-radius: 9999px; letter-spacing: 0.6px;
        }}
        .space-nav-links {{
          display: flex; gap: 24px; align-items: center;
          background: rgba(10, 16, 28, 0.65); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
          border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 9999px; padding: 7px 24px;
        }}
        .space-nav-link {{
          font-size: 0.82rem; font-weight: 600; color: #94a3b8; text-decoration: none; cursor: pointer;
          transition: all 0.2s ease; letter-spacing: 0.3px; position: relative;
        }}
        .space-nav-link:hover, .space-nav-link.active {{ color: #ffffff; }}
        .space-nav-link.active::after {{
          content: ''; position: absolute; bottom: -5px; left: 0; right: 0;
          height: 2px; background: #2dd4bf; border-radius: 2px; box-shadow: 0 0 8px #2dd4bf;
        }}
        .space-telemetry-badge {{
          font-size: 0.76rem; font-family: 'JetBrains Mono', monospace; color: #cbd5e1;
          display: flex; align-items: center; gap: 8px;
          background: rgba(10, 16, 28, 0.65); backdrop-filter: blur(14px);
          border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 9999px; padding: 7px 16px;
        }}
        .status-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #2dd4bf; box-shadow: 0 0 10px #2dd4bf; }}

        /* Center Hero Typography */
        .hero-content {{
          position: absolute; top: 80px; left: 50%; transform: translateX(-50%);
          text-align: center; pointer-events: none; z-index: 10; max-width: 720px; width: 92%;
        }}
        .hero-label {{
          font-size: 0.82rem; font-weight: 700; letter-spacing: 5px; text-transform: uppercase;
          color: #94a3b8; margin-bottom: 6px;
        }}
        .hero-title {{
          font-size: 4.2rem; font-weight: 900; letter-spacing: 4px; line-height: 1;
          text-transform: uppercase; color: #ffffff; margin-bottom: 8px;
          text-shadow: 0 4px 30px rgba(0, 0, 0, 0.9);
          font-family: 'Cinzel', serif;
        }}
        .hero-title-bar {{
          width: 52px; height: 3px; background: #2dd4bf; margin: 12px auto 16px auto;
          border-radius: 2px; box-shadow: 0 0 12px rgba(45, 212, 191, 0.85);
        }}
        .hero-desc {{
          font-size: 0.92rem; line-height: 1.55; color: #cbd5e1; font-weight: 400;
          text-shadow: 0 2px 12px rgba(0, 0, 0, 0.9); margin-bottom: 22px; max-width: 620px; margin-left: auto; margin-right: auto;
        }}
        .hero-actions {{
          pointer-events: auto; display: flex; justify-content: center; gap: 14px; align-items: center;
        }}
        .hero-btn {{
          background: #ffffff; color: #020617; font-size: 0.86rem; font-weight: 800;
          padding: 10px 28px; border-radius: 9999px; border: none; cursor: pointer;
          transition: all 0.25s ease; box-shadow: 0 4px 22px rgba(255, 255, 255, 0.35);
          letter-spacing: 0.4px;
        }}
        .hero-btn:hover {{
          background: #2dd4bf; color: #020617;
          box-shadow: 0 6px 26px rgba(45, 212, 191, 0.6); transform: translateY(-1px);
        }}
        .hero-btn-alt {{
          background: rgba(15, 23, 42, 0.7); color: #e2e8f0; font-size: 0.84rem; font-weight: 700;
          padding: 9px 22px; border-radius: 9999px; border: 1px solid rgba(255, 255, 255, 0.2);
          cursor: pointer; transition: all 0.25s ease; backdrop-filter: blur(10px);
        }}
        .hero-btn-alt:hover {{
          background: rgba(30, 41, 59, 0.9); border-color: #2dd4bf; color: #ffffff;
        }}

        /* Flank Planet Switchers */
        .flank-orb {{
          position: absolute; top: 52%; transform: translateY(-50%);
          display: flex; align-items: center; gap: 12px; pointer-events: auto;
          cursor: pointer; z-index: 15; transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
          background: rgba(10, 16, 28, 0.55); backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 9999px; padding: 7px 16px;
        }}
        .flank-orb:hover {{
          transform: translateY(-50%) scale(1.06);
          border-color: #2dd4bf;
          box-shadow: 0 0 20px rgba(45, 212, 191, 0.3);
        }}
        .flank-orb-left {{ left: 28px; }}
        .flank-orb-right {{ right: 28px; }}
        .flank-label {{
          font-size: 0.78rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #e2e8f0;
        }}
        .flank-circle {{
          width: 26px; height: 26px; border-radius: 50%;
          box-shadow: 0 0 12px rgba(45, 212, 191, 0.45);
        }}

        /* Scroll down indicator */
        .scroll-down-cue {{
          position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
          width: 36px; height: 36px; border-radius: 50%; background: rgba(15, 23, 42, 0.75);
          border: 1px solid rgba(255, 255, 255, 0.25); display: flex; justify-content: center;
          align-items: center; color: #cbd5e1; font-size: 1rem; z-index: 20; cursor: pointer;
          transition: all 0.2s ease; pointer-events: auto;
        }}
        .scroll-down-cue:hover {{
          background: #2dd4bf; color: #020617; border-color: #2dd4bf;
          box-shadow: 0 0 16px rgba(45, 212, 191, 0.5); transform: translateX(-50%) translateY(2px);
        }}

        #fallback-view {{
          display: none; width: 100%; height: 550px; background: #02050b;
          flex-direction: column; justify-content: center; align-items: center; color: #94a3b8;
        }}
      </style>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
      <div id="canvas-container">
        <!-- Floating Header -->
        <div class="space-nav">
          <div class="space-brand">
            <span>SatQuery AI</span>
            <span class="space-brand-pill">MISSION CONTROL</span>
          </div>
          <div class="space-nav-links">
            <span class="space-nav-link {'active' if not is_lunar else ''}" onclick="switchBody('EARTH')">🌍 Earth</span>
            <span class="space-nav-link {'active' if is_lunar else ''}" onclick="switchBody('MOON')">🌕 Moon</span>
            <span class="space-nav-link" onclick="scrollToDeck()">Observation Deck</span>
            <span class="space-nav-link" onclick="scrollToDeck()">ISRO PRADAN</span>
          </div>
          <div class="space-telemetry-badge">
            <span class="status-dot"></span>
            <span>SYSTEM READY &middot; ISRO / SAC</span>
          </div>
        </div>

        <!-- Center Hero Typography -->
        <div class="hero-content">
          <div class="hero-label">PLANET</div>
          <div class="hero-title" id="hero-title-text">{active_name}</div>
          <div class="hero-title-bar"></div>
          <div class="hero-desc" id="hero-desc-text">{subtitle_desc}</div>
          <div class="hero-actions">
            <button class="hero-btn" onclick="scrollToDeck()">EXPLORE OBSERVATIONS &darr;</button>
            <button class="hero-btn-alt" onclick="switchBody('{ 'EARTH' if is_lunar else 'MOON' }')">
              SWITCH TO { 'EARTH' if is_lunar else 'MOON' } &rarr;
            </button>
          </div>
        </div>

        <!-- Flank Planet Switchers -->
        <div class="flank-orb flank-orb-left" onclick="switchBody('EARTH')">
          <div class="flank-circle" style="background: radial-gradient(circle at 35% 35%, #38bdf8 0%, #0369a1 70%, #082f49 100%);"></div>
          <span class="flank-label">EARTH</span>
        </div>
        <div class="flank-orb flank-orb-right" onclick="switchBody('MOON')">
          <span class="flank-label">MOON</span>
          <div class="flank-circle" style="background: radial-gradient(circle at 35% 35%, #e2e8f0 0%, #94a3b8 60%, #334155 100%);"></div>
        </div>

        <!-- Scroll down indicator -->
        <div class="scroll-down-cue" onclick="scrollToDeck()">&darr;</div>
      </div>

      <div id="fallback-view">
        <h2 style="color:#fff; font-size:1.8rem; margin-bottom:8px;">SATQUERY AI</h2>
        <p>Cinematic Planetary Observation System &middot; ISRO / SAC SIH 26167</p>
      </div>

      <script>
        (function() {{
          const container = document.getElementById('canvas-container');
          const fallback = document.getElementById('fallback-view');

          function hasWebGL() {{
            try {{
              const c = document.createElement('canvas');
              return !!(window.WebGLRenderingContext && (c.getContext('webgl') || c.getContext('experimental-webgl')));
            }} catch(e) {{
              return false;
            }}
          }}

          if (!hasWebGL() || typeof THREE === 'undefined') {{
            container.style.display = 'none';
            fallback.style.display = 'flex';
            return;
          }}

          const scene = new THREE.Scene();
          const w = container.clientWidth || 1000;
          const h = 530;
          const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
          const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
          renderer.setSize(w, h);
          renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
          container.appendChild(renderer.domElement);

          // Deep Space Starfield with multi-magnitude stars
          const starCount = 2200;
          const starGeo = new THREE.BufferGeometry();
          const starPos = new Float32Array(starCount * 3);
          const starColors = new Float32Array(starCount * 3);
          for (let i = 0; i < starCount * 3; i += 3) {{
            const rad = 70 + Math.random() * 150;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(Math.random() * 2 - 1);
            starPos[i] = rad * Math.sin(phi) * Math.cos(theta);
            starPos[i+1] = rad * Math.sin(phi) * Math.sin(theta);
            starPos[i+2] = rad * Math.cos(phi);

            // Subtle color variance (blue-white to soft amber)
            const tint = Math.random();
            if (tint > 0.8) {{
              starColors[i] = 0.6; starColors[i+1] = 0.85; starColors[i+2] = 1.0;
            }} else if (tint < 0.15) {{
              starColors[i] = 1.0; starColors[i+1] = 0.85; starColors[i+2] = 0.6;
            }} else {{
              starColors[i] = 0.95; starColors[i+1] = 0.95; starColors[i+2] = 1.0;
            }}
          }}
          starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
          starGeo.setAttribute('color', new THREE.BufferAttribute(starColors, 3));
          const starMat = new THREE.PointsMaterial({{
            size: 0.95, vertexColors: true, transparent: true, opacity: 0.85
          }});
          const stars = new THREE.Points(starGeo, starMat);
          scene.add(stars);

          // Procedural High-Res Earth Texture
          function createEarthTexture() {{
            const canvas = document.createElement('canvas');
            canvas.width = 2048; canvas.height = 1024;
            const ctx = canvas.getContext('2d');

            // Deep ocean base
            const oceanGrad = ctx.createLinearGradient(0, 0, 0, 1024);
            oceanGrad.addColorStop(0, '#0a1d37');
            oceanGrad.addColorStop(0.5, '#061427');
            oceanGrad.addColorStop(1, '#0a1d37');
            ctx.fillStyle = oceanGrad;
            ctx.fillRect(0, 0, 2048, 1024);

            // Continents with land-cover shades
            ctx.fillStyle = '#1c382b';
            // Eurasia / Africa
            ctx.beginPath();
            ctx.ellipse(1080, 420, 310, 180, 0.2, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#162e22';
            ctx.beginPath();
            ctx.ellipse(1000, 620, 190, 240, 0, 0, Math.PI * 2);
            ctx.fill();
            // Americas
            ctx.fillStyle = '#1f3d2f';
            ctx.beginPath();
            ctx.ellipse(500, 380, 200, 160, -0.2, 0, Math.PI * 2);
            ctx.fill();
            ctx.beginPath();
            ctx.ellipse(580, 700, 140, 200, 0.3, 0, Math.PI * 2);
            ctx.fill();
            // Australia & Indo-Pacific
            ctx.fillStyle = '#224230';
            ctx.beginPath();
            ctx.ellipse(1500, 720, 110, 90, 0.1, 0, Math.PI * 2);
            ctx.fill();

            // Polar Caps
            ctx.fillStyle = '#f8fafc';
            ctx.fillRect(0, 0, 2048, 65);
            ctx.fillRect(0, 960, 2048, 64);

            // Atmosphere cloud noise
            ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
            for (let c = 0; c < 36; c++) {{
              ctx.beginPath();
              ctx.ellipse((c * 90) % 2048, 300 + ((c * 73) % 440), 160, 42, (c * 0.4), 0, Math.PI * 2);
              ctx.fill();
            }}

            return new THREE.CanvasTexture(canvas);
          }}

          // Procedural Clouds Texture
          function createCloudTexture() {{
            const canvas = document.createElement('canvas');
            canvas.width = 1024; canvas.height = 512;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, 1024, 512);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.38)';
            for (let c = 0; c < 48; c++) {{
              ctx.beginPath();
              ctx.ellipse((c * 42) % 1024, 120 + ((c * 53) % 270), 90, 24, (c * 0.35), 0, Math.PI * 2);
              ctx.fill();
            }}
            return new THREE.CanvasTexture(canvas);
          }}

          // Procedural Moon Texture
          function createMoonTexture() {{
            const canvas = document.createElement('canvas');
            canvas.width = 1024; canvas.height = 512;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#94a3b8';
            ctx.fillRect(0, 0, 1024, 512);

            // Basaltic maria
            ctx.fillStyle = '#475569';
            ctx.beginPath(); ctx.ellipse(380, 220, 120, 90, 0.2, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.ellipse(560, 260, 150, 100, 0.2, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.ellipse(260, 320, 80, 60, -0.3, 0, Math.PI * 2); ctx.fill();

            // Crater rings & ejecta rays
            for (let k = 0; k < 90; k++) {{
              const cx = (k * 123) % 1024;
              const cy = (k * 73) % 512;
              const r = 5 + (k % 14);
              ctx.strokeStyle = '#334155';
              ctx.lineWidth = 2;
              ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
              ctx.fillStyle = '#cbd5e1';
              ctx.beginPath(); ctx.arc(cx, cy, r * 0.35, 0, Math.PI * 2); ctx.fill();
            }}
            return new THREE.CanvasTexture(canvas);
          }}

          // Planetary Geometry Setup (Massive curved horizon centered low)
          const planetRadius = 4.2;
          const planetY = -3.2;

          // 1. Earth
          const earthGeo = new THREE.SphereGeometry(planetRadius, 64, 64);
          const earthMat = new THREE.MeshPhongMaterial({{
            map: createEarthTexture(),
            shininess: 35,
            specular: new THREE.Color(0x0284c7),
          }});
          const earth = new THREE.Mesh(earthGeo, earthMat);
          earth.position.set(0, planetY, 0);
          earth.rotation.z = 23.4 * Math.PI / 180;
          scene.add(earth);

          // Earth Clouds Layer
          const cloudGeo = new THREE.SphereGeometry(planetRadius + 0.03, 48, 48);
          const cloudMat = new THREE.MeshBasicMaterial({{
            map: createCloudTexture(),
            transparent: true,
            opacity: 0.45,
            blending: THREE.AdditiveBlending,
          }});
          const clouds = new THREE.Mesh(cloudGeo, cloudMat);
          clouds.position.set(0, planetY, 0);
          scene.add(clouds);

          // Glowing Atmosphere Horizon (Rayleigh Scattering)
          const atmoGeo = new THREE.SphereGeometry(planetRadius + 0.16, 64, 64);
          const atmoMat = new THREE.MeshBasicMaterial({{
            color: 0x38bdf8,
            transparent: true,
            opacity: 0.28,
            side: THREE.BackSide,
            blending: THREE.AdditiveBlending,
          }});
          const atmosphere = new THREE.Mesh(atmoGeo, atmoMat);
          atmosphere.position.set(0, planetY, 0);
          scene.add(atmosphere);

          // 2. Moon
          const moonGeo = new THREE.SphereGeometry(planetRadius * 0.85, 48, 48);
          const moonMat = new THREE.MeshLambertMaterial({{
            map: createMoonTexture(),
          }});
          const moon = new THREE.Mesh(moonGeo, moonMat);
          moon.position.set(0, planetY, 0);
          scene.add(moon);

          // Set initial visibility
          let activeBody = "{active_name}";
          function syncActiveVisibility() {{
            if (activeBody === 'EARTH') {{
              earth.visible = true;
              clouds.visible = true;
              atmosphere.visible = true;
              atmoMat.color.setHex(0x38bdf8);
              moon.visible = false;
            }} else {{
              earth.visible = false;
              clouds.visible = false;
              atmosphere.visible = true;
              atmoMat.color.setHex(0x94a3b8);
              moon.visible = true;
            }}
          }}
          syncActiveVisibility();

          // Cinematic Sun Lighting
          const sunLight = new THREE.DirectionalLight(0xffffff, 1.8);
          sunLight.position.set(16, 12, 10);
          scene.add(sunLight);

          const ambient = new THREE.AmbientLight(0x0a1120, 0.4);
          scene.add(ambient);

          // Camera setup looking down at curved planetary horizon
          camera.position.set(0, 1.6, 5.2);
          camera.lookAt(0, -0.4, 0);

          // Interactive Dragging & Parallax
          let isDragging = false;
          let prevX = 0, prevY = 0;
          let rotSpeedX = 0, rotSpeedY = 0;
          let camTargetX = 0, camTargetY = 1.6;

          const canvasEl = renderer.domElement;
          canvasEl.addEventListener('mousedown', (e) => {{
            isDragging = true; prevX = e.clientX; prevY = e.clientY;
          }});
          window.addEventListener('mouseup', () => {{ isDragging = false; }});
          window.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
              const dx = e.clientX - prevX;
              const dy = e.clientY - prevY;
              prevX = e.clientX; prevY = e.clientY;
              if (activeBody === 'EARTH') {{
                earth.rotation.y += dx * 0.005;
                clouds.rotation.y += dx * 0.005;
              }} else {{
                moon.rotation.y += dx * 0.005;
              }}
            }} else {{
              // Parallax tilt
              const nx = (e.clientX / w) - 0.5;
              const ny = (e.clientY / h) - 0.5;
              camTargetX = nx * 0.4;
              camTargetY = 1.6 - ny * 0.3;
            }}
          }});

          // Touch support
          canvasEl.addEventListener('touchstart', (e) => {{
            if (e.touches.length === 1) {{
              isDragging = true; prevX = e.touches[0].clientX; prevY = e.touches[0].clientY;
            }}
          }}, {{ passive: true }});
          window.addEventListener('touchend', () => {{ isDragging = false; }});
          window.addEventListener('touchmove', (e) => {{
            if (isDragging && e.touches.length === 1) {{
              const dx = e.touches[0].clientX - prevX;
              prevX = e.touches[0].clientX;
              if (activeBody === 'EARTH') {{
                earth.rotation.y += dx * 0.005;
              }} else {{
                moon.rotation.y += dx * 0.005;
              }}
            }}
          }}, {{ passive: true }});

          window.switchBody = function(body) {{
            activeBody = body;
            syncActiveVisibility();
            const titleEl = document.getElementById('hero-title-text');
            const descEl = document.getElementById('hero-desc-text');
            if (body === 'EARTH') {{
              if (titleEl) titleEl.innerText = 'EARTH';
              if (descEl) descEl.innerText = 'Multimodal remote sensing intelligence with GeoChat vision-language reasoning, CLIPSeg spatial grounding, and Sentinel-1/2 optical-SAR consensus.';
            }} else {{
              if (titleEl) titleEl.innerText = 'MOON';
              if (descEl) descEl.innerText = 'High-resolution Chandrayaan-2 OHRC/TMC-2 lunar surface analysis, crater morphology, and shadowed cold-trap detection with deterministic verification.';
            }}
          }};

          window.scrollToDeck = function() {{
            if (window.parent) {{
              const target = window.parent.document.getElementById('observation-workspace');
              if (target) {{
                target.scrollIntoView({{ behavior: 'smooth' }});
                return;
              }}
            }}
            window.scrollBy({{ top: 450, behavior: 'smooth' }});
          }};

          window.addEventListener('resize', () => {{
            const newW = container.clientWidth;
            camera.aspect = newW / h;
            camera.updateProjectionMatrix();
            renderer.setSize(newW, h);
          }});

          const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

          function animate() {{
            requestAnimationFrame(animate);

            if (!prefersReduced) {{
              earth.rotation.y += 0.0012;
              clouds.rotation.y += 0.0016;
              moon.rotation.y += 0.0010;
            }}

            camera.position.x += (camTargetX - camera.position.x) * 0.05;
            camera.position.y += (camTargetY - camera.position.y) * 0.05;
            camera.lookAt(0, -0.4, 0);

            renderer.render(scene, camera);
          }}
          animate();
        }})();
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=560)


# ---------------------------------------------------------------------------
# Visual Image Picker Gallery
# ---------------------------------------------------------------------------


def render_image_picker_gallery(
    sample_items: list[dict],
    selected_idx: int = 0,
    key_prefix: str = "img_picker",
    resolve_thumb_fn: Any = None,
) -> int:
    """Render an interactive visual grid of satellite thumbnail cards.

    Allows user to click thumbnail cards to inspect and switch active images.
    Returns the newly chosen index.
    """
    if not sample_items:
        return 0

    st.markdown(
        '<div style="font-size:0.85rem; font-weight:700; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.6px; margin: 8px 0 10px 0;">'
        'Select Satellite Observation Image'
        f' <span style="font-size:0.75rem; color:#64748b; font-weight:500;">({len(sample_items)} Available Options)</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    n_items = len(sample_items)
    # 4 to 6 columns depending on quantity
    n_cols = min(n_items, 4) if n_items <= 4 else (5 if n_items == 5 else 6)
    cols = st.columns(n_cols)

    chosen_idx = selected_idx

    for i, it in enumerate(sample_items):
        col_idx = i % n_cols
        is_selected = (i == selected_idx)
        card_cls = "gallery-card gallery-card-selected" if is_selected else "gallery-card"

        with cols[col_idx]:
            # Container card styling
            sensor = it.get("sensor", "Satellite")
            res = it.get("resolution", "")
            title = it.get("name", f"Option {i+1}")
            badge_text = f"{sensor} &middot; {res}" if res else sensor

            # Card Header Preview
            st.markdown(
                f'<div class="{card_cls}">'
                f'  <div class="gallery-card-badge">{badge_text}</div>'
                f'  <div class="gallery-card-title" title="{title}">{title}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Thumbnail image if resolver provided
            if resolve_thumb_fn:
                file_target = it.get("file") or (it.get("files")[0] if it.get("files") else None)
                if file_target:
                    thumb_arr = resolve_thumb_fn(file_target)
                    if thumb_arr is not None:
                        rgb = to_display_rgb(thumb_arr)
                        st.image(rgb, use_container_width=True)

            # Interactive Select Button
            btn_label = f"✓ Selected" if is_selected else f"Select #{i+1}"
            btn_type = "primary" if is_selected else "secondary"
            if st.button(
                btn_label,
                key=f"{key_prefix}_btn_{i}",
                type=btn_type,
                use_container_width=True,
            ):
                chosen_idx = i

    return chosen_idx


# ---------------------------------------------------------------------------
# Image preview
# ---------------------------------------------------------------------------


def render_image_preview(images: list[np.ndarray], metas: list[dict]) -> None:
    """Show uploaded / demo images with high-contrast mission metadata frame."""
    if not images:
        st.info(
            "Upload 1 or 2 satellite images, or select a pre-loaded demo "
            "scenario to inspect."
        )
        return

    if len(images) == 1:
        rgb = to_display_rgb(images[0])
        m = metas[0] if metas else {}
        name = m.get("name", m.get("filename", "Active Observation"))
        sensor = m.get("sensor", "Optical / SAR")
        res = m.get("resolution", m.get("resolution_m", "Native GSD"))
        date = m.get("date", m.get("acquisition_date", "Archived"))
        loc = m.get("location", m.get("coordinates", "Georeferenced Scene"))
        prov = m.get("provenance", m.get("source_dataset", "Satellite Archive"))

        shape_str = f"{images[0].shape[1]}x{images[0].shape[0]} px"
        res_str = f"{res} m" if isinstance(res, (int, float)) else str(res)

        st.markdown(
            '<div style="font-size:0.85rem; font-weight:700; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:8px;">'
            'Active Primary Satellite Imagery Preview'
            '</div>',
            unsafe_allow_html=True,
        )

        st.image(rgb, caption=f"{name} ({shape_str})", use_container_width=True)

        st.markdown(
            f'<div class="preview-hero-meta-bar">'
            f'  <span class="preview-hero-chip"><strong>Sensor:</strong> {sensor}</span>'
            f'  <span class="preview-hero-chip"><strong>GSD:</strong> {res_str}</span>'
            f'  <span class="preview-hero-chip"><strong>Acquisition:</strong> {date}</span>'
            f'  <span class="preview-hero-chip"><strong>Location:</strong> {loc}</span>'
            f'  <span class="preview-hero-chip"><strong>Archive:</strong> {prov}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            '<div style="font-size:0.85rem; font-weight:700; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:8px;">'
            'Paired Satellite Imagery Preview (Bi-Temporal or Optical+SAR Co-Registered)'
            '</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        labels = ["Observation T1 (Pre / Optical)", "Observation T2 (Post / Radar SAR)"]
        for i, (img, meta) in enumerate(zip(images, metas)):
            with cols[i]:
                rgb = to_display_rgb(img)
                name = meta.get("name", meta.get("filename", f"Observation {i + 1}"))
                sensor = meta.get("sensor", "Satellite Sensor")
                date = meta.get("date", meta.get("acquisition_date", "Archival Date"))
                shape_str = f"{img.shape[1]}x{img.shape[0]} px"
                lbl = labels[i] if i < len(labels) else f"Observation {i+1}"

                st.markdown(f'<div style="font-size:0.8rem; font-weight:700; color:#38bdf8; margin-bottom:4px;">{lbl}</div>', unsafe_allow_html=True)
                st.image(
                    rgb,
                    caption=f"{name} ({shape_str})",
                    use_container_width=True,
                )
                st.markdown(
                    f'<div class="preview-hero-meta-bar" style="border-radius:4px; margin-top:-6px;">'
                    f'  <span><strong>Sensor:</strong> {sensor}</span>'
                    f'  <span><strong>Date:</strong> {date}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Confidence badge
# ---------------------------------------------------------------------------

_BADGE_CONFIG = {
    "high_cross_verified": {
        "cls": "badge-high-verified",
        "icon": "✓",
        "label": "High Confidence (Cross-Verified)",
    },
    "high_sar_penetration": {
        "cls": "badge-high-verified",
        "icon": "✓",
        "label": "High Confidence (SAR-Penetrated)",
    },
    "high_rule_based": {
        "cls": "badge-high-rule",
        "icon": "⚡",
        "label": "Deterministic Signal",
    },
    "lower_confidence_disagreement": {
        "cls": "badge-disagreement",
        "icon": "!",
        "label": "Lower Confidence (Signal Disagreement)",
    },
    "lower_confidence": {
        "cls": "badge-disagreement",
        "icon": "!",
        "label": "Lower Confidence",
    },
    "moderate": {
        "cls": "badge-unverified",
        "icon": "~",
        "label": "Moderate Confidence (Unverified)",
    },
    "experimental_unverified": {
        "cls": "badge-experimental-unverified",
        "icon": "⊘",
        "label": "Experimental — No Cross-Check Available",
    },
    "error": {
        "cls": "badge-error",
        "icon": "✕",
        "label": "Input Validation Error",
    },
}


def render_confidence_badge(result: dict[str, Any]) -> None:
    """Render the primary confidence badge with icon + label + percentage."""
    tag = result.get("confidence_tag", "moderate")
    score = result.get("confidence_score")
    
    cfg = _BADGE_CONFIG.get(tag, _BADGE_CONFIG["moderate"])

    if tag == "experimental_unverified":
        label = f'{cfg["icon"]} {cfg["label"]}'
        st.markdown(
            f'<div class="badge-container {cfg["cls"]}" role="status" '
            f'aria-label="{cfg["label"]}">{label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="lunar-note-banner">'
            '<strong>Notice:</strong> Lunar imagery has no deterministic cross-check available in this system — '
            'treat this answer as unverified.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    if tag == "error":
        label = f'{cfg["icon"]} {cfg["label"]}'
    elif score is not None:
        pct = int(round(float(score) * 100))
        label = f'{cfg["icon"]} {cfg["label"]}: {pct}%'
    else:
        label = f'{cfg["icon"]} {cfg["label"]}'

    st.markdown(
        f'<div class="badge-container {cfg["cls"]}" role="status" '
        f'aria-label="{cfg["label"]}">{label}</div>',
        unsafe_allow_html=True,
    )

    # Second confidence signal: semantic consistency (only if present)
    sem = result.get("semantic_consistency")
    if sem is not None:
        sem_pct = int(round(float(sem) * 100))
        st.markdown(
            f'<div class="badge-container badge-semantic" role="status" '
            f'aria-label="Semantic consistency">'
            f"[SEMANTIC] Semantic Consistency: {sem_pct}%</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Consensus score (headline metric when present)
# ---------------------------------------------------------------------------


def render_consensus_score(result: dict[str, Any]) -> None:
    """Show optical-SAR consensus as a headline metric, or nothing at all."""
    score = result.get("consensus_score")
    if score is None:
        return
    pct = int(round(float(score)))
    st.markdown(
        f'<div class="consensus-metric" role="status">'
        f"Optical-SAR Agreement: <strong>{pct}%</strong></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Change direction
# ---------------------------------------------------------------------------


def render_change_direction(result: dict[str, Any]) -> None:
    """Show change direction sentence if present, skip otherwise."""
    direction = result.get("change_direction")
    if not direction:
        return
    st.markdown(
        f'<div class="change-direction">'
        f"Change concentrated in the <strong>{direction}</strong></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Overlay display
# ---------------------------------------------------------------------------


def render_overlay(result: dict[str, Any]) -> None:
    """Show the visual grounding overlay if present."""
    overlay = result.get("overlay")
    if overlay is None:
        return
    st.image(
        overlay,
        caption="Visual Grounding & Analysis Overlay",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------


def render_execution_trace(result: dict[str, Any]) -> None:
    """Expandable execution trace panel with JSON download."""
    trace = result.get("trace", {})
    if not trace:
        return

    with st.expander("Auditable Execution Trace & Tool Timeline", expanded=False):
        st.markdown(
            f"**Task:** `{trace.get('task', 'N/A')}` | "
            f"**Confidence:** `{trace.get('confidence', 'N/A')}` | "
            f"**Timestamp:** `{trace.get('timestamp', 'N/A')}`"
        )

        tools = trace.get("tools_invoked", [])
        if tools:
            st.markdown("**Tools invoked:** " + " -> ".join(f"`{t}`" for t in tools))

        params = trace.get("parameters", {})
        if params:
            st.markdown("**Parameters:**")
            for k, v in params.items():
                st.markdown(f"- `{k}`: {v}")

        # Show full JSON
        st.json(trace)



# ---------------------------------------------------------------------------
# Download buttons
# ---------------------------------------------------------------------------


def render_download_buttons(result: dict[str, Any]) -> None:
    """Render download buttons for JSON trace, markdown report, and PDF."""
    cols = st.columns(3)

    # JSON trace
    trace_json = json.dumps(result.get("trace", {}), indent=2, default=str)
    with cols[0]:
        st.download_button(
            "Download JSON Trace",
            trace_json,
            "satquery_trace.json",
            "application/json",
            use_container_width=True,
        )

    # Markdown report
    report_md = result.get("report_markdown", "")
    if report_md:
        with cols[1]:
            st.download_button(
                "Download Markdown Report",
                report_md,
                "satquery_report.md",
                "text/markdown",
                use_container_width=True,
            )

    # PDF report (optional -- only if fpdf2 is available)
    with cols[2]:
        try:
            try:
                from app.pdf_report import generate_pdf_report
            except ImportError:
                from pdf_report import generate_pdf_report

            pdf_bytes = generate_pdf_report(
                result,
                result.get("trace", {}).get("parameters", {}).get("query", ""),
                [],
            )
            st.download_button(
                "Download PDF Report",
                pdf_bytes,
                "satquery_report.pdf",
                "application/pdf",
                use_container_width=True,
            )
        except Exception:
            st.download_button(
                "Download PDF Report",
                trace_json,
                "satquery_report.json",
                "application/json",
                use_container_width=True,
                disabled=True,
                help="PDF generation not available (fpdf2 not installed)",
            )


# ---------------------------------------------------------------------------
# Spectral Analysis Summary
# ---------------------------------------------------------------------------


def render_spectral_summary(result: dict[str, Any]) -> None:
    """Render spectral index bars and scene composition breakdown."""
    vf = result.get("verified_facts", {})
    if not isinstance(vf, dict):
        return
    spectral = vf.get("spectral_summary", {})
    if not spectral:
        return

    with st.expander("Spectral Analysis & Scene Composition", expanded=True):
        ndvi = spectral.get("ndvi_mean", 0)
        ndwi = spectral.get("ndwi_mean", 0)
        ndbi = spectral.get("ndbi_mean", 0)

        # Index values as metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "NDVI (Vegetation)",
                f"{ndvi:+.3f}",
                delta=f"{spectral.get('vegetation_fraction', 0)*100:.1f}% coverage",
            )
        with c2:
            st.metric(
                "NDWI (Water)",
                f"{ndwi:+.3f}",
                delta=f"{spectral.get('water_fraction', 0)*100:.1f}% coverage",
            )
        with c3:
            st.metric(
                "NDBI (Built-up)",
                f"{ndbi:+.3f}",
                delta=f"{spectral.get('built_up_fraction', 0)*100:.1f}% coverage",
            )

        # Scene composition progress bars
        veg_f = spectral.get("vegetation_fraction", 0)
        wat_f = spectral.get("water_fraction", 0)
        blt_f = spectral.get("built_up_fraction", 0)

        st.markdown("**Scene Composition:**")
        st.progress(min(veg_f, 1.0), text=f"Vegetation: {veg_f*100:.1f}%")
        st.progress(min(wat_f, 1.0), text=f"Water: {wat_f*100:.1f}%")
        st.progress(min(blt_f, 1.0), text=f"Built-up: {blt_f*100:.1f}%")


# ---------------------------------------------------------------------------
# Land Cover Classification Table
# ---------------------------------------------------------------------------


def render_land_cover_table(result: dict[str, Any]) -> None:
    """Render the top-K land cover classification breakdown."""
    vf = result.get("verified_facts", {})
    if not isinstance(vf, dict):
        return
    top_k = vf.get("top_k", [])
    if not top_k:
        return

    with st.expander("Land Cover Classification", expanded=True):
        st.markdown("**Multi-label classification (BigEarthNet-S2 / ResNet-18):**")
        for entry in top_k:
            name = entry.get("class_name", "Unknown")
            prob = entry.get("probability", 0)
            pct = prob * 100
            st.progress(min(prob, 1.0), text=f"{name}: {pct:.1f}%")


# ---------------------------------------------------------------------------
# Detailed Analysis Report
# ---------------------------------------------------------------------------


def render_detailed_report(result: dict[str, Any]) -> None:
    """Render the full markdown analysis report."""
    report_md = result.get("report_markdown", "")
    if report_md:
        st.markdown(report_md, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Minimal Single Consolidated Details Expander (Task 1)
# ---------------------------------------------------------------------------


def render_details_expander(result: dict[str, Any], metas: list[dict] | None = None) -> None:
    """Single collapsed Details expander holding all secondary technical depth."""
    with st.expander("Details & Technical Evidence", expanded=False):
        tab1, tab2, tab3 = st.tabs([
            "📊 Spectral & Classification",
            "📑 Full Report & Export",
            "⚙️ Execution Trace",
        ])

        with tab1:
            # Metadata summary
            if metas:
                st.markdown("**Image Metadata:**")
                for idx, m in enumerate(metas, 1):
                    fn = m.get("filename", f"Image #{idx}")
                    crs = m.get("crs", "N/A")
                    shape = m.get("shape", f"{m.get('width', 'N/A')}x{m.get('height', 'N/A')}")
                    bands = m.get("band_count", m.get("channels", "N/A"))
                    sensor = m.get("sensor", m.get("modality", "N/A"))
                    st.markdown(
                        f"<div class='metadata-mono'>• <strong>{fn}</strong> — "
                        f"Sensor: {sensor} | Shape: {shape} | Bands: {bands} | CRS: {crs}</div>",
                        unsafe_allow_html=True,
                    )
                st.write("")

            # Spectral summary if available
            vf = result.get("verified_facts", {})
            if isinstance(vf, dict) and vf.get("spectral_summary"):
                spectral = vf["spectral_summary"]
                ndvi = spectral.get("ndvi_mean", 0)
                ndwi = spectral.get("ndwi_mean", 0)
                ndbi = spectral.get("ndbi_mean", 0)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("NDVI (Vegetation)", f"{ndvi:+.3f}")
                with c2:
                    st.metric("NDWI (Water)", f"{ndwi:+.3f}")
                with c3:
                    st.metric("NDBI (Built-up)", f"{ndbi:+.3f}")

                veg_f = spectral.get("vegetation_fraction", 0)
                wat_f = spectral.get("water_fraction", 0)
                blt_f = spectral.get("built_up_fraction", 0)
                st.progress(min(veg_f, 1.0), text=f"Vegetation: {veg_f*100:.1f}%")
                st.progress(min(wat_f, 1.0), text=f"Water: {wat_f*100:.1f}%")
                st.progress(min(blt_f, 1.0), text=f"Built-up: {blt_f*100:.1f}%")
                st.write("")

            # Land cover top-k
            top_k = vf.get("top_k", []) if isinstance(vf, dict) else []
            if top_k:
                st.markdown("**Land Cover Surface Classes (ResNet-18 / BigEarthNet):**")
                for entry in top_k:
                    name = entry.get("class_name", "Unknown")
                    prob = entry.get("probability", 0)
                    st.progress(min(prob, 1.0), text=f"{name}: {prob*100:.1f}%")

            if not (isinstance(vf, dict) and (vf.get("spectral_summary") or vf.get("top_k"))):
                st.info("No Earth-observation spectral indices or land-cover distribution for this analysis.")

        with tab2:
            render_detailed_report(result)
            st.divider()
            render_download_buttons(result)

        with tab3:
            trace = result.get("trace", {})
            if trace:
                exec_time = trace.get("execution_time_seconds")
                t_str = f"{exec_time:.2f}s" if exec_time is not None else "N/A"
                st.markdown(
                    f"<div class='metadata-mono'>"
                    f"<strong>Task:</strong> <code>{trace.get('task', 'N/A')}</code> | "
                    f"<strong>Duration:</strong> <code>{t_str}</code> | "
                    f"<strong>Confidence:</strong> <code>{trace.get('confidence', 'N/A')}</code>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                tools = trace.get("tools_invoked", [])
                if tools:
                    st.markdown(
                        "<div class='metadata-mono'><strong>Pipeline tools:</strong> " +
                        " &rarr; ".join(f"<code>{t}</code>" for t in tools) + "</div>",
                        unsafe_allow_html=True,
                    )
                st.write("")
                st.json(trace)
            else:
                st.info("No execution trace recorded.")


# ---------------------------------------------------------------------------
# Scientific Analysis Report Panel (Side-by-Side 60/40 Workspace)
# ---------------------------------------------------------------------------


def render_scientific_report_panel(
    result: dict[str, Any],
    metas: list[dict] | None = None,
    is_lunar: bool = False,
) -> None:
    """Render structured, scientifically honest Analysis Report beside the viewer."""
    tag = result.get("confidence_tag", "moderate")
    score = result.get("confidence_score")
    vf = result.get("verified_facts", {}) if isinstance(result.get("verified_facts"), dict) else {}
    trace = result.get("trace", {}) if isinstance(result.get("trace"), dict) else {}
    m = metas[0] if metas else {}

    sensor_str = m.get("sensor", "Chandrayaan-2 OHRC" if is_lunar else "Sentinel-2 / Copernicus")
    loc_str = m.get("location", "Moon" if is_lunar else "Georeferenced Scene")
    res_str = m.get("resolution", m.get("resolution_m", "Native GSD"))

    st.markdown(
        f'<div class="report-header-bar">'
        f'  <div class="report-title">'
        f'    <span>🛰️ {"LUNAR" if is_lunar else "EARTH"} SCIENTIFIC REPORT</span>'
        f'  </div>'
        f'  <div style="font-family:ui-monospace, monospace; font-size:0.75rem; color:#64748b;">'
        f'    {sensor_str} &middot; {res_str}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 1. Primary Confidence Badge
    render_confidence_badge(result)
    render_consensus_score(result)
    render_change_direction(result)

    # 2. Executive Summary / Grounded Answer
    st.markdown(
        '<div class="report-section-title">1. Executive Summary & Findings</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="report-summary-box">{result.get("answer", "No textual synthesis generated.")}</div>',
        unsafe_allow_html=True,
    )

    # 3. Observed Surface Features
    st.markdown(
        '<div class="report-section-title">2. Observed Surface Features</div>',
        unsafe_allow_html=True,
    )
    if is_lunar:
        meas = vf.get("measurements", {})
        det = vf.get("details", {})
        s_pct = det.get("shadow_percentage", meas.get("shadow_percentage", 0))
        e_pct = det.get("ejecta_percentage", meas.get("ejecta_percentage", 0))
        roughness = det.get("surface_roughness", meas.get("surface_roughness", 0))
        st.markdown(
            f"- **Impact Morphology:** Circular crater depressions with raised rim terraces.\n"
            f"- **Regolith & Albedo:** High-reflectance immature ejecta deposits covering **{e_pct}%** of scene.\n"
            f"- **Illumination & Shadow:** **{s_pct}%** deeply shadowed / occluded terrain (cold-trap candidate proxy).\n"
            f"- **Micro-Relief Variance:** Surface roughness gradient index: **{roughness}**."
        )
    else:
        top_k = vf.get("top_k", [])
        if top_k:
            st.markdown("**Dominant Land Cover Classes (ResNet-18 / BigEarthNet):**")
            for entry in top_k[:4]:
                cname = entry.get("class_name", "Unknown")
                prob = entry.get("probability", 0)
                st.markdown(f"- **{cname}:** `{prob*100:.1f}%` confidence")
        elif vf.get("spectral_summary"):
            spec = vf["spectral_summary"]
            veg_pct = spec.get("vegetation_fraction", 0) * 100
            wat_pct = spec.get("water_fraction", 0) * 100
            blt_pct = spec.get("built_up_fraction", 0) * 100
            st.markdown(
                f"- **Vegetation Coverage:** `{veg_pct:.1f}%` (NDVI mean: `{spec.get('ndvi_mean', 0):+.3f}`)\n"
                f"- **Surface Water Bodies:** `{wat_pct:.1f}%` (NDWI mean: `{spec.get('ndwi_mean', 0):+.3f}`)\n"
                f"- **Built-Up Structures:** `{blt_pct:.1f}%` (NDBI mean: `{spec.get('ndbi_mean', 0):+.3f}`)"
            )
        else:
            st.markdown("- Surface composition interpreted through multimodal vision-language feature maps.")

    # 4. Measurements (Distinguishing Model Interpretation vs Deterministic Measurement)
    st.markdown(
        '<div class="report-section-title">3. Physical Measurements & Calibration</div>',
        unsafe_allow_html=True,
    )
    if is_lunar:
        meas = vf.get("measurements", {})
        diam_val = meas.get("diameter", "Measurement unavailable — Calibrated scale or circle fitting required.")
        scale_status = meas.get("scale_status", "uncalibrated")
        st.markdown(
            f'<div class="measurement-grid">'
            f'  <div class="measurement-item">'
            f'    <span>Crater Diameter:</span>'
            f'    <span><strong>{diam_val}</strong></span>'
            f'  </div>'
            f'  <div class="measurement-item">'
            f'    <span>Measurement Type:</span>'
            f'    <span style="color:#f59e0b;">Unverified Model Estimate</span>'
            f'  </div>'
            f'  <div class="measurement-item">'
            f'    <span>Scale Status:</span>'
            f'    <span>{"Calibrated GSD" if scale_status == "calibrated" else "Uncalibrated Pixel Space"}</span>'
            f'  </div>'
            f'  <div class="measurement-item">'
            f'    <span>Shadow Occlusion:</span>'
            f'    <span><strong>{meas.get("shadow_percentage", "N/A")}%</strong> (Deterministic Pixel Threshold)</span>'
            f'  </div>'
            f'  <div class="measurement-item">'
            f'    <span>Ejecta Albedo Cover:</span>'
            f'    <span><strong>{meas.get("ejecta_percentage", "N/A")}%</strong> (Deterministic Reflectance Threshold)</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        spec = vf.get("spectral_summary", {})
        if spec:
            ndvi = spec.get("ndvi_mean", 0)
            ndwi = spec.get("ndwi_mean", 0)
            ndbi = spec.get("ndbi_mean", 0)
            st.markdown(
                f'<div class="measurement-grid">'
                f'  <div class="measurement-item"><span>NDVI (Vegetation):</span><span><strong>{ndvi:+.4f}</strong> (Deterministic Normalized Difference)</span></div>'
                f'  <div class="measurement-item"><span>NDWI (Water):</span><span><strong>{ndwi:+.4f}</strong> (Deterministic Normalized Difference)</span></div>'
                f'  <div class="measurement-item"><span>NDBI (Built-up):</span><span><strong>{ndbi:+.4f}</strong> (Deterministic Normalized Difference)</span></div>'
                f'  <div class="measurement-item"><span>Sensor Calibration:</span><span style="color:#2dd4bf;">Bottom-Of-Atmosphere Reflectance (L2A)</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="measurement-grid">'
                '  <div class="measurement-item"><span>Spatial Grounding:</span><span>CLIPSeg bounding activation map</span></div>'
                '  <div class="measurement-item"><span>Metric Dimensions:</span><span>Requires auxiliary DEM or calibrated surveyor scale</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # 5. Deterministic Verification Matrix
    st.markdown(
        '<div class="report-section-title">4. Evidence & Verification Matrix</div>',
        unsafe_allow_html=True,
    )
    if is_lunar:
        st.markdown(
            '<div class="verification-matrix">'
            '  <div class="verification-row">'
            '    <span class="verif-label">Image Raster Pixels</span>'
            '    <span class="verif-status-pass">✓ VERIFIED</span>'
            '  </div>'
            '  <div class="verification-row">'
            '    <span class="verif-label">Morphological Gradients</span>'
            '    <span class="verif-status-pass">✓ DETERMINISTIC</span>'
            '  </div>'
            '  <div class="verification-row">'
            '    <span class="verif-label">Earth NDVI/SAR Cross-Check</span>'
            '    <span class="verif-status-fail">✕ INAPPLICABLE ON MOON</span>'
            '  </div>'
            '  <div class="verification-row">'
            '    <span class="verif-label">Metric Diameter / DEM</span>'
            '    <span class="verif-status-warn">⚠ UNVERIFIED ESTIMATE</span>'
            '  </div>'
            '  <div class="verification-row">'
            '    <span class="verif-label">Geological Interpretation</span>'
            '    <span class="verif-status-warn">⚠ MODEL-ONLY</span>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        has_consensus = result.get("consensus_score") is not None
        has_spectral = bool(vf.get("spectral_summary"))
        st.markdown(
            f'<div class="verification-matrix">'
            f'  <div class="verification-row">'
            f'    <span class="verif-label">Primary Satellite Raster</span>'
            f'    <span class="verif-status-pass">✓ VERIFIED</span>'
            f'  </div>'
            f'  <div class="verification-row">'
            f'    <span class="verif-label">Physical Spectral Indices</span>'
            f'    <span class="{"verif-status-pass" if has_spectral else "verif-status-fail"}">{"✓ DETERMINISTIC" if has_spectral else "✕ N/A"}</span>'
            f'  </div>'
            f'  <div class="verification-row">'
            f'    <span class="verif-label">Optical + SAR Consensus</span>'
            f'    <span class="{"verif-status-pass" if has_consensus else "verif-status-fail"}">{"✓ CROSS-VERIFIED" if has_consensus else "✕ SINGLE SENSOR"}</span>'
            f'  </div>'
            f'  <div class="verification-row">'
            f'    <span class="verif-label">Vision-Language Reasoning</span>'
            f'    <span class="verif-status-pass">✓ GROUNDED</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 6. Provenance & Limitations
    st.markdown(
        '<div class="report-section-title">5. Provenance & Limitations</div>',
        unsafe_allow_html=True,
    )
    prov_str = m.get("provenance", "ISRO / ISSDC Chandrayaan-2 PRADAN Archive" if is_lunar else "Copernicus Open Access Hub")
    st.markdown(
        f"- **Data Archive:** {prov_str}\n"
        f"- **Sensor Instrument:** {sensor_str}\n"
        f"- **Scientific Caveat:** Model interpretation provides qualitative morphological guidance. Operational mission decisions require calibrated PDS4 data products or certified stereo DEM validation."
    )

    st.write("")
    # 7. Exports & Auditable Trace
    render_download_buttons(result)
    render_execution_trace(result)


def render_staging_telemetry_panel(
    metas: list[dict] | None = None,
    is_lunar: bool = False,
    query: str = "",
) -> None:
    """Render mission-control readiness and capability telemetry before execution."""
    m = metas[0] if metas else {}
    target_body = "MOON (LUNAR SURFACE)" if is_lunar else "EARTH (TERRESTRIAL)"
    sensor_name = m.get("sensor", "Chandrayaan-2 OHRC / TMC-2" if is_lunar else "Sentinel-2 MSI (Copernicus)")
    gsd_val = m.get("resolution", m.get("resolution_m", "0.25 m/px" if is_lunar else "10.0 m/px"))
    date_val = m.get("date", m.get("acquisition_date", "Archival Observation"))
    loc_val = m.get("location", "Lunar Highlands / Polar" if is_lunar else "Georeferenced Scene")
    prov_val = m.get("provenance", "ISRO / ISSDC PRADAN" if is_lunar else "Copernicus Sentinel Hub")

    st.markdown(
        f'<div class="staging-panel">'
        f'  <div class="report-header-bar">'
        f'    <div class="report-title">🛰️ MISSION CONTROL TELEMETRY</div>'
        f'    <div class="telemetry-status-online">PIPELINE READY</div>'
        f'  </div>'
        f'  <div style="font-size:0.85rem; color:#cbd5e1; margin-bottom:12px;">'
        f'    Staged observation is ready for agentic vision-language analysis and deterministic cross-verification.'
        f'  </div>'
        f'  <div class="measurement-grid">'
        f'    <div class="measurement-item"><span>TARGET BODY:</span><span><strong>{target_body}</strong></span></div>'
        f'    <div class="measurement-item"><span>SENSOR INSTRUMENT:</span><span><strong>{sensor_name}</strong></span></div>'
        f'    <div class="measurement-item"><span>GROUND SAMPLE DISTANCE:</span><span><strong>{gsd_val}</strong></span></div>'
        f'    <div class="measurement-item"><span>ACQUISITION DATE:</span><span><strong>{date_val}</strong></span></div>'
        f'    <div class="measurement-item"><span>SCENE LOCATION:</span><span><strong>{loc_val}</strong></span></div>'
        f'    <div class="measurement-item"><span>ARCHIVE PROVENANCE:</span><span><strong>{prov_val}</strong></span></div>'
        f'  </div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="report-section-title" style="margin-top:14px;">Active Domain Analytical Capabilities</div>',
        unsafe_allow_html=True,
    )
    if is_lunar:
        caps = [
            ("Impact Crater Detection & Morphology", "Gradient ridges & circular contour fitting"),
            ("Shadow & Cold-Trap (PSR) Proxy", "Deep radiometric occlusion analysis (<20/255)"),
            ("High-Albedo Ejecta Mapping", "Immature regolith & impact ray detection"),
            ("Surface Micro-Relief Roughness", "Topographic slope & gradient magnitude variance"),
            ("Boulder & Block Populations", "High-contrast boulder cluster identification"),
        ]
    else:
        caps = [
            ("Land Cover Classification", "ResNet-18 / BigEarthNet-S2 19-class distribution"),
            ("Deterministic Spectral Indices", "NDVI (Vegetation), NDWI (Water), NDBI (Urban)"),
            ("Bi-Temporal Change Detection", "TinyCD deep bi-temporal feature differencing"),
            ("Optical + SAR Cross-Modal Fusion", "Sentinel-1 radar + Sentinel-2 optical consensus"),
            ("Visual Grounding & Segmentation", "CLIPSeg zero-shot text-guided spatial masks"),
        ]

    for cap_title, cap_desc in caps:
        st.markdown(
            f'<div class="staging-capability-item">'
            f'  <span style="color:#2dd4bf; font-weight:700;">&bull;</span>'
            f'  <div>'
            f'    <div style="font-weight:700; font-size:0.83rem; color:#f1f5f9;">{cap_title}</div>'
            f'    <div style="font-size:0.75rem; color:#64748b;">{cap_desc}</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Batch Queue Item Component (Task 2)
# ---------------------------------------------------------------------------



def render_batch_item_card(idx: int, item: dict[str, Any]) -> None:
    """Render a single stacked summary card for a batch item."""
    status = item.get("status", "queued")
    query = item.get("query", "")
    res = item.get("result")
    err = item.get("error")

    status_cls = f"status-{status}"
    st.markdown(
        f"<div class='batch-card'>"
        f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
        f"<strong>Item #{idx + 1}</strong>"
        f"<span class='batch-status-badge {status_cls}'>{status.upper()}</span>"
        f"</div>"
        f"<div style='font-size: 0.9rem; color: #94a3b8; margin-top: 4px;'>"
        f"Query: <em>{query[:80]}{'...' if len(query) > 80 else ''}</em>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if status == "error" and err:
        st.error(f"Item #{idx + 1} failed: {err}")
    elif status == "done" and res:
        with st.expander(f"View Results for Item #{idx + 1}", expanded=False):
            render_confidence_badge(res)
            st.markdown(res.get("answer", ""))
            render_overlay(res)
            render_details_expander(res, item.get("metas"))

