"""Self-contained HTML report. The 2D map is the deliverable.

Renders a single .html file with:
  - scatter of surfaces (color by kind) and evals (different marker)
  - overlaid heatmap of g(x), the gap field
  - top-K gap regions outlined and labeled with bounding surfaces
  - hover tooltips with text + source location
  - sidebar with coverage score, region table, surface counts
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .density import GapField
from .evals import EvalInput
from .surfaces import Surface


@dataclass
class ReportData:
    target: str
    surfaces: list[Surface]
    evals: list[EvalInput]
    surface_points_2d: NDArray[np.float64]
    eval_points_2d: NDArray[np.float64]
    gap: GapField
    proposed_cases: list[dict]
    region_neighbors: dict[str, list[int]]


def write_html_report(report: ReportData, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize(report)
    html = _HTML_TEMPLATE.replace("__PAYLOAD__", _safe_json_for_script(payload))
    out.write_text(html, encoding="utf-8")
    return out


def _safe_json_for_script(payload: dict) -> str:
    """Serialize a payload safe for inlining inside an HTML <script> block.

    json.dumps does not escape `<`, `>`, `&`, `/`, or U+2028/U+2029, any of which can
    break out of the script-tag context if they appear in target-repo text (a docstring
    containing `</script>`, a prompt with U+2028 line separator, etc).
    """
    raw = json.dumps(payload, ensure_ascii=False)
    return (
        raw.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def _serialize(report: ReportData) -> dict:
    return {
        "target": report.target,
        "surfaces": [
            {
                "id": s.id,
                "text": s.text,
                "kind": s.kind,
                "source": f"{s.source_file}:{s.source_line}",
                "container": s.container,
                "x": float(report.surface_points_2d[i, 0]),
                "y": float(report.surface_points_2d[i, 1]),
            }
            for i, s in enumerate(report.surfaces)
        ],
        "evals": [
            {
                "id": e.id,
                "text": e.text,
                "source": e.source_file,
                "x": float(report.eval_points_2d[i, 0]),
                "y": float(report.eval_points_2d[i, 1]),
            }
            for i, e in enumerate(report.evals)
        ] if len(report.eval_points_2d) else [],
        "gap_field": {
            "x": report.gap.grid_x.tolist(),
            "y": report.gap.grid_y.tolist(),
            "z": report.gap.gap_field.tolist(),
            "threshold": report.gap.threshold,
        },
        "regions": [
            {
                "id": r.id,
                "centroid": list(r.centroid_2d),
                "bbox": list(r.bbox),
                "mass": r.integrated_mass,
                "neighbors": [
                    {
                        "id": report.surfaces[idx].id,
                        "text": report.surfaces[idx].text,
                        "kind": report.surfaces[idx].kind,
                        "source": f"{report.surfaces[idx].source_file}:{report.surfaces[idx].source_line}",
                    }
                    for idx in report.region_neighbors.get(r.id, [])
                    if idx < len(report.surfaces)
                ],
                "proposed": [c for c in report.proposed_cases if c.get("region_id") == r.id],
            }
            for r in report.gap.regions
        ],
        "summary": {
            "coverage": report.gap.coverage_score,
            "alpha": report.gap.alpha,
            "bandwidth": report.gap.bandwidth,
            "threshold": report.gap.threshold,
            "surface_count": len(report.surfaces),
            "eval_count": len(report.evals),
            "region_count": len(report.gap.regions),
            "kinds": _kind_counts(report.surfaces),
        },
    }


def _kind_counts(surfaces: list[Surface]) -> dict:
    counts: dict[str, int] = {}
    for s in surfaces:
        counts[s.kind] = counts.get(s.kind, 0) + 1
    return counts


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Kartograf — Semantic Coverage Map</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    --bg: #0b0e14;
    --panel: #131820;
    --border: #232a36;
    --text: #d7dde8;
    --muted: #7a8599;
    --surface: #f59e0b;
    --eval: #38bdf8;
    --gap: #ef4444;
    --accent: #a78bfa;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: var(--bg); color: var(--text); }
  header { padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; gap: 24px; align-items: baseline; }
  h1 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: 0.02em; }
  .subtitle { color: var(--muted); font-size: 13px; }
  .layout { display: grid; grid-template-columns: 1fr 380px; gap: 0; height: calc(100vh - 53px); }
  #map { background: var(--panel); }
  aside { background: var(--panel); border-left: 1px solid var(--border); overflow-y: auto; padding: 16px 18px; }
  .stat { display: flex; justify-content: space-between; padding: 6px 0; font-size: 13px; border-bottom: 1px solid var(--border); }
  .stat .label { color: var(--muted); }
  .stat .value { font-variant-numeric: tabular-nums; }
  .coverage-big { font-size: 38px; font-weight: 700; letter-spacing: -0.02em; margin: 4px 0 12px; }
  .coverage-big .frac { color: var(--muted); font-size: 18px; font-weight: 400; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin: 18px 0 8px; font-weight: 600; }
  .region { border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; margin-bottom: 8px; }
  .region-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
  .region-id { font-weight: 600; color: var(--gap); }
  .region-mass { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .region-neighbors { font-size: 12px; color: var(--text); }
  .region-neighbors li { margin: 2px 0; padding-left: 0; list-style: none; }
  .region-neighbors li::before { content: "·  "; color: var(--muted); }
  .kind-prompt { color: var(--accent); }
  .kind-tool, .kind-tool_arg { color: var(--surface); }
  .kind-docstring { color: #94a3b8; }
  .kind-eval { color: var(--eval); }
  .proposed { margin-top: 10px; padding: 10px 12px; background: #0f1722; border-radius: 4px; border: 1px solid #1f2735; font-size: 12px; }
  .proposed .label { color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-size: 10px; margin-bottom: 6px; display: flex; justify-content: space-between; }
  .proposed .scenario { font-weight: 600; color: var(--text); margin-bottom: 6px; font-size: 13px; }
  .proposed .bdd { line-height: 1.55; }
  .proposed .bdd .clause { display: block; margin: 3px 0; }
  .proposed .bdd .key { color: var(--accent); font-weight: 600; min-width: 48px; display: inline-block; }
  .proposed .bdd .key.given { color: #94a3b8; }
  .proposed .bdd .key.when  { color: #38bdf8; }
  .proposed .bdd .key.then  { color: #22c55e; }
  .proposed .validation-accepted { color: #22c55e; }
  .proposed .validation-rejected_redundant { color: #f59e0b; }
  .proposed .validation-rejected_off_spec  { color: #ef4444; }
  .proposed .validation-fallback_stub      { color: #94a3b8; }
  footer { padding: 8px 20px; border-top: 1px solid var(--border); color: var(--muted); font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>Kartograf</h1>
  <span class="subtitle" id="target-name"></span>
</header>
<div class="layout">
  <div id="map"></div>
  <aside>
    <div class="coverage-big">
      <span id="coverage-pct">—</span><span class="frac"> coverage</span>
    </div>
    <div class="stat"><span class="label">surfaces</span><span class="value" id="stat-surfaces">—</span></div>
    <div class="stat"><span class="label">evals</span><span class="value" id="stat-evals">—</span></div>
    <div class="stat"><span class="label">gap regions</span><span class="value" id="stat-regions">—</span></div>
    <div class="stat"><span class="label">α (mass-balance)</span><span class="value" id="stat-alpha">—</span></div>
    <div class="stat"><span class="label">bandwidth</span><span class="value" id="stat-bw">—</span></div>
    <h2>Top gap regions</h2>
    <div id="region-list"></div>
  </aside>
</div>
<footer>
  Surface points are claims the agent declares (tools, prompts, docstrings).
  Eval points are tests in the suite.
  Red regions are surface-dense, eval-empty — what the agent claims but no test exercises.
</footer>
<script>
const DATA = __PAYLOAD__;

document.getElementById('target-name').textContent = DATA.target;
document.getElementById('coverage-pct').textContent = (DATA.summary.coverage * 100).toFixed(1) + '%';
document.getElementById('stat-surfaces').textContent = DATA.summary.surface_count;
document.getElementById('stat-evals').textContent = DATA.summary.eval_count;
document.getElementById('stat-regions').textContent = DATA.summary.region_count;
document.getElementById('stat-alpha').textContent = DATA.summary.alpha.toFixed(2);
document.getElementById('stat-bw').textContent = DATA.summary.bandwidth.toFixed(3);

const KIND_COLOR = {
  prompt: '#a78bfa',
  tool: '#f59e0b',
  tool_arg: '#fbbf24',
  docstring: '#94a3b8',
};
const KIND_LABEL = {
  prompt: 'prompt',
  tool: 'tool',
  tool_arg: 'tool arg',
  docstring: 'docstring',
};

function truncate(text, n) {
  if (text.length <= n) return text;
  return text.slice(0, n - 1) + '…';
}

function makeSurfaceTrace(kind) {
  const items = DATA.surfaces.filter(s => s.kind === kind);
  return {
    type: 'scattergl',
    mode: 'markers',
    name: KIND_LABEL[kind] || kind,
    x: items.map(s => s.x),
    y: items.map(s => s.y),
    text: items.map(s => `<b>${KIND_LABEL[kind] || kind}</b><br>${truncate(s.text, 120).replace(/</g,'&lt;')}<br><i>${s.source}</i>`),
    hoverinfo: 'text',
    marker: { color: KIND_COLOR[kind] || '#888', size: 9, opacity: 0.85, line: { width: 0 } },
  };
}

const surfaceTraces = Object.keys(KIND_COLOR)
  .filter(k => DATA.surfaces.some(s => s.kind === k))
  .map(makeSurfaceTrace);

const evalTrace = {
  type: 'scattergl',
  mode: 'markers',
  name: 'eval case',
  x: DATA.evals.map(e => e.x),
  y: DATA.evals.map(e => e.y),
  text: DATA.evals.map(e => `<b>eval</b><br>${truncate(e.text, 120).replace(/</g,'&lt;')}<br><i>${e.source}</i>`),
  hoverinfo: 'text',
  marker: { color: '#38bdf8', size: 13, opacity: 0.95, symbol: 'diamond', line: { width: 1, color: '#0b0e14' } },
};

const heatTrace = {
  type: 'heatmap',
  x: DATA.gap_field.x,
  y: DATA.gap_field.y,
  z: DATA.gap_field.z,
  colorscale: [[0, 'rgba(11,14,20,0)'], [0.4, 'rgba(239,68,68,0.0)'], [0.7, 'rgba(239,68,68,0.35)'], [1, 'rgba(239,68,68,0.65)']],
  showscale: false,
  hoverinfo: 'skip',
  zsmooth: 'best',
};

const regionShapes = DATA.regions.slice(0, 5).map(r => ({
  type: 'rect',
  x0: r.bbox[0], y0: r.bbox[1], x1: r.bbox[2], y1: r.bbox[3],
  line: { color: '#ef4444', width: 1.5, dash: 'dot' },
  fillcolor: 'rgba(239,68,68,0.08)',
  layer: 'above',
}));

const regionAnnotations = DATA.regions.slice(0, 5).map(r => ({
  x: r.centroid[0], y: r.centroid[1],
  text: r.id,
  showarrow: false,
  font: { color: '#fff', size: 13, family: 'ui-sans-serif' },
  bgcolor: '#ef4444',
  bordercolor: '#7f1d1d',
  borderwidth: 1,
  borderpad: 3,
  opacity: 0.95,
}));

Plotly.newPlot('map', [heatTrace, ...surfaceTraces, evalTrace], {
  paper_bgcolor: '#131820',
  plot_bgcolor: '#0b0e14',
  font: { color: '#d7dde8', family: 'ui-sans-serif' },
  margin: { l: 40, r: 16, t: 16, b: 36 },
  xaxis: { title: 'UMAP-1', gridcolor: '#1f2735', zerolinecolor: '#1f2735' },
  yaxis: { title: 'UMAP-2', gridcolor: '#1f2735', zerolinecolor: '#1f2735' },
  legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0)', font: { size: 12 } },
  shapes: regionShapes,
  annotations: regionAnnotations,
  hoverlabel: { bgcolor: '#131820', bordercolor: '#232a36', font: { color: '#d7dde8' } },
}, { responsive: true, displaylogo: false });

const regionList = document.getElementById('region-list');
DATA.regions.forEach(region => {
  const wrap = document.createElement('div');
  wrap.className = 'region';
  const head = document.createElement('div');
  head.className = 'region-head';
  head.innerHTML = `<span class="region-id">${region.id}</span><span class="region-mass">mass ${region.mass.toFixed(3)}</span>`;
  wrap.appendChild(head);
  if (region.neighbors.length) {
    const ul = document.createElement('ul');
    ul.className = 'region-neighbors';
    region.neighbors.slice(0, 5).forEach(n => {
      const li = document.createElement('li');
      li.innerHTML = `<span class="kind-${n.kind}">${KIND_LABEL[n.kind] || n.kind}</span> · ${truncate(n.text, 90).replace(/</g,'&lt;')}`;
      ul.appendChild(li);
    });
    wrap.appendChild(ul);
  }
  region.proposed.forEach(p => {
    const div = document.createElement('div');
    div.className = 'proposed';
    const esc = (s) => (s || '').replace(/</g,'&lt;');
    const validation = p.validation || 'unknown';
    div.innerHTML = `
      <div class="label">
        <span>proposed eval (BDD)</span>
        <span class="validation-${validation}">${validation.replace('_',' ')}</span>
      </div>
      <div class="scenario">${esc(p.scenario || '(no scenario)')}</div>
      <div class="bdd">
        <span class="clause"><span class="key given">Given</span>${esc(p.given)}</span>
        <span class="clause"><span class="key when">When</span>${esc(p.when)}</span>
        <span class="clause"><span class="key then">Then</span>${esc(p.then)}</span>
      </div>
    `;
    wrap.appendChild(div);
  });
  regionList.appendChild(wrap);
});
</script>
</body>
</html>
"""
