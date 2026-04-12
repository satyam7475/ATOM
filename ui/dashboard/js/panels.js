import { escapeHtml } from './utils.js';

export function updateSystemPanel(data) {
  const el = document.getElementById('panel-system-content');
  if (!el || !data) return;

  const cpuPct = (data.cpu_percent || 0).toFixed(0);
  const ramPct = (data.ram_percent || 0).toFixed(0);
  const cpuColor = data.cpu_percent > 80 ? 'var(--red)' : data.cpu_percent > 60 ? 'var(--amber)' : '';
  const ramColor = data.ram_percent > 85 ? 'var(--red)' : data.ram_percent > 70 ? 'var(--amber)' : '';

  const batText = data.battery_percent != null
    ? `${data.battery_plugged ? 'AC' : 'BAT'} ${data.battery_percent.toFixed(0)}%`
    : 'N/A';

  el.innerHTML = `
    <div class="metrics-grid">
      <div class="metric">
        <span class="label">CPU</span>
        <div class="bar-bg"><div class="bar-fill" style="width:${cpuPct}%;${cpuColor ? 'background:' + cpuColor : ''}"></div></div>
        <span class="val">${cpuPct}%</span>
      </div>
      <div class="metric">
        <span class="label">RAM</span>
        <div class="bar-bg"><div class="bar-fill" style="width:${ramPct}%;${ramColor ? 'background:' + ramColor : ''}"></div></div>
        <span class="val">${ramPct}%</span>
      </div>
      <div class="metric" style="margin-top:6px">
        <span class="label">TEMP</span>
        <span class="val-text">${data.cpu_temp ? data.cpu_temp.toFixed(1) + '\u00B0C' : 'N/A'}</span>
      </div>
      <div class="metric">
        <span class="label">PWR</span>
        <span class="val-text">${batText}</span>
      </div>
    </div>
  `;
}

export function updateBrainPath(pathName, latencyMs) {
  const el = document.getElementById('panel-brainpath-content');
  if (!el) return;
  el.innerHTML = `
    <div class="path-indicator">
      <span class="path-name">${escapeHtml(pathName)}</span>
      <span class="path-latency">${latencyMs}ms</span>
    </div>
  `;
}

export function updateCognitivePanel(goals, predictions) {
  const el = document.getElementById('panel-cognitive-content');
  if (!el) return;

  let html = '<div class="cog-section"><h5>PRIORITY GOAL</h5>';
  if (goals && goals.length > 0) {
    html += `<div class="goal-item">${escapeHtml(goals[0].name || goals[0].text || String(goals[0]))}</div>`;
  } else {
    html += '<div class="empty-state">No active goals</div>';
  }

  html += '</div><div class="cog-section"><h5>PREDICTION</h5>';
  if (predictions && predictions.length > 0) {
    const p = predictions[0];
    const conf = typeof p.confidence === 'number' ? (p.confidence * 100).toFixed(0) : '?';
    html += `<div class="pred-item">${escapeHtml(p.text || String(p))} <span class="conf">${conf}%</span></div>`;
  } else {
    html += '<div class="empty-state">No predictions available.</div>';
  }

  html += '</div>';
  el.innerHTML = html;
}

export function updateMemoryPanel(nodes) {
  const el = document.getElementById('panel-memory-content');
  if (!el) return;
  if (!nodes || nodes.length === 0) {
    el.innerHTML = '<div class="empty-state">Memory context clear.</div>';
    return;
  }

  let html = '<ul class="memory-list">';
  for (const node of nodes.slice(0, 4)) {
    const text = typeof node === 'string' ? node : (node.content || node.text || JSON.stringify(node));
    const preview = text.length > 50 ? text.substring(0, 50) + '\u2026' : text;
    html += `<li>${escapeHtml(preview)}</li>`;
  }
  html += '</ul>';
  el.innerHTML = html;
}

export function updateAutonomyLog(logs) {
  const el = document.getElementById('panel-autonomy-content');
  if (!el) return;

  if (!logs || logs.length === 0) {
    el.innerHTML = '<div class="empty-state">No autonomous actions yet.</div>';
    return;
  }

  let html = '<div class="autonomy-list">';
  for (const log of logs.slice(0, 4)) {
    html += `
      <div class="auto-item">
        <div class="auto-action">${escapeHtml(log.action || '')}</div>
        <div class="auto-reason">${escapeHtml(log.reason || '')}</div>
      </div>
    `;
  }
  html += '</div>';
  el.innerHTML = html;
}
