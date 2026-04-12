import { addLog } from './conversation.js';
import { updateSystemPanel, updateCognitivePanel, updateAutonomyLog, updateMemoryPanel, updateBrainPath } from './panels.js';
import { initOrb } from './orb.js';

// ─── DOM REFS ───
const body = document.body;
const stateDot = document.getElementById('state-dot');
const stateBadge = document.getElementById('state-badge');
const orbLabel = document.getElementById('orb-label');
const orbSublabel = document.getElementById('orb-sublabel');
const hearingStrip = document.getElementById('hearing-strip');
const micBarFill = document.getElementById('mic-bar-fill');
const micNameEl = document.getElementById('mic-name');
const statusTextEl = document.getElementById('status-text');

// ─── STATE CONFIG ───
const STATE_CONFIG = {
  sleep:          { color: '#4a5568', glow: '#1a1d24',  accent: '#6B7280', label: 'SLEEP',     sub: 'Inactive' },
  idle:           { color: '#38bdf8', glow: '#0c2d4a',  accent: '#7dd3fc', label: 'IDLE',      sub: 'Ready' },
  listening:      { color: '#00e5c3', glow: '#003d32',  accent: '#00ffd5', label: 'LISTENING', sub: 'Listening...' },
  thinking:       { color: '#ffb86c', glow: '#3d2d00',  accent: '#ffa03e', label: 'THINKING',  sub: 'Processing...' },
  speaking:       { color: '#b48efa', glow: '#2a1f50',  accent: '#c4b5fd', label: 'SPEAKING',  sub: 'Speaking...' },
  error_recovery: { color: '#ff5555', glow: '#3d1a1a',  accent: '#ff8a8a', label: 'RECOVERY', sub: 'Recovering...' },
};

// ─── TITLE BAR EVENTS ───
document.getElementById('btn-close').addEventListener('click', () => {
  if (window.webkit?.messageHandlers?.atomBridge) {
    window.webkit.messageHandlers.atomBridge.postMessage({ action: 'shutdown' });
  }
});

document.getElementById('btn-minimize').addEventListener('click', () => {
  if (window.webkit?.messageHandlers?.atomBridge) {
    window.webkit.messageHandlers.atomBridge.postMessage({ action: 'minimize' });
  }
});

// ─── EXPOSED GLOBAL API FOR NATIVE BRIDGE ───
window.updateState = function(stateValue) {
  const cfg = STATE_CONFIG[stateValue] || STATE_CONFIG.sleep;
  body.setAttribute('data-state', stateValue);

  const root = document.documentElement;
  root.style.setProperty('--state-color', cfg.color);
  root.style.setProperty('--state-glow', cfg.glow);
  root.style.setProperty('--state-accent', cfg.accent);

  stateDot.style.background = cfg.color;
  stateBadge.textContent = cfg.label;
  stateBadge.style.color = cfg.color;
  orbLabel.textContent = cfg.label;
  orbLabel.style.color = cfg.color;
  orbSublabel.textContent = cfg.sub;

  const core = document.getElementById('orb-core');
  if (core) {
    core.style.background = `radial-gradient(circle at 38% 32%, ${cfg.color} 0%, ${cfg.glow} 100%)`;
    core.style.boxShadow = `0 0 35px ${cfg.color}, 0 0 70px ${cfg.glow}, inset 0 0 24px rgba(255,255,255,0.06)`;
  }

  const rc = document.getElementById('reactor-center');
  if (rc) {
    rc.style.background = cfg.color;
    rc.style.boxShadow = `0 0 14px ${cfg.color}`;
  }
};

window.addLog = function(tag, message) {
  addLog(tag, message);
};

window.showHearing = function(text) {
  hearingStrip.textContent = '\u203A ' + text;
  hearingStrip.classList.add('active');
};

window.clearHearing = function() {
  hearingStrip.classList.remove('active');
  hearingStrip.textContent = '';
};

window.updateMicLevel = function(value) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  micBarFill.style.width = pct + '%';
};

window.setMicName = function(name) {
  micNameEl.textContent = 'Voice I/O: ' + name;
};

window.setStatusText = function(text) {
  statusTextEl.textContent = text;
};

// ─── INTEL PANEL UPDATES FROM PYTHON ───
window.updateSystemStats = function(dataStr) {
  try { updateSystemPanel(JSON.parse(dataStr)); } catch(e) {}
};

window.updateGoals = function(dataStr) {
  try {
    const p = JSON.parse(dataStr);
    updateCognitivePanel(p.goals || [], p.predictions || [], p.mode || "default");
  } catch(e) {}
};

window.updateMemorySnapshot = function(dataStr) {
  try { updateMemoryPanel(JSON.parse(dataStr)); } catch(e) {}
};

window.updateAutonomy = function(dataStr) {
  try { updateAutonomyLog(JSON.parse(dataStr)); } catch(e) {}
};

window.updateBrain = function(pathName, latencyMs) {
  try { updateBrainPath(pathName, latencyMs); } catch(e) {}
};

// ─── STARTUP ───
window.addEventListener('DOMContentLoaded', () => {
  initOrb();
  window.updateState('sleep');
});
