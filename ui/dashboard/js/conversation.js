import { escapeHtml, formatTime } from './utils.js';

const convScroll = document.getElementById('conv-scroll');
const MAX_LOG_ENTRIES = 120;
const TRIM_COUNT = 30;
let logCount = 0;

export function addLog(tag, message) {
  const ts = formatTime(new Date());
  const entry = document.createElement('div');
  entry.className = 'log-entry';

  let html = `<span class="log-ts">${ts}</span>`;

  switch (tag) {
    case 'heard':
      html += `<span class="log-prefix-user">USER \u27E9 </span><span class="log-msg-user">${escapeHtml(message)}</span>`;
      break;

    case 'action':
    case 'speaking':
      if (message.startsWith('[stream]')) {
        const clean = message.substring(8).trim();
        if (!clean) return;
        html += `<span class="log-prefix-atom">ATOM \u27E9 </span><span class="log-msg-atom">${escapeHtml(clean)}</span>`;
      } else if (message.startsWith('Thinking with local brain')) {
        html += `<span class="log-thinking">\u27E1 ATOM is thinking\u2026</span>`;
      } else if (message.startsWith('Running:')) {
        html += `<span class="log-info">[CMD] </span><span class="log-system">${escapeHtml(message)}</span>`;
      } else {
        html += `<span class="log-prefix-atom">ATOM \u27E9 </span><span class="log-msg-atom">${escapeHtml(message)}</span>`;
      }
      break;

    case 'intent':
      html += `<span class="log-info">[INTENT] </span><span class="log-system">${escapeHtml(message)}</span>`;
      break;

    case 'error':
      html += `<span class="log-error">[ERROR] ${escapeHtml(message)}</span>`;
      break;

    case 'info':
      html += `<span class="log-system">[INFO] ${escapeHtml(message)}</span>`;
      break;

    case 'warning':
      html += `<span class="log-thinking">[WARN] ${escapeHtml(message)}</span>`;
      break;

    case 'reminder':
      html += `<span class="log-info">[REMINDER] </span><span class="log-msg-atom">${escapeHtml(message)}</span>`;
      break;

    default:
      html += `<span class="log-system">[${tag.toUpperCase()}] ${escapeHtml(message)}</span>`;
  }

  entry.innerHTML = html;
  convScroll.appendChild(entry);
  logCount++;

  if (logCount > MAX_LOG_ENTRIES) {
    const children = convScroll.children;
    for (let i = 0; i < TRIM_COUNT && children.length > 0; i++) {
      convScroll.removeChild(children[0]);
    }
    logCount -= TRIM_COUNT;
  }

  convScroll.scrollTop = convScroll.scrollHeight;
}
