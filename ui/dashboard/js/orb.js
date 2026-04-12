export function initOrb() {
  const CX = 120, CY = 120;
  const ORB_R = 46;
  const RING_COUNT = 7;
  const PARTICLE_COUNT = 22;
  const ARC_COUNT = 5;

  const ringsGroup = document.getElementById('rings-group');
  const arcsGroup = document.getElementById('arcs-group');
  const particlesGroup = document.getElementById('particles-group');

  if (!ringsGroup) return;

  const rings = [];
  for (let i = 0; i < RING_COUNT; i++) {
    const r = ORB_R + 8 + i * 6.5;
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', CX);
    circle.setAttribute('cy', CY);
    circle.setAttribute('r', r);
    circle.classList.add('ring');
    circle.style.strokeWidth = String(Math.max(0.4, 1.6 - i * 0.18));
    ringsGroup.appendChild(circle);
    rings.push({ el: circle, baseR: r, index: i });
  }

  const arcs = [];
  for (let i = 0; i < ARC_COUNT; i++) {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.classList.add('reactor-arc');
    path.style.strokeWidth = '1.5';
    arcsGroup.appendChild(path);
    arcs.push({ el: path, index: i });
  }

  const particles = [];
  for (let j = 0; j < PARTICLE_COUNT; j++) {
    const angle = Math.random() * Math.PI * 2;
    const dist = ORB_R + 12 + Math.random() * 44;
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    const size = 0.8 + Math.random() * 2.0;
    circle.setAttribute('r', String(size));
    circle.classList.add('particle');
    particlesGroup.appendChild(circle);
    particles.push({
      el: circle, angle, dist, size,
      speed: 0.002 + Math.random() * 0.02,
      orbit: j % 4,
    });
  }

  function arcPath(cx, cy, r, startDeg, endDeg) {
    const toRad = d => d * Math.PI / 180;
    const x1 = cx + r * Math.cos(toRad(startDeg));
    const y1 = cy + r * Math.sin(toRad(startDeg));
    const x2 = cx + r * Math.cos(toRad(endDeg));
    const y2 = cy + r * Math.sin(toRad(endDeg));
    const large = (endDeg - startDeg > 180) ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  }

  let animAngle = 0;

  const SPEED_MAP = {
    thinking: 3.2,
    speaking: 1.8,
    listening: 1.0,
    idle: 0.35,
    error_recovery: 2.5,
    sleep: 0.12,
  };

  function getSpeed(state) {
    return SPEED_MAP[state] || 0.12;
  }

  function breathe(phase, state) {
    switch (state) {
      case 'listening':      return 1.0 + 0.06 * Math.sin(phase * 1.0);
      case 'thinking':       return 1.0 + 0.12 * Math.sin(phase * 2.2);
      case 'speaking':       return 1.0 + 0.04 * Math.sin(phase * 1.4);
      case 'error_recovery': return 1.0 + 0.10 * Math.sin(phase * 3.0);
      default:               return 1.0 + 0.008 * Math.sin(phase * 0.25);
    }
  }

  function animate() {
    const state = document.body.getAttribute('data-state') || 'sleep';
    const speed = getSpeed(state);
    animAngle += 0.035 * speed;
    const a = animAngle;

    for (const ring of rings) {
      const phase = a + ring.index * 0.55;
      const b = breathe(phase, state);
      ring.el.setAttribute('r', String(ring.baseR * b));
      const intensity = Math.max(0.1, 0.5 - ring.index * 0.06 + 0.18 * Math.sin(phase));
      ring.el.style.opacity = String(intensity);
      ring.el.style.strokeWidth = String(Math.max(0.35, 1.5 - ring.index * 0.17 + 0.25 * Math.sin(phase)));
    }

    for (const p of particles) {
      p.angle += p.speed * speed;
      const wobble = 1.0 + 0.12 * Math.sin(a * (1.3 + p.orbit * 0.35) + p.angle * 2);
      const d = p.dist * wobble;
      p.el.setAttribute('cx', String(CX + d * Math.cos(p.angle)));
      p.el.setAttribute('cy', String(CY + d * Math.sin(p.angle)));
      const pulse = 0.65 + 0.55 * Math.sin(a * 1.4 + p.angle * 1.1);
      p.el.setAttribute('r', String(p.size * pulse));
      p.el.style.opacity = String(Math.max(0.08, 0.25 + 0.5 * Math.sin(a * 1.1 + p.angle)));
    }

    const arcSpeed = SPEED_MAP[state] || 0.1;
    const arcOffset = (a * arcSpeed * 55) % 360;
    const arcR = ORB_R * 0.52;
    for (const arc of arcs) {
      const gap = 360 / ARC_COUNT;
      const start = arc.index * gap + 8 + arcOffset;
      const end = start + gap * 0.7;
      arc.el.setAttribute('d', arcPath(CX, CY, arcR, start, end));
      arc.el.style.opacity = String(0.18 + 0.3 * Math.sin(a * 1.2 + arc.index * 1.26));
    }

    const rc = document.getElementById('reactor-center');
    if (rc) {
      const sz = 3 + 2.2 * Math.sin(a * 1.8);
      rc.style.width = (sz * 2) + 'px';
      rc.style.height = (sz * 2) + 'px';
    }

    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
}
