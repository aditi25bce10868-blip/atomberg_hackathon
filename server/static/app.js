/* Efficio Apex dashboard. Talks only to /ws/live and /api/* on the same origin, so it works from any device.
   Every number comes from the backend; nothing is invented here. */
(() => {
  'use strict';

  /* ------------------------------------------------------------------ config you may edit */
  const SENSORS = [   // key matches the backend row; color is a CSS variable from style.css
    { key: 'mq135',    name: 'MQ-135',      sub: 'air quality',  dec: 2, unit: '',   icon: 'bi-wind',             color: '--s-mq135' },
    { key: 'mq2',      name: 'MQ-2',        sub: 'smoke / gas',  dec: 2, unit: '',   icon: 'bi-fire',             color: '--s-mq2' },
    { key: 'humidity', name: 'Humidity',    sub: '',             dec: 1, unit: '%',  icon: 'bi-droplet-half',     color: '--s-humidity' },
    { key: 'temp_c',   name: 'Temperature', sub: '',             dec: 1, unit: '°C', icon: 'bi-thermometer-half', color: '--s-temp_c' },
    { key: 'pir',      name: 'PIR',         sub: 'presence',     dec: 0, unit: '',   icon: 'bi-person-walking',   color: '--s-pir', binary: true },
  ];
  const CHARTED = SENSORS.filter(s => !s.binary);
  const LABELS = ['clean', 'steam', 'cooking', 'foul'];
  const LABEL_ICON = { clean: 'bi-wind', steam: 'bi-cloud-haze2', cooking: 'bi-fire', foul: 'bi-emoji-dizzy' };
  const STATE_TEXT = {
    'WARMUP':  'Sensors warming up, learning the baseline',
    'IDLE':    'Air is clear, watching',
    'RUNNING': 'Air is dirty, fan running',
    'RUN-ON':  'Air is clear, fan running on to finish the job',
    'FAULT':   'Sensor fault, automation is off',
    'MANUAL':  'Wall switch is ON, manual control',
  };
  const END_TEXT = { clear: 'air cleared', cap: 'hard cap', fault: 'sensor fault', override: 'wall switch' };
  const KEEP_S = 600;          // seconds of history kept in the browser

  /* ------------------------------------------------------------------ helpers */
  const $  = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const isNum = v => v != null && v !== '' && Number.isFinite(+v);
  const num = (v, d = 2) => isNum(v) ? (+v).toFixed(d) : '—';
  const signed = (v, d = 2) => { if (!isNum(v)) return '—'; const r = +(+v).toFixed(d); return (r > 0 ? '+' : r < 0 ? '−' : '') + Math.abs(r).toFixed(d); };
  const pct = (v, d = 0) => isNum(v) ? (100 * v).toFixed(d) + '%' : '—';
  const clock = ts => isNum(ts) ? new Date(ts * 1000).toLocaleTimeString([], { hour12: false }) : '—';
  const mmss = s => { if (!isNum(s)) return '—'; s = Math.max(0, Math.round(s)); return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0'); };
  const stateKey = s => (s || '').toLowerCase().replace(/[^a-z]/g, '');            // RUN-ON -> runon
  const hexA = (c, a) => {                                                          // "#rrggbb" -> rgba()
    const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(c.trim());
    return m ? `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})` : c;
  };
  const api = (url, opt) => fetch(url, opt).then(r => { if (!r.ok) throw new Error(url + ' ' + r.status); return r.json(); });
  const clear = n => { while (n.firstChild) n.removeChild(n.firstChild); return n; };

  /* ------------------------------------------------------------------ state */
  const S = {
    cfg: null, row: null, stale: true, ageS: null,
    pts: [], lastTs: null, windowS: 300,
    wsOk: false, lastMsgAt: 0, reconnects: 0, lastWsAt: null,
    events: [], runs: [],
    collectFile: null, collectStartTs: null,
    health: null,
  };

  /* ------------------------------------------------------------------ tabs (hash routing, works with the phone back button) */
  const charts = {};
  function showTab(name) {
    if (!$('#pane-' + name)) name = 'live';
    $$('.pane').forEach(p => p.classList.toggle('active', p.id === 'pane-' + name));
    $$('.tab-link').forEach(a => { const on = a.dataset.tab === name; a.classList.toggle('active', on); on ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current'); });
    window.scrollTo(0, 0);
    if (name === 'ml') loadReport();
    if (name === 'runs') { loadRuns(); loadEvents(); }
    if (name === 'system') { loadHealth(); loadConfig(); renderSystem(); }
    requestAnimationFrame(() => Object.values(charts).forEach(c => c.resize()));
    renderCharts();
  }
  window.addEventListener('hashchange', () => showTab(location.hash.slice(1)));

  /* ------------------------------------------------------------------ build repeated pieces */
  function buildStatCards() {
    const host = $('#statCards');
    SENSORS.forEach(s => {
      const col = el('div', 'col-6 col-md-4 col-xl');
      const card = el('div', 'panel stat'); card.id = 'stat-' + s.key; card.style.setProperty('--c', `var(${s.color})`);
      const name = el('div', 'name');
      const left = el('span'); left.append(el('i', 'bi ' + s.icon), document.createTextNode(s.name));
      name.append(left, el('span', 'hdot'));
      const big = el('div', 'big num'); big.append(el('span', 'v', '—'), el('small', 'u', s.unit ? ' ' + s.unit : ''));
      const meta = el('div', 'meta');
      if (s.binary) { meta.append(el('span', null, s.sub), el('span', 'rise')); }
      else {
        const b = el('span'); b.append('base ', el('b', 'base num', '—'));
        const r = el('span'); r.append('rise ', el('b', 'rise num', '—'));
        meta.append(b, r);
      }
      card.append(name, big, meta);
      if (!s.binary) { const bar = el('div', 'bar'); bar.append(el('i')); card.append(bar); }
      col.append(card); host.append(col);
    });
  }

  function buildChartCards() {
    const host = $('#chartCards');
    CHARTED.forEach(s => {
      const col = el('div', 'col-12 col-lg-6');
      const p = el('div', 'panel');
      const t = el('h2', 'panel-title'); t.append(el('span', null, s.name + (s.sub ? ' · ' + s.sub : '')), el('span', 'num', s.unit));
      const box = el('div', 'chart-box'); const cv = el('canvas'); cv.id = 'ch-' + s.key; box.append(cv);
      p.append(t, box); col.append(p); host.append(col);
    });
  }

  function buildProbBars(host, compact) {
    clear(host);
    LABELS.forEach(l => {
      const r = el('div', 'prob l-' + l); r.dataset.label = l;
      const track = el('div', 'track'); track.append(el('i'));
      r.append(el('span', 'nm', l), track, el('span', 'pc', '—'));
      host.append(r);
    });
  }
  function fillProbBars(host, probs, top) {
    $$('.prob', host).forEach(r => {
      const v = probs ? probs[r.dataset.label] : null;
      $('i', r).style.width = isNum(v) ? (100 * v).toFixed(1) + '%' : '0%';
      $('.pc', r).textContent = isNum(v) ? pct(v, 0) : '—';
      r.classList.toggle('top', r.dataset.label === top);
    });
  }

  function buildSensorTable() {
    const tb = $('#sensorTable tbody');
    SENSORS.forEach(s => {
      const tr = el('tr'); tr.id = 'srow-' + s.key; tr.style.setProperty('--c', `var(${s.color})`);
      const n = el('td'); n.append(el('i', 'bi ' + s.icon + ' me-2'), document.createTextNode(s.name));
      n.firstChild.style.color = `var(${s.color})`;
      const bar = el('div', 'bar cell-bar'); bar.append(el('i'));
      const pc = el('td'); pc.append(bar);
      tr.append(n, el('td', 'num v'), el('td', 'num b'), el('td', 'num r'), el('td', 'num t'), pc, el('td', 'h'));
      tb.append(tr);
    });
  }

  function buildLabelPick() {
    const host = $('#labelPick');
    LABELS.forEach((l, i) => {
      const id = 'lp-' + l;
      const inp = el('input'); inp.type = 'radio'; inp.name = 'collectLabel'; inp.id = id; inp.value = l; inp.checked = i === 0;
      const lab = el('label', 'l-' + l); lab.htmlFor = id; lab.append(el('i', 'bi ' + LABEL_ICON[l]), document.createTextNode(l));
      host.append(inp, lab);
    });
  }

  /* ------------------------------------------------------------------ charts */
  // paints the fan-running / run-on / fault periods behind the lines
  const bandsPlugin = {
    id: 'bands',
    beforeDatasetsDraw(chart) {
      const st = chart.$states, x = chart.scales.x, a = chart.chartArea;
      if (!st || !st.length || !x) return;
      const col = { 'RUNNING': css('--running'), 'RUN-ON': css('--runon'), 'FAULT': css('--fault') };
      const step = st.length > 1 ? (x.getPixelForValue(st.length - 1) - x.getPixelForValue(0)) / (st.length - 1) : 0;
      const ctx = chart.ctx; ctx.save();
      for (let i = 0; i < st.length;) {
        const s = st[i];
        if (!col[s]) { i++; continue; }
        let j = i; while (j + 1 < st.length && st[j + 1] === s) j++;
        ctx.fillStyle = hexA(col[s], 0.16);
        const x0 = x.getPixelForValue(i) - step / 2, x1 = x.getPixelForValue(j) + step / 2;
        ctx.fillRect(Math.max(a.left, x0), a.top, Math.min(a.right, x1) - Math.max(a.left, x0), a.height);
        i = j + 1;
      }
      ctx.restore();
    },
  };

  function chartOptions(dec, yMax) {
    const muted = css('--muted'), line = hexA(css('--line'), 0.9);
    return {
      responsive: true, maintainAspectRatio: false, animation: false, spanGaps: false,
      interaction: { mode: 'index', intersect: false },
      elements: { point: { radius: 0, hitRadius: 8 }, line: { borderWidth: 2, tension: 0.25 } },
      scales: {
        x: { ticks: { color: muted, maxTicksLimit: 6, maxRotation: 0, font: { size: 10 } }, grid: { color: line } },
        y: { ticks: { color: muted, font: { size: 10 } }, grid: { color: line }, grace: yMax ? 0 : '8%', ...(yMax ? { min: 0, max: yMax } : {}) },
      },
      plugins: {
        legend: { labels: { color: muted, boxWidth: 22, boxHeight: 2, font: { size: 11 } } },
        tooltip: { callbacks: { label: c => ` ${c.dataset.label}: ${isNum(c.parsed.y) ? c.parsed.y.toFixed(dec) : '—'}` } },
      },
    };
  }

  function buildCharts() {
    Chart.defaults.font.family = css('--font-ui');
    CHARTED.forEach(s => {
      const c = css(s.color), faint = hexA(css('--text'), 0.38), trig = hexA(css('--text'), 0.7);
      charts[s.key] = new Chart($('#ch-' + s.key), {
        type: 'line',
        data: {
          labels: [], datasets: [
            { label: s.name, data: [], borderColor: c, backgroundColor: c, order: 1 },
            { label: 'baseline', data: [], borderColor: faint, backgroundColor: faint, borderDash: [6, 4], borderWidth: 1.5, order: 2 },
            { label: 'trigger level', data: [], borderColor: trig, backgroundColor: trig, borderDash: [2, 3], borderWidth: 1.5, order: 3 },
          ],
        },
        options: chartOptions(s.dec),
        plugins: [bandsPlugin],
      });
    });
    charts.prob = new Chart($('#probChart'), {
      type: 'line',
      data: { labels: [], datasets: LABELS.map(l => ({ label: l, data: [], borderColor: css('--c-' + l), backgroundColor: css('--c-' + l) })) },
      options: chartOptions(2, 1),
      plugins: [bandsPlugin],
    });
  }

  const windowPts = () => { const cut = (S.lastTs ?? 0) - S.windowS; return S.pts.filter(p => p.ts >= cut); };

  function renderCharts() {
    if (!charts.prob || !S.pts.length) return;
    const pts = windowPts(), labels = pts.map(p => clock(p.ts)), states = pts.map(p => p.state);
    const thr = (S.cfg && S.cfg.thresholds) || {};
    CHARTED.forEach(s => {
      const c = charts[s.key], d = c.data.datasets;
      c.data.labels = labels; c.$states = states;
      d[0].data = pts.map(p => isNum(p.sensors[s.key]) ? +p.sensors[s.key] : null);
      d[1].data = pts.map(p => isNum(p.baseline[s.key]) ? +p.baseline[s.key] : null);
      d[2].data = pts.map(p => isNum(p.baseline[s.key]) && isNum(thr[s.key]) ? +p.baseline[s.key] + +thr[s.key] : null);
      c.update('none');
    });
    const pc = charts.prob;
    pc.data.labels = labels; pc.$states = states;
    LABELS.forEach((l, i) => { pc.data.datasets[i].data = pts.map(p => p.probs && isNum(p.probs[l]) ? +p.probs[l] : null); });
    pc.update('none');
  }

  /* ------------------------------------------------------------------ incoming rows */
  function pushRow(r) {
    if (!r || !isNum(r.ts) || r.ts === S.lastTs) return false;
    S.pts.push({ ts: r.ts, sensors: r.sensors || {}, baseline: r.baseline || {}, state: r.state, probs: r.ml && r.ml.probs });
    S.lastTs = r.ts;
    while (S.pts.length && S.pts[0].ts < S.lastTs - KEEP_S) S.pts.shift();
    return true;
  }

  function handle(msg) {
    S.lastMsgAt = Date.now();
    S.stale = !!msg.stale; S.ageS = msg.age_s;
    S.row = msg.row || null;
    const fresh = S.row ? pushRow(S.row) : false;
    renderTop(); renderHero(); renderStats(); renderClassifier(); renderSensorTable(); renderCollect();
    if (fresh) renderCharts();
    renderLink();
  }

  /* ------------------------------------------------------------------ rendering: top bar + live */
  function renderTop() {
    const r = S.row, badge = $('#modeBadge');
    const mode = r ? r.mode : null;
    $('#modeText').textContent = mode || 'NO DATA';
    badge.className = 'mode-badge' + (mode ? ' mode-' + mode.toLowerCase() : '');
    $('#staleChip').classList.toggle('d-none', !(r && S.stale));
    const pill = $('#statePill');
    $('#stateText').textContent = r ? r.state : '—';
    pill.className = 'state-pill' + (r ? ' st-' + stateKey(r.state) : '');
    const fan = $('#fanPill');
    const on = !!(r && r.fan_on);
    fan.classList.toggle('on', on);
    $('#fanText').textContent = !r ? 'FAN —' : on ? (r.override ? 'FAN ON · wall switch' : 'FAN ON') : 'FAN OFF';
    $('#demoChip').classList.toggle('d-none', !(r && r.demo_timers));
  }

  function chip(text, cls, icon) { const c = el('span', 'chip ' + (cls || '')); if (icon) c.append(el('i', 'bi ' + icon)); c.append(text); return c; }

  function renderHero() {
    const r = S.row, hero = $('#heroState');
    hero.className = 'panel' + (r ? ' st-' + stateKey(r.state) : '');
    $('#heroStateText').textContent = r ? r.state : '—';
    $('#heroStateSub').textContent = r ? (STATE_TEXT[r.state] || '') : 'Waiting for data…';
    $('#heroAge').textContent = r && isNum(S.ageS) ? S.ageS + ' s ago' : '';
    const chips = clear($('#heroChips'));
    if (r) {
      chips.append(chip(r.fan_on ? 'Fan ON' : 'Fan off', r.fan_on ? 'chip-warn' : '', 'bi-fan'));
      chips.append(chip(r.relay_on ? 'Relay ON' : 'Relay off', r.relay_on ? 'chip-warn' : '', 'bi-toggle-' + (r.relay_on ? 'on' : 'off')));
      if (r.override) chips.append(chip('Wall switch', '', 'bi-hand-index'));
      if (r.fault_led) chips.append(chip('Fault LED', 'chip-bad', 'bi-exclamation-octagon'));
      if (r.collecting && r.collecting.active) chips.append(chip('Recording ' + r.collecting.label, 'chip-bad', 'bi-record-fill'));
    }
    renderTimers();
  }

  function renderTimers() {
    const host = $('#timers'), r = S.row, T = (S.cfg && S.cfg.timers) || {};
    const defs = [
      { k: 'warmup_left', total: T.warmup_s, name: 'Warm-up left', st: 'warmup' },
      { k: 'run_on_left', total: T.run_on_s,  name: 'Run-on left',  st: 'runon' },
      { k: 'cap_left',    total: T.cap_s,     name: 'Hard cap left', st: 'running' },
    ];
    if (!host.children.length) defs.forEach(d => {
      const row = el('div', 'timer-row'); row.dataset.k = d.k; row.style.setProperty('--c', `var(--${d.st})`);
      const head = el('div', 'head'); head.append(el('span', null, d.name), el('span', 'val', '—'));
      const bar = el('div', 'bar'); bar.append(el('i'));
      row.append(head, bar); host.append(row);
    });
    const tm = r && r.timers || {};
    defs.forEach(d => {
      const row = $(`[data-k="${d.k}"]`, host), v = tm[d.k];
      const active = isNum(v) && v > 0 || (d.k === 'cap_left' && isNum(v));
      row.classList.toggle('idle-timer', !active);
      $('.val', row).textContent = active ? mmss(v) : '—';
      $('.bar > i', row).style.width = active && isNum(d.total) && d.total > 0 ? Math.min(100, 100 * v / d.total).toFixed(1) + '%' : '0%';
    });
    $('#timersNote').classList.toggle('d-none', !(r && r.demo_timers));
  }

  function renderStats() {
    const r = S.row, thr = (S.cfg && S.cfg.thresholds) || {};
    SENSORS.forEach(s => {
      const card = $('#stat-' + s.key), v = r ? r.sensors[s.key] : null;
      const health = r && r.health ? r.health[s.key] : null, bad = r && health && health !== 'ok';
      $('.hdot', card).classList.toggle('bad', !!bad); $('.hdot', card).title = health || '';
      card.classList.toggle('faulty', !!bad);
      if (s.binary) { $('.v', card).textContent = !r || v == null ? '—' : (+v ? 'Present' : 'None'); $('.rise', card).textContent = ''; return; }
      $('.v', card).textContent = bad && !isNum(v) ? 'FAULT' : num(v, s.dec);
      const base = r && r.baseline ? r.baseline[s.key] : null, d = r && r.deltas ? r.deltas[s.key] : null;
      $('.base', card).textContent = num(base, s.dec);
      $('.rise', card).textContent = signed(d, s.dec);
      const t = thr[s.key], over = isNum(d) && isNum(t) && d >= t;
      card.classList.toggle('over', over);
      $('.bar > i', card).style.width = isNum(d) && isNum(t) && t > 0 ? Math.max(0, Math.min(100, 100 * d / t)).toFixed(0) + '%' : '0%';
    });
  }

  function renderEventList(host, list, limit) {
    clear(host);
    if (!list.length) { const li = el('li', 'empty', 'No events yet.'); li.style.display = 'block'; host.append(li); return; }
    list.slice(0, limit).forEach(e => {
      const li = el('li');
      li.append(el('span', 't', clock(e.ts)), el('span', 'lvl lvl-' + e.level, e.level), el('span', null, e.msg));
      host.append(li);
    });
  }

  /* ------------------------------------------------------------------ classifier */
  function renderClassifier() {
    const r = S.row, ml = r && r.ml;
    const lab = ml ? ml.label : null;
    // mini card on Live
    const ml1 = $('#miniLabel'); ml1.textContent = lab || '—'; ml1.style.textTransform = 'capitalize'; ml1.style.color = lab ? css('--c-' + lab) : '';
    $('#miniConf').textContent = ml ? pct(ml.confidence, 0) : '';
    fillProbBars($('#miniProbs'), ml && ml.probs, lab);
    // full tab
    const big = $('#clfLabel'); big.textContent = lab || '—'; big.style.color = lab ? css('--c-' + lab) : '';
    $('#clfConf').textContent = ml ? pct(ml.confidence, 1) + ' confident' : '';
    fillProbBars($('#clfProbs'), ml && ml.probs, lab);
    $('#clfSource').textContent = ml ? 'source: ' + ml.source : '';
    const note = clear($('#clfNote'));
    if (ml && String(ml.source).startsWith('rules')) note.append(chip('rules placeholder, no trained model loaded', 'chip-warn', 'bi-info-circle'));
    const truth = $('#clfTruth');
    if (r && r.truth) {
      const ok = r.truth === lab;
      truth.textContent = `Simulator ground truth: ${r.truth} · ${ok ? 'prediction matches' : 'prediction differs'}`;
      truth.style.color = ok ? css('--idle') : css('--running');
    } else truth.textContent = '';
  }

  /* ------------------------------------------------------------------ sensors tab */
  function renderSensorTable() {
    const r = S.row, thr = (S.cfg && S.cfg.thresholds) || {};
    SENSORS.forEach(s => {
      const tr = $('#srow-' + s.key), v = r ? r.sensors[s.key] : null;
      const health = r && r.health ? r.health[s.key] : null;
      if (s.binary) {
        $('.v', tr).textContent = !r || v == null ? '—' : (+v ? 'present' : 'none');
        ['.b', '.r', '.t'].forEach(k => { $(k, tr).textContent = '—'; });
        $('.bar > i', tr).style.width = '0%';
      } else {
        const b = r && r.baseline ? r.baseline[s.key] : null, d = r && r.deltas ? r.deltas[s.key] : null, t = thr[s.key];
        $('.v', tr).textContent = num(v, s.dec) + (s.unit && isNum(v) ? ' ' + s.unit : '');
        $('.b', tr).textContent = num(b, s.dec);
        $('.r', tr).textContent = signed(d, s.dec);
        $('.t', tr).textContent = isNum(t) ? '+' + num(t, s.dec) : '—';
        const f = isNum(d) && isNum(t) && t > 0 ? Math.max(0, Math.min(100, 100 * d / t)) : 0;
        const bar = $('.bar > i', tr); bar.style.width = f.toFixed(0) + '%'; bar.style.setProperty('--c', f >= 100 ? 'var(--running)' : `var(${s.color})`);
        bar.style.background = f >= 100 ? css('--running') : css(s.color);
      }
      const h = $('.h', tr); clear(h);
      if (health) h.append(chip(health, health === 'ok' ? 'chip-ok' : 'chip-bad'));
    });
  }

  /* ------------------------------------------------------------------ runs and logs */
  const avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
  function labChip(l) { const c = el('span', 'lab-chip' + (l ? ' l-' + l : ''), l || 'rules'); return c; }

  function renderRuns() {
    const runs = S.runs, tb = clear($('#runTable tbody'));
    $('#runCount').textContent = runs.length ? runs.length + ' shown' : '';
    if (!runs.length) { const tr = el('tr'), td = el('td', 'empty', 'No completed runs yet.'); td.colSpan = 6; tr.append(td); tb.append(tr); }
    runs.forEach(r => {
      const tr = el('tr');
      const c = el('td'); c.append(labChip(r.label));
      tr.append(el('td', 'num l', clock(r.start_ts)), c, el('td', 'num', isNum(r.trigger_to_start_s) ? num(r.trigger_to_start_s, 0) + ' s' : '—'),
        el('td', 'num', mmss(r.duration_s)), el('td', 'num', isNum(r.run_on_s) ? mmss(r.run_on_s) : '—'), el('td', null, END_TEXT[r.end_reason] || r.end_reason || '—'));
      tb.append(tr);
    });
    const host = clear($('#runSummary'));
    const card = (k, v) => { const col = el('div', 'col-6 col-md-3'), m = el('div', 'metric-card'); m.append(el('div', 'k', k), el('div', 'v', v)); col.append(m); host.append(col); };
    card('Runs', String(runs.length));
    card('Avg trigger → fan', runs.length ? num(avg(runs.map(r => r.trigger_to_start_s).filter(isNum)), 1) + ' s' : '—');
    card('Avg fan time', runs.length ? mmss(avg(runs.map(r => r.duration_s).filter(isNum))) : '—');
    card('Avg run-on', runs.some(r => isNum(r.run_on_s)) ? mmss(avg(runs.map(r => r.run_on_s).filter(isNum))) : '—');
  }

  function renderEvents() {
    renderEventList($('#liveLog'), S.events, 12);
    const f = $('#evFilter').value;
    renderEventList($('#fullLog'), f ? S.events.filter(e => e.level === f) : S.events, 200);
    const rec = S.events.filter(e => e.kind === 'collect');
    renderEventList($('#collectLog'), rec, 6);
  }

  const loadEvents = () => api('/api/events?limit=200').then(j => { S.events = j; renderEvents(); }).catch(() => {});
  const loadRuns   = () => api('/api/runs?limit=100').then(j => { S.runs = j; renderRuns(); }).catch(() => {});

  /* ------------------------------------------------------------------ ML report */
  const toObj = x => (x && typeof x === 'object' && !Array.isArray(x)) ? x : null;
  const scalar = v => v == null || ['string', 'number', 'boolean'].includes(typeof v);
  const pretty = v => typeof v === 'number' ? (Number.isInteger(v) ? String(v) : (+v).toFixed(4).replace(/0+$/, '').replace(/\.$/, '')) : v == null ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v);
  const isRate = (k, v) => typeof v === 'number' && v >= 0 && v <= 1 && /acc|f1|prec|recall|auc|score|rate|kappa/i.test(k);

  function section(title, icon) {
    const col = el('div', 'col-12'), p = el('div', 'panel'), t = el('h2', 'panel-title');
    t.append(el('span', null, title)); if (icon) t.firstChild.prepend(el('i', 'bi ' + icon));
    p.append(t); col.append(p); return { col, body: p };
  }
  function kvList(obj) {
    const dl = el('dl', 'kv');
    Object.entries(obj).forEach(([k, v]) => { dl.append(el('dt', null, k.replace(/_/g, ' ')), el('dd', null, pretty(v))); });
    return dl;
  }
  function tableOf(rows, cols) {
    const wrap = el('div', 'table-scroll'), t = el('table', 'table'), thead = el('thead'), tr = el('tr');
    tr.append(el('th', null, ''));
    cols.forEach(c => tr.append(el('th', 'num', c.replace(/_/g, ' '))));
    thead.append(tr); t.append(thead);
    const tb = el('tbody');
    Object.entries(rows).forEach(([name, o]) => {
      const r = el('tr'); r.append(el('td', null, name));
      cols.forEach(c => r.append(el('td', 'num', isRate(c, o[c]) ? pct(o[c], 1) : pretty(o[c]))));
      tb.append(r);
    });
    t.append(tb); wrap.append(t); return wrap;
  }

  function renderMetrics(m) {
    const { col, body } = section('Metrics', 'bi-speedometer2');
    if (m.error) { body.append(el('div', 'empty', 'metrics.json could not be read.')); return col; }
    const scalars = Object.entries(m).filter(([, v]) => scalar(v));
    if (scalars.length) {
      const g = el('div', 'row g-2 mb-3');
      scalars.forEach(([k, v]) => {
        const c = el('div', 'col-6 col-md-3 col-xl-2'), card = el('div', 'metric-card');
        card.append(el('div', 'k', k.replace(/_/g, ' ')), el('div', 'v', isRate(k, v) ? pct(v, 1) : pretty(v))); c.append(card); g.append(c);
      });
      body.append(g);
    }
    const objs = Object.entries(m).filter(([, v]) => toObj(v));
    const flat = objs.filter(([, v]) => Object.values(v).every(scalar));                       // e.g. per-class rows
    const nested = objs.filter(([, v]) => Object.keys(v).length && Object.values(v).every(x => toObj(x) && Object.values(x).every(scalar)));
    const other = objs.filter(x => !flat.includes(x) && !nested.includes(x));
    const colsOf = rows => [...new Set(Object.values(rows).flatMap(o => Object.keys(o)))];
    if (flat.length) { const rows = Object.fromEntries(flat); body.append(tableOf(rows, colsOf(rows))); }
    nested.forEach(([k, v]) => { body.append(el('h3', 'panel-title mt-3', k.replace(/_/g, ' '))); body.append(tableOf(v, colsOf(v))); });
    other.forEach(([k, v]) => { body.append(el('h3', 'panel-title mt-3', k.replace(/_/g, ' '))); body.append(kvList(v)); });
    return col;
  }

  // accepts [[..]], {labels,matrix}, {classes,confusion_matrix} or {true:{pred:n}}
  function normCM(x) {
    let labels = null, m = null;
    if (Array.isArray(x)) m = x;
    else if (toObj(x)) {
      labels = x.labels || x.classes || x.names || x.class_names || null;
      m = x.matrix || x.cm || x.data || x.values || x.confusion_matrix || null;
      if (!m) {
        const ks = Object.keys(x);
        if (ks.length && ks.every(k => toObj(x[k]))) {
          const cols = [...new Set(ks.flatMap(k => Object.keys(x[k])))];
          labels = labels || ks; m = ks.map(k => cols.map(c => x[k][c] || 0));
        }
      }
    }
    if (!Array.isArray(m) || !m.length || !m.every(r => Array.isArray(r))) return null;
    if (!labels || labels.length !== m.length) labels = m.length === LABELS.length ? LABELS : m.map((_, i) => String(i));
    return { labels: labels.map(l => (typeof l === 'number' && LABELS[l]) ? LABELS[l] : String(l)), m };
  }

  function renderConfusion(raw) {
    const { col, body } = section('Confusion matrix', 'bi-grid-3x3');
    const cm = normCM(raw);
    if (!cm) { body.append(el('div', 'empty', 'confusion_matrix.json has a layout this page does not recognise.')); return col; }
    body.append(el('div', 'cm-axis', 'predicted →'));
    const wrap = el('div', 'table-scroll'), t = el('table', 'cm'), head = el('tr');
    head.append(el('th'));
    cm.labels.forEach(l => head.append(el('th', null, l)));
    t.append(head);
    const good = css('--idle'), bad = css('--fault');
    cm.m.forEach((row, i) => {
      const tot = row.reduce((a, b) => a + (+b || 0), 0), tr = el('tr');
      tr.append(el('th', 'rowh', cm.labels[i]));
      row.forEach((v, j) => {
        const rate = tot ? (+v || 0) / tot : 0, td = el('td');
        td.style.background = hexA(i === j ? good : bad, v ? 0.12 + 0.68 * rate : 0.04);
        td.style.color = v ? css('--text') : css('--muted');
        td.append(el('b', null, pretty(v)), el('small', null, tot ? pct(rate, 0) : ''));
        td.title = `true ${cm.labels[i]} → predicted ${cm.labels[j]}: ${v}`;
        tr.append(td);
      });
      t.append(tr);
    });
    wrap.append(t); body.append(wrap);
    body.append(el('div', 'cm-axis', '↑ true label (rows) · green = correct, red = confused · % of the true class'));
    return col;
  }

  function renderImportance(raw) {
    const { col, body } = section('Feature importance', 'bi-sort-down');
    let items = [];
    if (Array.isArray(raw)) items = raw.map(o => Array.isArray(o) ? o : [o.feature || o.name, o.importance ?? o.value]);
    else if (toObj(raw)) {
      const names = raw.features || raw.names, vals = raw.importances || raw.importance || raw.values;
      if (Array.isArray(names) && Array.isArray(vals)) items = names.map((n, i) => [n, vals[i]]);
      else items = Object.entries(raw);
    }
    items = items.filter(([, v]) => isNum(v)).map(([n, v]) => [String(n), +v]).sort((a, b) => b[1] - a[1]).slice(0, 20);
    if (!items.length) { body.append(el('div', 'empty', 'No feature importances in this report.')); return col; }
    const mx = items[0][1] || 1;
    items.forEach(([n, v]) => {
      const r = el('div', 'fi-row'), tr = el('div', 'track'), i = el('i');
      i.style.width = (100 * v / mx).toFixed(1) + '%'; tr.append(i);
      r.append(el('span', 'nm', n), tr, el('span', 'pc', v.toFixed(3))); r.firstChild.title = n; body.append(r);
    });
    return col;
  }

  function renderKV(title, icon, obj) {
    const { col, body } = section(title, icon);
    if (toObj(obj) && !obj.error) body.append(kvList(obj)); else body.append(el('div', 'empty', 'Not available.'));
    return col;
  }

  function loadReport() {
    const body = $('#mlBody');
    api('/api/ml/report').then(rep => {
      clear(body);
      if (!rep.available) {
        const p = el('div', 'panel');
        p.append(el('h2', 'panel-title', 'No report yet'),
          el('p', 'muted mb-0', 'The training script writes metrics.json, confusion_matrix.json, feature_importance.json, dataset_summary.json and model_info.json into ml/reports/latest/. Once they exist, they show up here. Nothing on this page is made up.'));
        body.append(p); $('#mlSub').textContent = ''; return;
      }
      const info = toObj(rep.model_info);
      $('#mlSub').textContent = info && (info.name || info.model || info.type) ? String(info.name || info.model || info.type) : '';
      const row = el('div', 'row g-3');
      if (rep.metrics) row.append(renderMetrics(rep.metrics));
      if (rep.confusion_matrix) { const c = renderConfusion(rep.confusion_matrix); c.className = 'col-12 col-xl-7'; row.append(c); }
      if (rep.feature_importance) { const c = renderImportance(rep.feature_importance); c.className = 'col-12 col-xl-5'; row.append(c); }
      if (rep.dataset_summary) { const c = renderKV('Dataset', 'bi-database', rep.dataset_summary); c.className = 'col-12 col-lg-6'; row.append(c); }
      if (rep.model_info) { const c = renderKV('Model', 'bi-box', rep.model_info); c.className = 'col-12 col-lg-6'; row.append(c); }
      body.append(row);
    }).catch(() => { clear(body).append(el('div', 'panel empty', 'Could not reach the server.')); });
  }

  /* ------------------------------------------------------------------ collect */
  function renderCollect() {
    const r = S.row, c = r && r.collecting, active = !!(c && c.active);
    $('#btnStart').disabled = active || !r;
    $('#btnStop').disabled = !active;
    $('#recBanner').classList.toggle('active', active);
    if (active) {
      if (S.collectFile !== c.file) { S.collectFile = c.file; S.collectStartTs = r.ts; }
      $('#recTitle').textContent = 'Recording ' + c.label;
      $('#recSub').textContent = `${c.file} · ${mmss(r.ts - S.collectStartTs)}`;
    } else { S.collectFile = null; $('#recTitle').textContent = 'Not recording'; $('#recSub').textContent = ''; }
  }

  function collectCall(path, body) {
    const msg = $('#collectMsg'); msg.style.color = '';
    return api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
      .then(() => { msg.textContent = 'Sent. The control process picks it up within a second.'; msg.style.color = css('--idle'); setTimeout(loadEvents, 1800); })
      .catch(e => { msg.textContent = 'Failed: ' + e.message; msg.style.color = css('--fault'); });
  }

  /* ------------------------------------------------------------------ system */
  const loadHealth = () => api('/api/health').then(j => { S.health = j; renderSystem(); }).catch(() => { S.health = null; renderSystem(); });
  const loadConfig = () => api('/api/config').then(j => { S.cfg = j; renderSystem(); renderTimers(); renderStats(); }).catch(() => {});

  function kv(host, pairs) {
    clear(host);
    pairs.forEach(([k, v, plain]) => { host.append(el('dt', null, k)); const dd = el('dd', plain ? 'plain' : null, v == null ? '—' : String(v)); host.append(dd); });
  }
  function renderSystem() {
    const h = S.health, c = S.cfg;
    const skew = h && isNum(h.server_time) ? (Date.now() / 1000 - h.server_time) : null;
    kv($('#sysBackend'), [
      ['Source', h ? h.source : '—'], ['Mode', h ? h.mode || '—' : '—'], ['Schema version', h ? h.schema : '—'],
      ['Last row age', h && isNum(h.last_row_age_s) ? h.last_row_age_s + ' s' : '—'],
      ['Data is stale', h ? (h.stale ? 'yes' : 'no') : '—'],
      ['Rows / events / runs in DB', h && h.db ? `${h.db.rows} / ${h.db.events} / ${h.db.runs}` : '—'],
    ]);
    kv($('#sysLink'), [
      ['Live link (WebSocket)', S.wsOk ? 'connected' : 'not connected', true],
      ['Last message', S.lastMsgAt ? Math.round((Date.now() - S.lastMsgAt) / 1000) + ' s ago' : '—'],
      ['Reconnects', S.reconnects],
      ['Served from', location.host],
      ['Clock difference vs server', skew == null ? '—' : (skew >= 0 ? '+' : '−') + Math.abs(skew).toFixed(1) + ' s'],
    ]);
    const T = c && c.timers;
    kv($('#sysTimers'), T ? Object.entries(T).map(([k, v]) => [k.replace(/_/g, ' '), v]).concat([['Timer set', c.demo_timers ? 'demo (shortened)' : 'real', true]]) : [['—', '—']]);
    kv($('#sysThr'), c ? Object.entries(c.thresholds || {}).map(([k, v]) => ['rise ' + k, v]).concat([['clear ratio', c.clear_ratio], ['tick (s)', c.tick_s]]) : [['—', '—']]);
  }

  /* ------------------------------------------------------------------ link + websocket */
  function renderLink() {
    const link = $('#link'), age = (Date.now() - S.lastMsgAt) / 1000;
    let cls = 'wait', text = 'connecting';
    if (S.wsOk && age < 4) { cls = 'ok'; text = 'connected'; }
    else if (S.wsOk) { cls = 'bad'; text = 'no data ' + Math.round(age) + ' s'; }
    else if (S.lastMsgAt && age < 5) { cls = 'wait'; text = 'polling'; }
    else if (S.lastMsgAt) { cls = 'bad'; text = 'offline'; }
    link.className = 'link ' + cls; $('#linkText').textContent = text;
  }

  let ws = null, retry = 0;
  function connect() {
    try { ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/live'); } catch (e) { return scheduleRetry(); }
    ws.onopen = () => { S.wsOk = true; retry = 0; renderLink(); loadConfig(); };
    ws.onmessage = e => { try { handle(JSON.parse(e.data)); } catch (_) { /* ignore a bad frame */ } };
    ws.onclose = () => { if (S.wsOk) S.reconnects++; S.wsOk = false; renderLink(); scheduleRetry(); };
    ws.onerror = () => { try { ws.close(); } catch (_) { } };
  }
  function scheduleRetry() { retry = Math.min(retry + 1, 5); setTimeout(connect, 600 * retry); }

  // if the WebSocket is blocked (some proxies), fall back to polling /api/live
  setInterval(() => { if (!S.wsOk) api('/api/live').then(handle).catch(renderLink); }, 2000);
  setInterval(() => { renderLink(); if ($('#pane-system').classList.contains('active')) renderSystem(); }, 1000);
  setInterval(loadEvents, 3000);
  setInterval(() => { if ($('#pane-runs').classList.contains('active')) loadRuns(); }, 5000);
  setInterval(() => { if ($('#pane-system').classList.contains('active')) loadHealth(); }, 5000);

  /* ------------------------------------------------------------------ start */
  function init() {
    buildStatCards(); buildChartCards(); buildSensorTable(); buildLabelPick();
    buildProbBars($('#miniProbs'), true); buildProbBars($('#clfProbs'), false);
    buildCharts();

    $('#windowSeg').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      S.windowS = +b.dataset.w; $$('#windowSeg button').forEach(x => x.classList.toggle('active', x === b)); renderCharts();
    });
    $('#evFilter').addEventListener('change', renderEvents);
    $('#mlReload').addEventListener('click', loadReport);
    $('#btnStart').addEventListener('click', () => {
      const label = ($('input[name=collectLabel]:checked') || {}).value;
      collectCall('/api/collect/start', { label, note: $('#collectNote').value.trim() });
    });
    $('#btnStop').addEventListener('click', () => collectCall('/api/collect/stop'));

    showTab(location.hash.slice(1));
    // config first (thresholds for the charts), then history, then live
    Promise.allSettled([loadConfig(), api('/api/history?minutes=10').then(rows => { rows.forEach(pushRow); renderCharts(); }), loadEvents()])
      .finally(connect);
  }
  init();
})();
