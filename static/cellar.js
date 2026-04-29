// ── State ─────────────────────────────────────────────────────────────────────
let wines = [], filtered = [], current = null;
let memory = [];
let socket = null;
let audioContext = null;
let audioChunks = [], isPlaying = false;

// Bench state
let benchWines    = [];          // ordered list of wines on the bench
let introduced    = new Set();   // wine IDs the barkeep has already pitched
let wineMemory    = {};          // per-wine conversation history: { [id]: [{role,content}] }

// In-flight request tracking (captures state at send time, not at callback time)
let _streamingReply   = '';      // accumulates WS text_chunk content for reliable memory save
let _pendingWineId    = null;    // id of the wine that issued the current in-flight request (null = global)
let _pendingRecIntent = false;   // whether the in-flight request was a recommendation ask

// VAD state
let vadStream = null, vadAnalyser = null, vadProcessor = null, vadSource = null, vadContext = null;
let vadActive = false;          // mic open and monitoring
let speechDetected = false;     // currently above threshold
let silenceTimer = null;
let pendingTranscript = '';

// Recognition
let recog = null;
let recognizing = false;

// Orb state: idle | listening | thinking | speaking
let orbState = 'idle';

// VAD onset timer (module-scoped so stopVAD can cancel it)
let speechOnsetTimer = null;

const API     = '/api';
const API_WS  = `ws://${window.location.host}/api/ws/chat`;

// VAD config
const VAD_THRESHOLD   = 0.025;   // RMS threshold to detect voice (higher = less sensitive)
const SILENCE_MS      = 900;     // ms of silence before we consider utterance done
const SPEECH_ONSET_MS = 150;     // ms signal must stay above threshold before triggering

// ── WebSocket ─────────────────────────────────────────────────────────────────
function initSocket() {
  socket = new WebSocket(API_WS);
  socket.binaryType = 'arraybuffer';
  socket.onopen  = () => console.log('[WS] connected');
  socket.onmessage = async (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      if (ev.data.byteLength > 0) enqueueAudio(ev.data);
    } else {
      const d = JSON.parse(ev.data);
      if (d.type === 'text_chunk') {
        console.log(`[WS] text_chunk: "${d.content}"`);
        _streamingReply += d.content;
        appendBotText(d.content);
      } else if (d.type === 'rec_wines') {
        // Tool-driven recommendation: the LLM called recommend_similar_wines_tool
        // and the router forwarded the results here. source_wine_id tells us which
        // wine was used as the seed — could differ from current (bench scenario).
        const sourceWine = wines.find(w => w.id === d.source_wine_id) || current;
        addSimilarToRecPanel(sourceWine, d.wines);
        console.log('[REC] tool-driven recs for', sourceWine?.name, ':', (d.wines||[]).map(w => w.name));
      } else if (d.type === 'audio_done') {
        // Drain captured state before anything async can change it
        const reply     = _streamingReply.trim();
        const pendingId = _pendingWineId;
        const recIntent = _pendingRecIntent;
        _streamingReply   = '';
        _pendingWineId    = null;
        _pendingRecIntent = false;

        if (reply && pendingId != null) {
          if (!wineMemory[pendingId]) wineMemory[pendingId] = [];
          wineMemory[pendingId].push({ role: 'assistant', content: reply });
          if (current?.id === pendingId) {
            memory.push({ role: 'assistant', content: reply });
            updateMemUI();
          }
        }
        // Fallback: if rec was intended but LLM didn't call the tool, fetch manually
        if (recIntent && pendingId != null) {
          const alreadyHasGroup = recGroups.some(g => g.sourceWine.id === pendingId);
          if (!alreadyHasGroup) {
            const srcWine = wines.find(w => w.id === pendingId);
            if (srcWine) fetchSimilarWines(srcWine);
          }
        }
        const chat = document.getElementById('modalChat');
        if (chat) chat.querySelectorAll('.modal-chat-bot.streaming').forEach(el => el.classList.remove('streaming'));
        await playAccumulatedAudio();
      }
    }
  };
  socket.onerror = e => console.error('[WS] error', e);
  socket.onclose = () => setTimeout(initSocket, 1000);
}

// ── Audio — accumulate all chunks then decode+play the complete buffer ────────
function enqueueAudio(buffer) {
  // Just collect — don't try to decode partial MP3 frames mid-stream
  audioChunks.push(new Uint8Array(buffer));
  console.log(`[AUDIO] Chunk received: ${buffer.byteLength} bytes, total chunks=${audioChunks.length}`);
}

async function playAccumulatedAudio() {
  if (!audioChunks.length) return;

  // Concatenate all chunks into one contiguous buffer
  const totalLength = audioChunks.reduce((sum, c) => sum + c.byteLength, 0);
  const merged = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of audioChunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  audioChunks = [];   // clear for next response

  setOrb('speaking');
  isPlaying = true;
  try {
    if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === 'suspended') await audioContext.resume();
    const decoded = await audioContext.decodeAudioData(merged.buffer);
    await new Promise(res => {
      const src = audioContext.createBufferSource();
      src.buffer = decoded;
      src.connect(audioContext.destination);
      src.onended = res;
      src.start();
    });
  } catch(e) { console.error('[AUDIO] decode/play error:', e); }

  isPlaying = false;
  speechDetected = false;
  if (vadActive) {
    setOrb('listening');
    rearmRecognition();
  } else setOrb('idle');
}

function stopAudio() {
  audioChunks = [];
  isPlaying  = false;
  if (audioContext) { audioContext.close().then(() => audioContext = null); }
}

// ── Orb state ─────────────────────────────────────────────────────────────────
const ORB_LABELS = {
  idle:      'tap to start',
  listening: 'listening…',
  thinking:  'thinking…',
  speaking:  'speaking…',
};

function setOrb(state) {
  orbState = state;
  const wrap  = document.getElementById('orbWrap');
  const label = document.getElementById('orbLabel');
  if (wrap)  wrap.className  = `orb-wrap ${state}`;
  if (label) label.textContent = ORB_LABELS[state] || '';
  updateBkStatus(state);
}

function updateBkStatus(state) {
  const map = { idle:'ready when you are', listening:'listening…', thinking:'thinking…', speaking:'speaking…' };
  document.getElementById('bkStatus').textContent = map[state] || '';
}

// ── Transcript UI (bench is primary; modal-chat mirrors it) ───────────────────
let _lastYou = '';   // track so we can add a "you" bubble to modal-chat

function setYouText(text) {
  _lastYou = text || '';
  document.getElementById('benchYou').textContent = text ? `"${text}"` : '';
  // Add a bubble to modal-chat if modal is open
  if (document.getElementById('modalBg').classList.contains('open') && text) {
    const chat = document.getElementById('modalChat');
    document.getElementById('modalChatEmpty').style.display = 'none';
    const el = document.createElement('div');
    el.className = 'modal-chat-you';
    el.textContent = `"${text}"`;
    el.dataset.role = 'you-pending'; // will be matched by appendBotText
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
  }
}

function appendBotText(chunk) {
  // Bench mini-transcript
  document.getElementById('benchBot').textContent += chunk;
  // Modal-chat: stream into the current bot bubble (create if needed)
  const chat = document.getElementById('modalChat');
  if (!chat) return;
  document.getElementById('modalChatEmpty').style.display = 'none';
  let botEl = chat.querySelector('.modal-chat-bot.streaming');
  if (!botEl) {
    botEl = document.createElement('div');
    botEl.className = 'modal-chat-bot streaming';
    chat.appendChild(botEl);
  }
  botEl.textContent += chunk;
  chat.scrollTop = chat.scrollHeight;
}

function clearTranscript() {
  document.getElementById('benchYou').textContent = '';
  document.getElementById('benchBot').textContent  = '';
  // Modal-chat: remove the streaming class so next reply starts a fresh bubble
  const chat = document.getElementById('modalChat');
  if (chat) {
    chat.querySelectorAll('.modal-chat-bot.streaming').forEach(el => el.classList.remove('streaming'));
  }
}

// ── Send question ─────────────────────────────────────────────────────────────
function benchContext() {
  if (!benchWines.length) return '';
  const names = benchWines.map(w => w.name).join(', ');
  return `Wines currently on the tasting bench: ${names}. The active/selected wine is: ${current?.name || 'none'}.`;
}

function activeFiltersList() {
  const list = [];
  Object.entries(activeFilters).forEach(([col, vals]) => {
    if (vals.size) list.push(`${colLabel(col)}: ${[...vals].join(', ')}`);
  });
  return list;
}

// Keywords that signal the user wants recommendations
const REC_INTENT_KEYWORDS = [
  'recommend', 'similar', 'suggest', 'what else', 'other wine', 'alternative',
  'like this', 'pair', 'pairing', 'what should', 'what would you', 'something like'
];

function isRecIntent(q) {
  const lower = q.toLowerCase();
  return REC_INTENT_KEYWORDS.some(kw => lower.includes(kw));
}

function sendQuestion(q) {
  if (!q) return;
  const isGlobal = !current;   // no wine in focus → route to catalog-search agent
  console.log(`[SEND] "${q}" wine=${current?.name ?? '(global)'}`);

  // Capture intent at send time — audio_done uses these, not live state
  _pendingWineId    = current?.id ?? null;
  _pendingRecIntent = !isGlobal && isRecIntent(q);
  _streamingReply   = '';

  stopAudio();
  clearTranscript();   // clears benchYou, benchBot, streaming classes — no second clear needed
  setYouText(q);
  setOrb('thinking');

  let payload;
  if (isGlobal) {
    // Global browsing mode: backend uses catalog-search agent with tool-use
    payload = { question: q, wine: null, history: [] };
  } else {
    const history = (wineMemory[current.id] || []).slice(-10);
    const filters = activeFiltersList();
    const contextParts = [];
    if (benchWines.length > 1) contextParts.push(benchContext());
    contextParts.push(`Active filters: ${filters.length ? filters.join('; ') : 'none'}.`);
    if (_pendingRecIntent) contextParts.push('The user is asking for wine recommendations. Suggest wines that would complement or contrast well with the current wine based on what you know.');
    const qWithContext = `[Context: ${contextParts.join(' ')}]\n${q}`;
    payload = { question: qWithContext, wine: current, history };

    // Push user turn to memory immediately (before response arrives)
    if (!wineMemory[current.id]) wineMemory[current.id] = [];
    wineMemory[current.id].push({ role: 'user', content: q });
    memory.push({ role: 'user', content: q });
    updateMemUI();
  }

  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  } else {
    console.warn('[WS] Socket not open, reconnecting…');
    initSocket();
    setTimeout(() => socket.send(JSON.stringify(payload)), 600);
  }
}

// ── Stop barkeep + always-on VAD ─────────────────────────────────────────────

function stopVAD() {
  vadActive = false;
  speechDetected = false;
  clearTimeout(silenceTimer);
  clearTimeout(speechOnsetTimer); // cancel any pending onset debounce
  if (vadSource)  { try { vadSource.disconnect(); } catch(_) {} vadSource = null; }
  if (vadContext) { try { vadContext.close(); } catch(_) {} vadContext = null; }
  if (vadStream)  { vadStream.getTracks().forEach(t => t.stop()); vadStream = null; }
  console.log('[VAD] Stopped and cleaned up');
}

function onSilence() {
  console.log('[VAD] 🔇 Silence — waiting for recognition result');
  speechDetected = false;
}

// ── Speech recognition ────────────────────────────────────────────────────────
function setupRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;
  const r = new SR();
  r.lang = 'en-US';
  r.interimResults = true;
  r.continuous = false;
  r.onresult = e => {
    let interim = '', final = '';
    for (const res of e.results) {
      if (res.isFinal) final += res[0].transcript;
      else interim += res[0].transcript;
    }
    const text = (final || interim).trim();
    console.log(`[RECOG] result — interim="${interim.trim()}" final="${final.trim()}"`);
    setYouText(text);
    if (final && final.trim()) {
      console.log(`[RECOG] ✅ Final transcript: "${final.trim()}" — sending to barkeep`);
      recognizing = false;
      sendQuestion(final.trim());
    }
  };
  r.onerror = (e) => {
    console.error('[RECOG] ❌ Error:', e.error, e.message);
    recognizing = false;
    if (vadActive) setOrb('listening');
  };
  r.onend = () => {
    console.log(`[RECOG] ended — recognizing=${recognizing} orbState=${orbState} vadActive=${vadActive}`);
    recognizing = false;
    rearmRecognition();   // single canonical rearm — deduplicates with playAccumulatedAudio's rearm
  };
  r.onstart = () => console.log('[RECOG] 🎙️ Recognition started');
  return r;
}

// Single canonical place to restart recognition after audio ends or recognition ends.
// Guards against double-start (playAccumulatedAudio and r.onend both call this).
function rearmRecognition() {
  if (!vadActive) return;
  setTimeout(() => {
    if (vadActive && !recognizing && orbState !== 'thinking' && orbState !== 'speaking') {
      startRecognition();
    }
  }, 250);
}

function startRecognition() {
  if (!recog) { console.warn('[RECOG] No recognizer available'); return; }
  if (recognizing) { console.log('[RECOG] Already recognizing, skipping'); return; }
  recognizing = true;
  setOrb('listening');
  console.log('[RECOG] Starting recognition…');
  try { recog.start(); } catch(e) { console.error('[RECOG] start() threw:', e); recognizing = false; }
}

// ── Stop barkeep (the only manual control now) ────────────────────────────────
function stopBarkeep() {
  stopAudio();
  audioChunks = [];
  // Stay in listening state — mic keeps running
  if (vadActive) setOrb('listening');
  else setOrb('idle');
}

// ── Start always-on VAD (called once when first bottle hits bench) ────────────
async function ensureVADRunning() {
  if (vadActive) return;
  try {
    vadStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch(e) {
    console.warn('[VAD] Mic denied:', e);
    return;
  }
  if (vadContext) { try { vadContext.close(); } catch(_) {} }
  vadContext = new (window.AudioContext || window.webkitAudioContext)();
  if (vadContext.state === 'suspended') await vadContext.resume();

  vadSource   = vadContext.createMediaStreamSource(vadStream);
  vadAnalyser = vadContext.createAnalyser();
  vadAnalyser.fftSize = 512;
  vadSource.connect(vadAnalyser);

  const data = new Float32Array(vadAnalyser.fftSize);
  let frameCount = 0;
  speechOnsetTimer = null;
  vadActive = true;
  setOrb('listening');
  console.log('[VAD] ✅ Always-on VAD started');

  function tick() {
    if (!vadActive) return;
    vadAnalyser.getFloatTimeDomainData(data);
    let rms = 0;
    for (let i = 0; i < data.length; i++) rms += data[i] * data[i];
    rms = Math.sqrt(rms / data.length);

    frameCount++;
    if (frameCount % 60 === 0) console.log(`[VAD] rms=${rms.toFixed(4)} speech=${speechDetected} playing=${isPlaying}`);

    if (rms > VAD_THRESHOLD) {
      if (!speechDetected && !speechOnsetTimer) {
        speechOnsetTimer = setTimeout(() => {
          speechDetected = true;
          speechOnsetTimer = null;
          console.log(`[VAD] 🎤 Speech confirmed rms=${rms.toFixed(4)}`);
          // Barge-in: stop audio if playing, then listen
          if (isPlaying) { stopBarkeep(); }
          startRecognition();
        }, SPEECH_ONSET_MS);
      }
      clearTimeout(silenceTimer);
      silenceTimer = setTimeout(onSilence, SILENCE_MS);
    } else {
      if (speechOnsetTimer) { clearTimeout(speechOnsetTimer); speechOnsetTimer = null; }
    }
    requestAnimationFrame(tick);
  }
  tick();
}

// ── Memory UI ─────────────────────────────────────────────────────────────────
function updateMemUI() {
  // memory still tracks current wine's convo — update bench bottle badge if needed
  const badge = document.getElementById('memBadge');
  if (!badge) return;
  const ex = Math.floor(memory.length / 2);
  badge.textContent = ex === 0 ? 'No context yet' : `${ex} turn${ex !== 1 ? 's' : ''} of context`;
  badge.className   = ex > 0 ? 'mem-badge active' : 'mem-badge';
}

function clearMem() {
  if (!memory.length) return;
  if (!confirm("Clear the barkeep's memory for this wine?")) return;
  memory = [];
  if (current) wineMemory[current.id] = [];
  updateMemUI();
  clearTranscript();
}

// ── Bench ─────────────────────────────────────────────────────────────────────
function renderBench() {
  const row   = document.getElementById('benchBottles');
  const empty = document.getElementById('benchEmpty');
  // remove all existing bottle els
  row.querySelectorAll('.bench-bottle').forEach(el => el.remove());
  if (!benchWines.length) { empty.style.display = ''; return; }
  empty.style.display = 'none';
  benchWines.forEach(w => {
    const div = document.createElement('div');
    div.className = 'bench-bottle' + (current?.id === w.id ? ' active' : '');
    div.dataset.id = w.id;
    const col = (w.color||'').toLowerCase();
    div.innerHTML = `
      <span class="bench-bottle-x" onclick="removeFromBench(event,${w.id})">✕</span>
      ${w.image_url
        ? `<img src="${escAttr(w.image_url)}" alt="" onerror="this.style.display='none'">`
        : `<div style="width:28px;height:44px;display:flex;align-items:center;justify-content:center;font-size:18px">🍷</div>`}
      <div class="bench-bottle-name">${esc(w.name||'')}</div>`;
    div.addEventListener('click', () => openModal(w.id));
    row.appendChild(div);
  });
}

function addToBench(w) {
  if (benchWines.find(b => b.id === w.id)) return;
  benchWines.push(w);
  if (!wineMemory[w.id]) wineMemory[w.id] = [];
  renderBench();
  // Start always-on mic when first bottle arrives
  ensureVADRunning();
  setTimeout(() => {
    const el = document.querySelector(`.bench-bottle[data-id="${w.id}"]`);
    if (el) el.scrollIntoView({ behavior: 'smooth', inline: 'nearest', block: 'nearest' });
  }, 50);
}

function removeFromBench(e, id) {
  e.stopPropagation();
  benchWines = benchWines.filter(w => w.id !== id);
  delete wineMemory[id];
  introduced.delete(id);
  if (current?.id === id) {
    stopAudio();
    // Only kill VAD if nothing left on bench
    if (benchWines.length === 0) {
      stopVAD();
      setOrb('idle');
    }
    try { recog && recog.abort(); } catch(_) {}
    recognizing = false;
    current = null;
    clearTranscript();
    document.getElementById('modalBg').classList.remove('open');
    if (benchWines.length) openModal(benchWines[benchWines.length - 1].id, true);
  }
  renderBench();
}

function autoIntroduce(w) {
  stopAudio();
  audioChunks = [];
  _streamingReply   = '';
  _pendingWineId    = w.id;   // capture now — audio_done saves to this wine's memory
  _pendingRecIntent = false;

  const ctx = benchWines.length > 1
    ? ` (Other wines on the tasting bench: ${benchWines.filter(b => b.id !== w.id).map(b => b.name).join(', ')})`
    : '';
  const pitch = `In 2-3 warm sentences, introduce this wine and make me want to try it. Don't ask questions.${ctx}`;
  clearTranscript();
  setOrb('thinking');
  const payload = { question: pitch, wine: w, history: [] };
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  } else {
    initSocket();
    setTimeout(() => socket.send(JSON.stringify(payload)), 600);
  }
  // audio_done handler saves the real intro text to wineMemory[_pendingWineId]
}

// ── Wine grid + modal ─────────────────────────────────────────────────────────
function esc(s)     { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function escAttr(s) { return String(s).replace(/"/g,'&quot;'); }

async function loadWines() {
  try {
    const r = await fetch(`${API}/wines`);
    wines    = await r.json();
    filtered = [...wines];
    document.getElementById('countLabel').textContent = `${wines.length} bottles`;
    renderGrid();
    buildFilterUI();
  } catch(e) {
    document.getElementById('grid').innerHTML =
      `<div style="grid-column:1/-1;padding:40px;text-align:center;color:var(--text3)">
        Could not reach server.<br><code style="font-size:11px;margin-top:6px;display:block">${e.message}</code>
      </div>`;
  }
}

function parseRatings(raw) {
  if (!raw) return [];
  try { const a = typeof raw === 'string' ? JSON.parse(raw) : raw; return Array.isArray(a) ? a : []; }
  catch { return []; }
}

function renderGrid() {
  const g = document.getElementById('grid');
  if (!filtered.length) { g.innerHTML = '<div style="grid-column:1/-1;padding:40px;text-align:center;color:var(--text3)">No wines match</div>'; return; }
  g.innerHTML = filtered.map(w => {
    const col    = (w.color||'').toLowerCase();
    const dotCls = col==='red'?'d-red':col==='white'?'d-white':col==='sparkling'?'d-sparkling':col==='rose'||col==='rosé'?'d-rose':'d-other';
    const ratings = parseRatings(w.professional_ratings);
    const top     = ratings[0];
    const price   = w.retail ? `$${parseFloat(w.retail).toFixed(0)}` : '—';
    const imgHtml = w.image_url
      ? `<img src="${escAttr(w.image_url)}" alt="" loading="lazy" onerror="this.parentElement.innerHTML='<div style=font-size:11px;color:var(--text3)>No image</div>'">`
      : `<div style="font-size:11px;color:var(--text3)">${col||'wine'}</div>`;
    return `<div class="wine-card" onclick="openModal(${w.id})">
      <div class="card-img">
        ${imgHtml}
        <div class="color-dot ${dotCls}"></div>
        <div class="card-hover">
          <div class="h-region">${esc(w.region||'')}${w.country?' · '+esc(w.country):''}</div>
          <div class="h-appellation">${esc(w.appellation||'')}</div>
          ${top?`<div class="h-score">${top.score}</div><div class="h-source">${esc(top.source)}</div>`:''}
          <div class="h-price">${price}</div>
          <div class="h-varietal">${esc(w.varietal||'')} ${esc(String(w.vintage||''))}</div>
        </div>
      </div>
      <div class="card-body">
        <div class="card-name">${esc(w.name||'')}</div>
        <div class="card-sub">${esc(w.producer||'')}</div>
      </div>
    </div>`;
  }).join('');
}

function filter(color, btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  if (color === 'recommended') {
    document.getElementById('viewGrid').style.display = 'none';
    document.getElementById('viewRec').style.display  = '';
    return;
  }
  document.getElementById('viewGrid').style.display = '';
  document.getElementById('viewRec').style.display  = 'none';
  currentColorFilter = color;
  applyFilters();
}

function openModal(id, skipIntro = false) {
  const w = wines.find(x => x.id == id);
  if (!w) return;

  const isSwitch  = current && current.id !== w.id;
  const firstTime = !introduced.has(w.id);

  // If switching wines, abort recognizer but keep audio + VAD running
  if (isSwitch) {
    try { recog && recog.abort(); } catch(_) {}
    recognizing = false;
    if (current) wineMemory[current.id] = [...memory];
  }

  // Restore this wine's memory
  memory  = wineMemory[w.id] ? [...wineMemory[w.id]] : [];
  current = w;

  // Add to bench (no-op if already there)
  addToBench(w);
  renderBench(); // re-render to update active state

  // Populate modal
  document.getElementById('mImg').src          = w.image_url || '';
  document.getElementById('mTag').textContent  = [w.appellation, w.region, w.country].filter(Boolean).join(' · ');
  document.getElementById('mName').textContent = w.name || '';
  document.getElementById('mProd').textContent = w.producer || '';
  document.getElementById('mPrice').textContent = w.retail ? `$${parseFloat(w.retail).toFixed(2)}` : '';

  document.getElementById('mChips').innerHTML = [
    w.varietal   && `<div class="chip">${esc(w.varietal)}</div>`,
    w.vintage    && `<div class="chip">Vintage <b>${w.vintage}</b></div>`,
    w.abv        && `<div class="chip">ABV <b>${w.abv}%</b></div>`,
    w.volume_ml  && `<div class="chip"><b>${w.volume_ml}ml</b></div>`,
  ].filter(Boolean).join('');

  const ratings = parseRatings(w.professional_ratings);
  document.getElementById('mRatings').innerHTML = ratings.map(r =>
    `<div class="rating-pill"><span class="s">${r.score}</span><span class="src">${esc(r.source)}</span></div>`
  ).join('');

  // Reset modal-chat and restore conversation bubbles for this wine
  const chat = document.getElementById('modalChat');
  const emptyEl = document.getElementById('modalChatEmpty');
  chat.innerHTML = '';
  chat.appendChild(emptyEl);
  emptyEl.style.display = '';
  const hist = wineMemory[w.id] || [];
  const convoHist = hist.filter(m => m.content !== '(auto-introduction)');
  if (convoHist.length) {
    emptyEl.style.display = 'none';
    convoHist.forEach(m => {
      const el = document.createElement('div');
      el.className = m.role === 'user' ? 'modal-chat-you' : 'modal-chat-bot';
      el.textContent = m.role === 'user' ? `"${m.content}"` : m.content;
      chat.appendChild(el);
    });
    chat.scrollTop = chat.scrollHeight;
  }

  updateMemUI();
  setOrb(vadActive ? orbState : 'idle');

  document.getElementById('modalBg').classList.add('open');

  // Auto-introduce on first visit
  if (firstTime && !skipIntro) {
    introduced.add(w.id);
    setTimeout(() => autoIntroduce(w), 400); // slight delay feels natural
  }
}

function closeModal() {
  // Save memory for this wine before closing
  if (current) wineMemory[current.id] = [...memory];
  document.getElementById('modalBg').classList.remove('open');
  // Don't stop audio or VAD — barkeep keeps talking from bench
  // Don't null current — orb still works for follow-ups
}

function bgClose(e) {
  if (e.target !== document.getElementById('modalBg')) return;
  // Check if the click position is over a wine-card in the grid below
  // by temporarily hiding the overlay and doing an elementFromPoint lookup
  const overlay = document.getElementById('modalBg');
  overlay.style.pointerEvents = 'none';
  const below = document.elementFromPoint(e.clientX, e.clientY);
  overlay.style.pointerEvents = '';
  const card = below?.closest('.wine-card');
  if (card) {
    const onclickAttr = card.getAttribute('onclick');
    const idMatch = onclickAttr && onclickAttr.match(/openModal\((\d+)\)/);
    closeModal();
    if (idMatch) setTimeout(() => openModal(Number(idMatch[1])), 0);
  } else {
    closeModal();
  }
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

// ── Recommended wines ─────────────────────────────────────────────────────────
// Each entry: { sourceWine: {id, name}, recs: [{...wine fields}] }
let recGroups = [];

/**
 * Push a pre-fetched list of similar wines into the rec panel for sourceWine.
 * Called directly from the rec_wines WS message handler (tool-driven path)
 * and from fetchSimilarWines (manual fetch fallback).
 */
function addSimilarToRecPanel(sourceWine, similar) {
  if (!sourceWine || !similar?.length) return;

  const benchIds = new Set(benchWines.map(w => w.id));
  const existingGroup = recGroups.find(g => g.sourceWine.id === sourceWine.id);
  const groupRecIds = new Set(existingGroup ? existingGroup.recs.map(r => r.id) : []);

  const newRecs = similar.filter(w => !benchIds.has(w.id) && !groupRecIds.has(w.id));
  if (!newRecs.length) return;

  let group = existingGroup;
  if (!group) {
    group = { sourceWine: { id: sourceWine.id, name: sourceWine.name }, recs: [] };
    recGroups.push(group);
  }
  newRecs.forEach(w => group.recs.push(w));

  renderRecPanel();
  const tab = document.getElementById('recTab');
  if (tab) tab.style.display = '';
  console.log('[REC]', sourceWine.name, '→', newRecs.map(w => w.name));
}

/**
 * Fallback: manually fetch similar wines for sourceWine and add to rec panel.
 * This path is only used if the LLM didn't call recommend_similar_wines_tool
 * (e.g. the user triggered rec intent but the LLM answered from memory).
 */
async function fetchSimilarWines(sourceWine) {
  if (sourceWine?.id == null) return;
  try {
    const res = await fetch(`${API}/similar/${sourceWine.id}?top_k=6`);
    if (!res.ok) { console.warn('[REC] /similar returned', res.status); return; }
    const similar = await res.json();
    addSimilarToRecPanel(sourceWine, similar);
  } catch (err) {
    console.error('[REC] fetchSimilarWines error:', err);
  }
}

function renderRecPanel() {
  const panel = document.getElementById('recPanel');
  if (!panel) return;

  recGroups.forEach(group => {
    // Find or create the group element
    let groupEl = panel.querySelector(`[data-source-id="${group.sourceWine.id}"]`);
    if (!groupEl) {
      groupEl = document.createElement('div');
      groupEl.className = 'rec-group';
      groupEl.dataset.sourceId = group.sourceWine.id;
      groupEl.innerHTML = `
        <div class="rec-group-title">
          Because you're looking at <span class="rec-group-wine">&nbsp;${esc(group.sourceWine.name)}</span>
        </div>
        <div class="rec-grid"></div>`;
      panel.appendChild(groupEl);
    }

    const grid = groupEl.querySelector('.rec-grid');
    // Track which wine ids are already rendered in this grid
    const rendered = new Set([...grid.querySelectorAll('.rec-card')].map(el => Number(el.dataset.wineId)));

    group.recs.forEach(w => {
      if (rendered.has(w.id)) return;
      const col = (w.color||'').toLowerCase();
      const dotCls = col==='red'?'d-red':col==='white'?'d-white':col==='sparkling'?'d-sparkling':col==='rose'||col==='rosé'?'d-rose':'d-other';
      const ratings = parseRatings(w.professional_ratings);
      const top = ratings[0];
      const price = w.retail ? `$${parseFloat(w.retail).toFixed(0)}` : '—';
      const imgHtml = w.image_url
        ? `<img src="${escAttr(w.image_url)}" alt="" loading="lazy" onerror="this.style.display='none'">`
        : `<div style="font-size:11px;color:var(--text3)">${col||'wine'}</div>`;
      const card = document.createElement('div');
      card.className = 'wine-card rec-card';
      card.dataset.wineId = w.id;
      card.onclick = () => openModal(w.id);
      card.innerHTML = `
        <div class="rec-badge">✦ rec</div>
        <div class="card-img">
          ${imgHtml}
          <div class="color-dot ${dotCls}"></div>
          <div class="card-hover">
            <div class="h-region">${esc(w.region||'')}${w.country?' · '+esc(w.country):''}</div>
            ${top?`<div class="h-score">${top.score}</div><div class="h-source">${esc(top.source)}</div>`:''}
            <div class="h-price">${price}</div>
          </div>
        </div>
        <div class="card-body">
          <div class="card-name">${esc(w.name||'')}</div>
          <div class="card-sub">${esc(w.producer||'')}</div>
        </div>`;
      grid.appendChild(card);
    });
  });
}

// ── Column filters ────────────────────────────────────────────────────────────
// activeFilters: { [colKey]: Set of selected values }
let activeFilters = {};
let currentColorFilter = 'all'; // track color tab separately

// Columns to skip (not useful as filters)
const SKIP_COLS = new Set(['id','image_url','professional_ratings','name','producer','description','notes','tasting_notes']);

// Label overrides for cleaner display
const COL_LABELS = {
  color:'Color', varietal:'Varietal', vintage:'Vintage', region:'Region',
  country:'Country', appellation:'Appellation', abv:'ABV', retail:'Price',
  volume_ml:'Volume', color:'Color'
};

function colLabel(key) {
  return COL_LABELS[key] || key.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
}

function buildFilterUI() {
  if (!wines.length) return;
  const cols = Object.keys(wines[0]).filter(k => !SKIP_COLS.has(k));
  const container = document.getElementById('filterCols');
  container.innerHTML = '';
  activeFilters = {};

  cols.forEach(col => {
    const vals = [...new Set(wines.map(w => w[col]).filter(v => v !== null && v !== undefined && v !== ''))]
      .sort((a,b) => String(a).localeCompare(String(b), undefined, {numeric:true}));
    if (vals.length < 2 || vals.length > 200) return;

    activeFilters[col] = new Set();

    const wrap = document.createElement('div');
    wrap.className = 'f-col';
    wrap.dataset.col = col;

    const btn = document.createElement('button');
    btn.className = 'f-col-btn';
    btn.id = `fbtn-${col}`;
    btn.textContent = colLabel(col);
    btn.onclick = (e) => { e.stopPropagation(); toggleDropdown(col); };

    wrap.appendChild(btn);
    container.appendChild(wrap);
  });

  document.addEventListener('click', closeAllDropdowns);
}

// ── Single portal dropdown appended to body ───────────────────────────────────
let _openDropCol = null;

function getPortal() {
  let p = document.getElementById('f-portal');
  if (!p) {
    p = document.createElement('div');
    p.id = 'f-portal';
    p.style.cssText = `
      position: fixed; z-index: 99;
      background: var(--surface2); border: 0.5px solid var(--border2);
      border-radius: var(--r); min-width: 200px; max-height: 260px;
      display: flex; flex-direction: column;
      box-shadow: 0 8px 32px rgba(0,0,0,0.55);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    `;
    p.onclick = e => e.stopPropagation();
    document.body.appendChild(p);
  }
  return p;
}

function toggleDropdown(col) {
  if (_openDropCol === col) { closeAllDropdowns(); return; }
  closeAllDropdowns();
  _openDropCol = col;

  const btn = document.getElementById(`fbtn-${col}`);
  const rect = btn.getBoundingClientRect();
  const portal = getPortal();

  // Search input
  const search = document.createElement('input');
  search.style.cssText = `
    width:100%; padding:8px 10px; font-size:11px; color:var(--text);
    background:transparent; border:none; border-bottom:0.5px solid var(--border2);
    outline:none; flex-shrink:0;
  `;
  search.placeholder = `Search ${colLabel(col)}…`;

  // Options list
  const list = document.createElement('div');
  list.style.cssText = 'overflow-y:auto; flex:1; scrollbar-width:thin;';

  portal.innerHTML = '';
  portal.appendChild(search);
  portal.appendChild(list);

  const renderOpts = (q) => {
    const vals = [...new Set(wines.map(w => w[col]).filter(v => v !== null && v !== undefined && v !== ''))]
      .sort((a,b) => String(a).localeCompare(String(b), undefined, {numeric:true}));
    const shown = q ? vals.filter(v => String(v).toLowerCase().includes(q.toLowerCase())) : vals;
    list.innerHTML = '';
    shown.forEach(v => {
      const opt = document.createElement('div');
      const selected = activeFilters[col]?.has(String(v));
      opt.style.cssText = `padding:7px 11px; font-size:11px; cursor:pointer; display:flex; align-items:center; gap:7px; color:${selected?'var(--gold)':'var(--text2)'}`;
      opt.innerHTML = `<span style="width:10px;font-size:10px">${selected?'✓':''}</span>${esc(String(v))}`;
      opt.onmouseenter = () => { if(!selected) opt.style.color = 'var(--text)'; };
      opt.onmouseleave = () => { if(!selected) opt.style.color = 'var(--text2)'; };
      opt.onclick = (e) => { e.stopPropagation(); toggleFilterValue(col, String(v)); renderOpts(search.value); };
      list.appendChild(opt);
    });
  };
  search.oninput = () => renderOpts(search.value);
  renderOpts('');

  // Position portal below the button, flip up if near bottom
  const spaceBelow = window.innerHeight - rect.bottom;
  const portalH = Math.min(260, 40 + Math.min(10, [...new Set(wines.map(w=>w[col]).filter(Boolean))].length) * 32);
  const top = spaceBelow > portalH ? rect.bottom + 6 : rect.top - portalH - 6;
  portal.style.top  = `${top}px`;
  portal.style.left = `${rect.left}px`;
  portal.style.display = 'flex';

  setTimeout(() => search.focus(), 50);
}

function closeAllDropdowns() {
  _openDropCol = null;
  const p = document.getElementById('f-portal');
  if (p) p.style.display = 'none';
}

function renderOptions(col, searchVal) { /* kept for compatibility — no-op now */ }

function toggleFilterValue(col, val) {
  if (!activeFilters[col]) activeFilters[col] = new Set();
  if (activeFilters[col].has(val)) activeFilters[col].delete(val);
  else activeFilters[col].add(val);
  updateColBtn(col);
  updatePills();
  applyFilters();
}

function updateColBtn(col) {
  const btn = document.querySelector(`.f-col[data-col="${col}"] .f-col-btn`);
  if (!btn) return;
  const count = activeFilters[col]?.size || 0;
  btn.classList.toggle('has-value', count > 0);
  btn.textContent = count > 0 ? `${colLabel(col)} (${count})` : colLabel(col);
}

function updatePills() {
  const pills = document.getElementById('filterPills');
  pills.innerHTML = '';
  let total = 0;
  Object.entries(activeFilters).forEach(([col, vals]) => {
    vals.forEach(v => {
      total++;
      const pill = document.createElement('div');
      pill.className = 'f-pill';
      pill.innerHTML = `<span>${colLabel(col)}: <b>${v}</b></span><span class="f-pill-x" onclick="toggleFilterValue('${col}','${v}')">✕</span>`;
      pills.appendChild(pill);
    });
  });
  const count = document.getElementById('filterActiveCount');
  const toggle = document.getElementById('filterToggle');
  if (total > 0) {
    count.textContent = total;
    count.style.display = '';
    toggle.classList.add('active');
  } else {
    count.style.display = 'none';
    toggle.classList.remove('active');
  }
}

function applyFilters() {
  let result = [...wines];
  // Apply color tab
  if (currentColorFilter !== 'all') {
    result = result.filter(w => (w.color||'').toLowerCase() === currentColorFilter);
  }
  // Apply each column filter (AND between columns, OR within a column)
  Object.entries(activeFilters).forEach(([col, vals]) => {
    if (!vals.size) return;
    result = result.filter(w => vals.has(String(w[col] ?? '')));
  });
  filtered = result;
  renderGrid();
  document.getElementById('countLabel').textContent = `${filtered.length} of ${wines.length} bottles`;
}

function toggleFilterPanel() {
  const panel = document.getElementById('filterPanel');
  const btn   = document.getElementById('filterToggle');
  panel.classList.toggle('open');
  btn.classList.toggle('active', panel.classList.contains('open'));
}

// ── Boot ──────────────────────────────────────────────────────────────────────
recog = setupRecognition();

window.onload = () => {
  loadWines();
  initSocket();

  // Orb button — works at any time, even before a wine is on the bench:
  //   • idle/listening  → start mic (enables browsing-mode voice)
  //   • speaking        → barge-in / stop
  //   • thinking        → no-op (already waiting on a reply)
  document.getElementById('orbBtn').addEventListener('click', () => {
    if (orbState === 'speaking')  { stopBarkeep(); return; }
    if (orbState === 'thinking')  { return; }
    if (!vadActive) {
      // First tap: start always-on mic (browser will prompt for permission)
      ensureVADRunning();
    } else if (!recognizing) {
      // Already listening but recognition not started — kick it manually
      startRecognition();
    }
  });
};