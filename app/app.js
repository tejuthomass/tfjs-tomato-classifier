/* Tomato Leaf Check - all inference runs locally with TensorFlow.js. */

const LOW_CONFIDENCE = 0.6;

// ---------------------------------------------------------------- advisory content (static)
// Sources (checked Nov 2026):
//  Early blight: NC State Extension, "Early Blight of Tomato" (Meadows, Hinchliffe, Villani; rev. Apr 2025)
//  Late blight:  Washington State University HortSense, "Tomato: Late blight"
const SOURCES = {
  Early_Blight: { name: 'NC State Extension – Early Blight of Tomato', url: 'https://content.ces.ncsu.edu/early-blight-of-tomato' },
  Late_Blight: { name: 'WSU HortSense – Tomato: Late blight', url: 'https://hortsense.cahnrs.wsu.edu/fact-sheet/tomato-late-blight/' },
  Healthy: { name: 'NC State Extension; WSU HortSense (general prevention)', url: 'https://content.ces.ncsu.edu/early-blight-of-tomato' },
};

const ADVICE = {
  en: {
    heading: '3. Advice',
    names: { Healthy: 'Healthy', Early_Blight: 'Early Blight', Late_Blight: 'Late Blight' },
    bands: { Low: 'Low', Medium: 'Medium', High: 'High' },
    Healthy: {
      title: 'Looks healthy',
      items: [
        'Check lower leaves weekly.',
        'Water the soil, not the leaves.',
        'Space plants for airflow.',
      ],
    },
    Early_Blight: {
      title: 'Early blight',
      items: [
        'Remove infected lower leaves.',
        'Mulch and stake plants.',
        'No tomatoes in this spot for 2–3 years.',
        'Fungicide (e.g. chlorothalonil) every 7–14 days, per label.',
      ],
    },
    Late_Blight: {
      title: 'Late blight: act fast',
      items: [
        'Remove infected plants now.',
        'Destroy or bury 2 ft deep. Never compost.',
        'No overhead watering. Keep away from potatoes.',
        'Fungicide at first sign, per label. Copper alone isn’t enough.',
      ],
    },
    consultTitle: 'Consult an extension officer',
    consult: 'Screening only, not a diagnosis. Confirm with your local extension officer or KVK before spraying.',
    source: 'Source',
  },
  hi: {
    heading: '3. सलाह',
    names: { Healthy: 'स्वस्थ', Early_Blight: 'अगेती झुलसा', Late_Blight: 'पछेती झुलसा' },
    bands: { Low: 'कम', Medium: 'मध्यम', High: 'अधिक' },
    Healthy: {
      title: 'स्वस्थ दिख रही है',
      items: [
        'हर हफ़्ते निचली पत्तियाँ जाँचें।',
        'मिट्टी में पानी दें, पत्तियों पर नहीं।',
        'हवा के लिए पौधों में दूरी रखें।',
      ],
    },
    Early_Blight: {
      title: 'अगेती झुलसा',
      items: [
        'संक्रमित निचली पत्तियाँ हटाएँ।',
        'मल्च बिछाएँ, पौधों को सहारा दें।',
        '2–3 साल इस जगह टमाटर न लगाएँ।',
        'फफूंदनाशक (जैसे क्लोरोथैलोनिल) हर 7–14 दिन, लेबल के अनुसार।',
      ],
    },
    Late_Blight: {
      title: 'पछेती झुलसा: तुरंत कदम उठाएँ',
      items: [
        'संक्रमित पौधे तुरंत हटाएँ।',
        'नष्ट करें या 2 फुट गहरा गाड़ें। कम्पोस्ट न करें।',
        'ऊपर से सिंचाई न करें। आलू से दूर रखें।',
        'पहले लक्षण पर फफूंदनाशक, लेबल के अनुसार। केवल कॉपर काफ़ी नहीं।',
      ],
    },
    consultTitle: 'कृषि विस्तार अधिकारी से सलाह लें',
    consult: 'यह केवल जाँच है, निदान नहीं। छिड़काव से पहले स्थानीय कृषि विस्तार अधिकारी या KVK से पुष्टि करें।',
    source: 'स्रोत',
  },
};

// ---------------------------------------------------------------- state & elements
const $ = (id) => document.getElementById(id);
const els = {
  status: $('status'), statusText: $('status-text'), video: $('video'), preview: $('preview'),
  overlay: $('overlay'), guide: $('guide'), placeholder: $('placeholder'), btnCamera: $('btn-camera'),
  btnScan: $('btn-scan'), file: $('file'), resultCard: $('result-card'), advisoryCard: $('advisory-card'),
  resLabel: $('res-label'), resBars: $('res-bars'), resSev: $('res-sev'),
  resMs: $('res-ms'), resWarn: $('res-warn'), chkOverlay: $('chk-overlay'), advBody: $('adv-body'),
  advHeading: $('adv-heading'),
};
const state = { model: null, meta: null, stream: null, lang: 'en', last: null };
try { state.lang = localStorage.getItem('lang') || 'en'; } catch (e) { /* storage blocked */ }

function setStatus(text, kind) {
  els.statusText.textContent = text;
  els.status.className = 'pill' + (kind ? ' ' + kind : '');
}

// ---------------------------------------------------------------- model
async function loadModel() {
  try {
    state.meta = await (await fetch('model/metadata.json')).json();
    state.model = await tf.loadLayersModel('model/model.json');
    tf.tidy(() => state.model.predict(tf.zeros([1, state.meta.input_size, state.meta.input_size, 3])));  // warm-up
    setStatus(navigator.onLine ? 'Model ready' : 'Offline · ready', 'ok');
    els.btnScan.disabled = !state.stream;
  } catch (err) {
    console.error(err);
    setStatus('Model missing', 'warn');
    alert('Could not load the model. Run scripts/evaluate_models.py first, then reload.');
  }
}

// ---------------------------------------------------------------- camera
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Live camera needs HTTPS or localhost. Use "Upload photo" instead – it can open the camera too.');
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 1280 } },
      audio: false,
    });
    els.video.srcObject = state.stream;
    await els.video.play();
    show(els.video); show(els.guide); hide(els.preview); hide(els.overlay); hide(els.placeholder);
    els.btnScan.disabled = !state.model;
    els.btnCamera.textContent = 'Retake';
  } catch (err) {
    alert('Camera unavailable: ' + err.message + '\nYou can still use "Upload photo".');
  }
}

// Draw the centre square of any image/video onto the 224x224 preview canvas
function drawCentreSquare(source, w, h) {
  const size = state.meta ? state.meta.input_size : 224;
  const s = Math.min(w, h);
  const ctx = els.preview.getContext('2d', { willReadFrequently: true });
  els.preview.width = els.preview.height = els.overlay.width = els.overlay.height = size;
  ctx.drawImage(source, (w - s) / 2, (h - s) / 2, s, s, 0, 0, size, size);
  show(els.preview); hide(els.video); hide(els.guide); hide(els.placeholder);
  els.overlay.classList.toggle('hidden', !els.chkOverlay.checked);
}

function scanFromCamera() {
  drawCentreSquare(els.video, els.video.videoWidth, els.video.videoHeight);
  analyse();
}

function scanFromFile(file) {
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    drawCentreSquare(img, img.naturalWidth, img.naturalHeight);
    URL.revokeObjectURL(img.src);
    analyse();
  };
  img.src = URL.createObjectURL(file);
}

// ---------------------------------------------------------------- severity (same rules as evaluate_models.py)
function computeSeverity(imageData, cfg) {
  const d = imageData.data;
  const mask = new Uint8Array(d.length / 4);
  let leaf = 0, green = 0;
  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    const r = d[i] / 255, g = d[i + 1] / 255, b = d[i + 2] / 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), delta = mx - mn;
    const sat = mx === 0 ? 0 : delta / mx;
    if (sat <= cfg.leaf_min_sat || mx <= cfg.leaf_min_val) continue;   // background / shadow
    let h = 0;
    if (delta !== 0) {
      if (mx === r) h = ((((g - b) / delta) % 6) + 6) % 6;
      else if (mx === g) h = (b - r) / delta + 2;
      else h = (r - g) / delta + 4;
      h *= 60;
    }
    leaf++;
    if (h >= cfg.green_hue[0] && h <= cfg.green_hue[1]) green++;
    else mask[p] = 1;
  }
  if (leaf < 50) return { pct: null, band: null, mask };
  const pct = 100 * (1 - green / leaf);
  const band = (cfg.bands.find((b) => pct < b.below) || cfg.bands[cfg.bands.length - 1]).name;
  return { pct, band, mask };
}

function drawOverlay(mask) {
  const ctx = els.overlay.getContext('2d');
  const img = ctx.createImageData(els.overlay.width, els.overlay.height);
  for (let p = 0; p < mask.length; p++) {
    if (mask[p]) { img.data[p * 4] = 220; img.data[p * 4 + 1] = 38; img.data[p * 4 + 2] = 38; img.data[p * 4 + 3] = 150; }
  }
  ctx.putImageData(img, 0, 0);
}

// ---------------------------------------------------------------- classify + render
async function analyse() {
  if (!state.model) { alert('Model is still loading.'); return; }
  const t0 = performance.now();
  const probsT = tf.tidy(() => {
    const x = tf.browser.fromPixels(els.preview).toFloat().div(127.5).sub(1).expandDims(0);
    return state.model.predict(x);
  });
  const probs = Array.from(await probsT.data());
  probsT.dispose();
  const ms = performance.now() - t0;

  const ctx = els.preview.getContext('2d', { willReadFrequently: true });
  const sev = computeSeverity(ctx.getImageData(0, 0, els.preview.width, els.preview.height), state.meta.severity);
  drawOverlay(sev.mask);

  const best = probs.indexOf(Math.max(...probs));
  state.last = { label: state.meta.labels[best], probs, conf: probs[best], sev, ms };
  render();
}

function render() {
  const r = state.last;
  if (!r) return;
  const t = ADVICE[state.lang];
  const names = ADVICE.en.names;  // result card stays English; advisory card is bilingual

  els.resLabel.textContent = names[r.label];
  els.resBars.innerHTML = state.meta.labels.map((l, i) => `
    <div class="bar-row"><span>${names[l]}</span>
      <div class="bar"><span style="width:${(100 * r.probs[i]).toFixed(1)}%"></span></div>
      <span class="pct">${(100 * r.probs[i]).toFixed(0)}%</span></div>`).join('');

  if (r.label === 'Healthy') {
    els.resSev.textContent = 'Not applicable';
  } else if (r.sev.pct === null) {
    els.resSev.textContent = 'No leaf found';
  } else {
    els.resSev.innerHTML = `<span class="badge ${r.sev.band}">${r.sev.band}</span> ${r.sev.pct.toFixed(0)}%`;
  }
  els.resMs.textContent = `${r.ms.toFixed(0)} ms`;
  els.resWarn.classList.toggle('hidden', r.conf >= LOW_CONFIDENCE);
  show(els.resultCard);

  // Advisory card
  const a = t[r.label];
  const src = SOURCES[r.label];
  els.advHeading.textContent = t.heading;
  els.advBody.innerHTML = `
    <p class="adv-title">${a.title}</p>
    <ul class="adv-list">${a.items.map((i) => `<li>${i}</li>`).join('')}</ul>
    <div class="consult"><strong>${t.consultTitle}</strong>${t.consult}</div>
    <p class="source">${t.source}: <a href="${src.url}" target="_blank" rel="noopener">${src.name}</a></p>`;
  document.querySelectorAll('.segmented button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.lang === state.lang));
  document.documentElement.lang = state.lang;
  show(els.advisoryCard);
}

// ---------------------------------------------------------------- helpers & wiring
function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

els.btnCamera.addEventListener('click', startCamera);
els.btnScan.addEventListener('click', scanFromCamera);
els.file.addEventListener('change', (e) => { scanFromFile(e.target.files[0]); e.target.value = ''; });
els.chkOverlay.addEventListener('change', () => els.overlay.classList.toggle('hidden', !els.chkOverlay.checked));
document.querySelectorAll('.segmented button').forEach((b) => b.addEventListener('click', () => {
  state.lang = b.dataset.lang;
  try { localStorage.setItem('lang', state.lang); } catch (e) { /* storage blocked */ }
  render();
}));
window.addEventListener('online', () => state.model && setStatus('Model ready', 'ok'));
window.addEventListener('offline', () => state.model && setStatus('Offline · ready', 'ok'));

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch((err) => console.warn('Service worker not registered:', err));
}
loadModel();
