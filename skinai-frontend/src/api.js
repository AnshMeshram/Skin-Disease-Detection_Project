// Central API layer – all fetch calls to the FastAPI backend
const RAW_BASE = import.meta.env.VITE_API_URL;
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true' || RAW_BASE === 'demo';
const BASE = DEMO_MODE ? '' : (RAW_BASE || 'http://localhost:8080');

const CLASS_KEYS = [
  'MEL',
  'NV',
  'BCC',
  'AK',
  'BKL',
  'DF',
  'VASC',
  'SCC',
  'Healthy',
];

const PREDICTION_LABELS = {
  MEL: 'melanoma',
  NV: 'nevus',
  BCC: 'basal_cell_carcinoma',
  AK: 'actinic_keratosis',
  BKL: 'benign_keratosis',
  DF: 'dermatofibroma',
  VASC: 'vascular_lesion',
  SCC: 'squamous_cell_carcinoma',
  Healthy: 'healthy',
};

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function seedFromFile(file) {
  if (!file) return Date.now();
  const nameSeed = (file.name || '').split('').reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
  return (file.size || 0) + nameSeed + (file.lastModified || 0);
}

function buildDemoProbabilities(file) {
  const rand = mulberry32(seedFromFile(file));
  const raw = CLASS_KEYS.map(() => rand() + 0.05);
  const total = raw.reduce((a, b) => a + b, 0) || 1;
  const probs = raw.map((v) => v / total);
  const out = {};
  CLASS_KEYS.forEach((key, idx) => {
    out[key] = probs[idx];
  });
  return out;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function buildDemoPipelineImages(file) {
  try {
    const dataUrl = await fileToDataUrl(file);
    const base64 = String(dataUrl).split(',')[1] || '';
    return {
      original: base64,
      step1_resized: base64,
      step2_denoised: base64,
      step3_lab: base64,
      step4_clahe: base64,
      step5_no_hair: base64,
    };
  } catch {
    return {};
  }
}

export async function checkHealth() {
  if (DEMO_MODE) {
    return { status: 'demo', model_loaded: false, classes: 9 };
  }
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error('API offline');
  return res.json(); // { status, model_loaded, classes }
}

export async function predictImage(file) {
  if (DEMO_MODE) {
    const probabilities = buildDemoProbabilities(file);
    const entries = Object.entries(probabilities);
    const [topClass, topProb] = entries.reduce(
      (best, current) => (current[1] > best[1] ? current : best),
      entries[0],
    );
    const prediction = PREDICTION_LABELS[topClass] || topClass.toLowerCase();
    const pipelineImages = await buildDemoPipelineImages(file);
    return {
      status: 'success',
      prediction,
      class_index: CLASS_KEYS.indexOf(topClass),
      confidence: Number(topProb),
      is_uncertain: Number(topProb) < 0.7,
      probabilities,
      preprocessing_applied: false,
      pipeline_images: pipelineImages,
    };
  }
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/predict`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
  // Returns: { status, prediction, class_index, confidence, is_uncertain,
  //            probabilities, preprocessing_applied }
}

export async function getMetrics(model = null, fold = null) {
  if (DEMO_MODE) {
    return {
      model: 'demo',
      fold: 0,
      run_dir: 'demo',
      current: {
        epoch: 0,
        train_acc: null,
        val_acc: null,
        val_balanced_accuracy: null,
        val_f1: null,
        lr: null,
      },
      history: { train_acc: [], val_acc: [], val_balanced_accuracy: [] },
      latest_checkpoint: null,
      best_checkpoint: null,
      updated_at: new Date().toISOString(),
    };
  }
  const params = new URLSearchParams();
  if (model) params.set('model', model);
  if (fold !== null) params.set('fold', fold);
  const res = await fetch(`${BASE}/metrics/latest?${params}`);
  if (!res.ok) throw new Error('Metrics unavailable');
  return res.json();
}
