/**
 * SkinAI — Skin Disease Detection Frontend
 * Connects to FastAPI backend at /predict and /preprocess
 *
 * Backend endpoints expected:
 *   POST /predict          → { disease, confidence, model, ensemble_vote, class_probabilities }
 *   POST /preprocess       → { steps: [ { name, image_b64 }, ... ] }
 *   GET  /health           → { status: "ok" }
 *   GET  /classes          → [ { id, name, abbreviation } ]
 *   GET  /models           → [ { name, params, input_size, attention } ]
 */

// ─────────────────────────────────────────────
// CONFIG — update to your FastAPI server URL
// ─────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

// ─────────────────────────────────────────────
// DISEASE → SYMPTOMS MAP
// ─────────────────────────────────────────────
const DISEASE_META = {
    'Melanoma':                { symptoms: 'Asymmetric, irregular borders, multicolor',   risk: 'high'   },
    'Melanocytic nevi':        { symptoms: 'Uniform color, round/oval, well-defined',      risk: 'low'    },
    'Basal cell carcinoma':    { symptoms: 'Pearly bump, waxy, may bleed',                 risk: 'medium' },
    'Actinic keratosis':       { symptoms: 'Rough scaly patch, redness, itching',          risk: 'medium' },
    'Benign keratosis':        { symptoms: 'Waxy "stuck-on" look, varied color',           risk: 'low'    },
    'Dermatofibroma':          { symptoms: 'Firm bump, dimples when pinched, brown',       risk: 'low'    },
    'Vascular lesion':         { symptoms: 'Red/purple, flat or raised, visible vessels',  risk: 'low'    },
    'Squamous cell carcinoma': { symptoms: 'Firm red nodule, flat lesion, may crust',      risk: 'high'   },
    'Healthy skin':            { symptoms: 'No lesion detected, normal appearance',        risk: 'none'   },
};

const CLASSES = [
    { id: 0, name: 'Melanoma',                abbr: 'MEL'  },
    { id: 1, name: 'Melanocytic nevi',         abbr: 'NV'   },
    { id: 2, name: 'Basal cell carcinoma',     abbr: 'BCC'  },
    { id: 3, name: 'Actinic keratosis',        abbr: 'AK'   },
    { id: 4, name: 'Benign keratosis',         abbr: 'BKL'  },
    { id: 5, name: 'Dermatofibroma',           abbr: 'DF'   },
    { id: 6, name: 'Vascular lesion',          abbr: 'VASC' },
    { id: 7, name: 'Squamous cell carcinoma',  abbr: 'SCC'  },
    { id: 8, name: 'Healthy skin',             abbr: 'HLT'  },
];

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────
let uploadedFile = null;
let isAnalyzing  = false;

// ─────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    buildClassGrid();
    checkAPIHealth();
    initScrollReveal();
    initDragDrop();
    initNavScroll();
});

// ─────────────────────────────────────────────
// BUILD DISEASE CLASS GRID
// ─────────────────────────────────────────────
function buildClassGrid() {
    const grid = document.getElementById('classGrid');
    if (!grid) return;
    grid.innerHTML = CLASSES.map((c, i) => `
        <div class="class-card">
            <span class="class-num">${String(i + 1).padStart(2, '0')}</span>
            <div>
                <div class="class-name">${c.name}</div>
                <div class="class-abbr">${c.abbr}</div>
            </div>
        </div>
    `).join('');
}

// ─────────────────────────────────────────────
// API HEALTH CHECK
// ─────────────────────────────────────────────
async function checkAPIHealth() {
    const dot  = document.getElementById('apiDot');
    const text = document.getElementById('apiStatus');
    try {
        const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        if (r.ok) {
            dot.className  = 'status-dot online';
            text.textContent = 'API Online';
        } else { throw new Error(); }
    } catch {
        dot.className  = 'status-dot offline';
        text.textContent = 'Demo Mode';
    }
}

// ─────────────────────────────────────────────
// FILE HANDLING
// ─────────────────────────────────────────────
function triggerUpload() {
    document.getElementById('fileInput').click();
}

function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
        showError('Please upload a valid image file (JPG, PNG, JPEG).');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showError('File too large. Please upload an image under 10 MB.');
        return;
    }

    uploadedFile = file;
    hideError();

    const reader = new FileReader();
    reader.onload = (ev) => {
        const img = document.getElementById('previewImg');
        img.src = ev.target.result;
        img.style.display = 'block';

        document.getElementById('camPlaceholder').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('scanLine').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// ─────────────────────────────────────────────
// ANALYZE
// ─────────────────────────────────────────────
async function analyze() {
    if (!uploadedFile || isAnalyzing) return;

    isAnalyzing = true;
    setLoadingState(true);
    hideError();
    resetPipeline();

    try {
        // Fetch preprocessing and prediction in parallel
        const [predResult, prepResult] = await Promise.allSettled([
            callPredict(),
            callPreprocess(),
        ]);

        // Animate pipeline steps
        animatePipelineSteps(
            prepResult.status === 'fulfilled' ? prepResult.value : null
        );

        // Show results
        if (predResult.status === 'fulfilled') {
            setTimeout(() => showResults(predResult.value), 1200);
        } else {
            // Demo fallback when API unavailable
            console.warn('API unavailable, using demo data.');
            setTimeout(() => showResults(getDemoResult()), 1200);
        }

    } catch (err) {
        showError('Analysis failed. Running in demo mode.');
        setTimeout(() => showResults(getDemoResult()), 1200);
    } finally {
        isAnalyzing = false;
        setLoadingState(false);
    }
}

async function callPredict() {
    const form = new FormData();
    form.append('file', uploadedFile);
    const r = await fetch(`${API_BASE}/predict`, { method: 'POST', body: form });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
}

async function callPreprocess() {
    const form = new FormData();
    form.append('file', uploadedFile);
    const r = await fetch(`${API_BASE}/preprocess`, { method: 'POST', body: form });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
}

function getDemoResult() {
    const names  = Object.keys(DISEASE_META);
    const disease = names[Math.floor(Math.random() * names.length)];
    return {
        disease,
        confidence:      (70 + Math.random() * 25).toFixed(1),
        model:           'EfficientNet-B3 (Ensemble)',
        ensemble_vote:   '3/3 models agree',
        demo:            true,
    };
}

// ─────────────────────────────────────────────
// PIPELINE ANIMATION
// ─────────────────────────────────────────────
function resetPipeline() {
    for (let i = 0; i < 6; i++) {
        const el = document.getElementById(`pstep-${i}`);
        if (el) { el.classList.remove('active', 'done'); }
        const img = document.getElementById(`pimg-${i}`);
        if (img) { img.style.display = 'none'; }
    }
}

function animatePipelineSteps(prepData) {
    for (let i = 0; i < 6; i++) {
        setTimeout(() => {
            const el = document.getElementById(`pstep-${i}`);
            if (!el) return;
            if (i > 0) {
                const prev = document.getElementById(`pstep-${i - 1}`);
                if (prev) { prev.classList.remove('active'); prev.classList.add('done'); }
            }
            el.classList.add('active');

            // If API returned base64 images, display them
            if (prepData && prepData.steps && prepData.steps[i]) {
                const imgEl = document.getElementById(`pimg-${i}`);
                const placeholder = el.querySelector('.pipe-placeholder');
                if (imgEl) {
                    imgEl.src = `data:image/jpeg;base64,${prepData.steps[i].image_b64}`;
                    imgEl.style.display = 'block';
                    if (placeholder) placeholder.style.display = 'none';
                }
            }

            // Mark last step done
            if (i === 5) {
                setTimeout(() => {
                    el.classList.remove('active');
                    el.classList.add('done');
                }, 500);
            }
        }, i * 200);
    }
}

// ─────────────────────────────────────────────
// SHOW RESULTS
// ─────────────────────────────────────────────
function showResults(data) {
    const card = document.getElementById('resultsCard');
    card.classList.add('visible');

    const disease    = data.disease || data.prediction || 'Unknown';
    const confidence = parseFloat(data.confidence || data.confidence_score || 0);
    const meta       = DISEASE_META[disease] || { symptoms: 'Consult a dermatologist', risk: 'unknown' };

    document.getElementById('diseaseName').textContent = disease;
    document.getElementById('confScore').textContent   = confidence.toFixed(1) + '%';
    document.getElementById('symptoms').textContent    = meta.symptoms;
    document.getElementById('modelUsed').textContent   = data.model || 'EfficientNet-B3 Ensemble';
    document.getElementById('ensembleVote').textContent = data.ensemble_vote || '—';

    // Result image
    const resultImg = document.getElementById('resultImg');
    resultImg.src   = document.getElementById('previewImg').src;

    // Demo badge
    if (data.demo) {
        document.getElementById('resultImgBadge').textContent = 'Demo';
        document.getElementById('resultImgBadge').style.color  = '#FCD34D';
        document.getElementById('resultImgBadge').style.borderColor = 'rgba(245,158,11,0.3)';
    }

    // Animate confidence bar
    setTimeout(() => {
        document.getElementById('confBar').style.width = Math.min(confidence, 100) + '%';
    }, 100);

    // Scroll to results
    card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─────────────────────────────────────────────
// RESET
// ─────────────────────────────────────────────
function resetAnalysis() {
    uploadedFile = null;

    document.getElementById('previewImg').style.display   = 'none';
    document.getElementById('previewImg').src             = '';
    document.getElementById('camPlaceholder').style.display = 'flex';
    document.getElementById('scanLine').style.display    = 'none';
    document.getElementById('analyzeBtn').disabled        = true;
    document.getElementById('btnText').textContent        = 'Analyze Image';
    document.getElementById('fileInput').value            = '';
    document.getElementById('resultsCard').classList.remove('visible');
    document.getElementById('confBar').style.width        = '0%';
    resetPipeline();
    hideError();

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ─────────────────────────────────────────────
// UI HELPERS
// ─────────────────────────────────────────────
function setLoadingState(loading) {
    const btn     = document.getElementById('analyzeBtn');
    const spinner = document.getElementById('spinner');
    const btnIcon = document.getElementById('btnIcon');
    const btnText = document.getElementById('btnText');

    btn.disabled             = loading;
    spinner.style.display    = loading ? 'block'  : 'none';
    btnIcon.style.display    = loading ? 'none'   : 'block';
    btnText.textContent      = loading ? 'Analyzing...' : 'Analyze Image';
}

function showError(msg) {
    const el = document.getElementById('errorMsg');
    el.textContent   = msg;
    el.style.display = 'block';
}

function hideError() {
    document.getElementById('errorMsg').style.display = 'none';
}

// ─────────────────────────────────────────────
// MODEL TABS
// ─────────────────────────────────────────────
function switchTab(btn, panelId) {
    document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.model-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(panelId).classList.add('active');
}

// ─────────────────────────────────────────────
// NAV ACTIVE STATE
// ─────────────────────────────────────────────
function setActive(el) {
    document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
    el.classList.add('active');
}

// ─────────────────────────────────────────────
// SCROLL REVEAL
// ─────────────────────────────────────────────
function initScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(e => {
            if (e.isIntersecting) { e.target.classList.add('in-view'); }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}

// ─────────────────────────────────────────────
// NAV SCROLL EFFECT
// ─────────────────────────────────────────────
function initNavScroll() {
    const nav = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        nav.classList.toggle('scrolled', window.scrollY > 60);

        // Update active nav link based on scroll position
        const sections = ['home', 'model', 'students'];
        for (const id of sections.reverse()) {
            const el = document.getElementById(id);
            if (el && window.scrollY >= el.offsetTop - 100) {
                document.querySelectorAll('.nav-links a').forEach(a => {
                    a.classList.toggle('active', a.getAttribute('href') === `#${id}`);
                });
                break;
            }
        }
    });
}

// ─────────────────────────────────────────────
// DRAG & DROP
// ─────────────────────────────────────────────
function initDragDrop() {
    const box = document.getElementById('cameraBox');
    if (!box) return;

    box.addEventListener('dragover', e => {
        e.preventDefault();
        box.style.borderColor = 'var(--blue)';
        box.style.background  = 'var(--blue-dim)';
    });

    box.addEventListener('dragleave', () => {
        box.style.borderColor = '';
        box.style.background  = '';
    });

    box.addEventListener('drop', e => {
        e.preventDefault();
        box.style.borderColor = '';
        box.style.background  = '';
        const file = e.dataTransfer.files[0];
        if (file) {
            const dt = new DataTransfer();
            dt.items.add(file);
            document.getElementById('fileInput').files = dt.files;
            handleFile({ target: { files: [file] } });
        }
    });
}
