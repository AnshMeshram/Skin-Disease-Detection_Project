// Central API layer – all fetch calls to the FastAPI backend
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8080";

export async function checkHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error("API offline");
  return res.json(); // { status, model_loaded, classes }
}

export async function predictImage(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/predict`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
  // Returns: { status, prediction, class_index, confidence, is_uncertain,
  //            probabilities, preprocessing_applied }
}

export async function getMetrics(model = null, fold = null) {
  const params = new URLSearchParams();
  if (model) params.set("model", model);
  if (fold !== null) params.set("fold", fold);
  const res = await fetch(`${BASE}/metrics/latest?${params}`);
  if (!res.ok) throw new Error("Metrics unavailable");
  return res.json();
}
