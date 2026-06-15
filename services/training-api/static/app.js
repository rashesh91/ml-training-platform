const API = '';
let selectedFile = null;
let pollInterval = null;

// --- File drop zone ---

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('dataset-file');
const dropLabel = document.getElementById('drop-label');
const submitBtn = document.getElementById('submit-btn');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});
fileInput.addEventListener('change', e => { if (e.target.files[0]) setFile(e.target.files[0]); });

function setFile(f) {
  selectedFile = f;
  dropLabel.textContent = `✓ ${f.name} (${(f.size / 1024).toFixed(1)} KB)`;
  dropLabel.style.color = 'var(--text)';
  submitBtn.disabled = false;
  document.getElementById('submit-label').textContent = 'Start Fine-tuning';
}

// --- Submit job ---

async function submitJob() {
  if (!selectedFile) return;

  submitBtn.disabled = true;
  document.getElementById('submit-label').innerHTML = '<span class="spinner"></span> Submitting...';

  const fd = new FormData();
  fd.append('dataset', selectedFile);
  fd.append('model_name', document.getElementById('model-name').value.trim() || 'my-model');
  fd.append('base_model', document.getElementById('base-model').value);
  fd.append('lora_rank', document.getElementById('lora-rank').value);
  fd.append('epochs', document.getElementById('epochs').value);
  fd.append('learning_rate', document.getElementById('lr').value);

  try {
    const res = await fetch(`${API}/api/jobs`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const job = await res.json();
    toast(`Job ${job.job_id} queued!`, 'success');
    showTab('jobs', document.querySelector('.tab'));
    document.getElementById('submit-label').textContent = 'Start Fine-tuning';
    submitBtn.disabled = false;
    pollJobs();
  } catch (e) {
    toast(`Error: ${e.message}`, 'error');
    submitBtn.disabled = false;
    document.getElementById('submit-label').textContent = 'Start Fine-tuning';
  }
}

// --- Poll jobs ---

function pollJobs() {
  if (pollInterval) clearInterval(pollInterval);
  loadJobs();
  pollInterval = setInterval(loadJobs, 4000);
}

async function loadJobs() {
  try {
    const res = await fetch(`${API}/api/jobs`);
    const jobs = await res.json();
    renderJobs(jobs);
    updateMetrics(jobs);
    // Stop polling when no active jobs
    const active = jobs.some(j => j.status === 'running' || j.status === 'queued' || j.status === 'evaluating');
    if (!active && pollInterval) { clearInterval(pollInterval); pollInterval = null; }
  } catch (e) { /* silent */ }
}

function renderJobs(jobs) {
  const tbody = document.getElementById('jobs-body');
  if (!jobs.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty">No jobs yet. Submit your first fine-tuning job →</td></tr>';
    return;
  }
  tbody.innerHTML = jobs.map(j => {
    const pct = j.progress != null ? Math.round(j.progress * 100) : null;
    const progressCell = pct != null
      ? `<div style="display:flex;align-items:center;gap:6px">
           <div style="flex:1;background:var(--border);border-radius:4px;height:6px;min-width:80px">
             <div style="width:${pct}%;background:var(--blue);height:6px;border-radius:4px;transition:width 0.5s"></div>
           </div>
           <span style="font-size:11px;color:var(--muted)">${pct}%</span>
           ${j.epoch != null ? `<span style="font-size:11px;color:var(--muted)">ep${j.epoch.toFixed(1)}</span>` : ''}
         </div>`
      : '—';
    const lossCell = j.loss != null ? `<span style="color:var(--warn);font-size:12px">${j.loss.toFixed(4)}</span>` : '—';
    return `
    <tr>
      <td><code style="font-size:12px">${j.job_id}</code></td>
      <td>${j.model_name}</td>
      <td style="color:var(--muted);font-size:12px">${shortModel(j.base_model)}</td>
      <td>${statusBadge(j.status)}</td>
      <td style="min-width:140px">${progressCell}</td>
      <td>${lossCell}</td>
      <td style="color:var(--muted);font-size:12px">${relTime(j.created_at)}</td>
      <td><button class="btn btn-sm" style="background:var(--border);color:var(--text)" onclick="showLogs('${j.job_id}','${j.model_name}')">Logs</button></td>
    </tr>`;
  }).join('');
}

async function showLogs(jobId, modelName) {
  const panel = document.getElementById('log-panel');
  const content = document.getElementById('log-content');
  document.getElementById('log-panel-title').textContent = `Logs — ${modelName} (${jobId})`;
  panel.style.display = '';
  content.textContent = 'Loading…';
  try {
    const res = await fetch(`${API}/api/jobs/${jobId}/logs`);
    const data = await res.json();
    content.textContent = data.logs.length ? data.logs.join('\n') : '(no logs yet)';
    content.scrollTop = content.scrollHeight;
  } catch (e) {
    content.textContent = `Error: ${e.message}`;
  }
}

async function loadModels() {
  try {
    const res = await fetch(`${API}/api/models`);
    const models = await res.json();
    renderModels(models);
    document.getElementById('m-models').textContent = models.length;
    document.getElementById('m-prod').textContent = models.filter(m => m.stage === 'Production').length;
  } catch (e) { /* silent */ }
}

function renderModels(models) {
  const tbody = document.getElementById('models-body');
  if (!models.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty">No registered models yet. Complete a training job first.</td></tr>';
    return;
  }
  tbody.innerHTML = models.map(m => `
    <tr>
      <td><strong>${m.name}</strong></td>
      <td>v${m.version}</td>
      <td><span class="stage-pill ${m.stage || 'None'}">${m.stage || 'None'}</span></td>
      <td>${m.eval_score != null ? `${(m.eval_score*100).toFixed(1)}%` : '—'}</td>
      <td style="color:var(--muted);font-size:12px">${relTime(m.created_at)}</td>
      <td>
        ${m.stage !== 'Production' ? `<button class="btn btn-sm btn-purple" onclick="deployModel('${m.name}','${m.version}')">Deploy →</button>` : '<span style="color:#3fb950;font-size:12px">✓ In Production</span>'}
      </td>
    </tr>
  `).join('');
}

async function deployModel(name, version) {
  try {
    const res = await fetch(`${API}/api/models/${name}/deploy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ version }),
    });
    if (!res.ok) throw new Error(await res.text());
    toast(`${name} v${version} promoted to Production`, 'success');
    loadModels();
  } catch (e) {
    toast(`Deploy failed: ${e.message}`, 'error');
  }
}

function updateMetrics(jobs) {
  document.getElementById('m-total').textContent = jobs.length;
  document.getElementById('m-succeeded').textContent = jobs.filter(j => j.status === 'succeeded').length;
  document.getElementById('m-running').textContent = jobs.filter(j => ['running','evaluating','queued'].includes(j.status)).length;
  document.getElementById('m-failed').textContent = jobs.filter(j => j.status === 'failed').length;
}

// --- Tabs ---

function showTab(name, el) {
  ['jobs','models','metrics'].forEach(t => {
    document.getElementById(`tab-${t}`).style.display = 'none';
  });
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(`tab-${name}`).style.display = '';
  el.classList.add('active');
  if (name === 'models') loadModels();
  if (name === 'metrics') { loadJobs(); loadModels(); }
}

// --- Sample dataset download ---

function downloadSample() {
  const lines = [
    { prompt: "What is machine learning?", response: "Machine learning is a branch of AI where systems learn patterns from data to make predictions or decisions." },
    { prompt: "Explain gradient descent in simple terms.", response: "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function by moving in the direction of steepest descent." },
    { prompt: "What is overfitting?", response: "Overfitting occurs when a model learns noise in training data too well, resulting in poor performance on new, unseen data." },
    { prompt: "What is a transformer architecture?", response: "A transformer is a deep learning architecture that uses self-attention mechanisms to process sequences in parallel, enabling efficient training on large datasets." },
    { prompt: "What is LoRA fine-tuning?", response: "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that trains only a small set of additional weight matrices, reducing memory and compute requirements significantly." },
    { prompt: "What is the difference between supervised and unsupervised learning?", response: "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data without explicit guidance." },
    { prompt: "What is a GPU?", response: "A GPU (Graphics Processing Unit) is a parallel processor designed to handle thousands of simultaneous computations, making it ideal for training neural networks." },
    { prompt: "What is Kubernetes?", response: "Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications." },
    { prompt: "Explain the attention mechanism.", response: "Attention mechanisms allow models to focus on different parts of the input when generating each part of the output, computing weighted combinations of all input positions." },
    { prompt: "What is MLflow?", response: "MLflow is an open-source platform for managing the ML lifecycle, including experiment tracking, model versioning, and deployment." },
  ];
  const blob = new Blob([lines.map(l => JSON.stringify(l)).join('\n')], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'sample.jsonl';
  a.click();
}

// --- Helpers ---

function statusBadge(status) {
  const pulse = ['running', 'evaluating', 'queued'].includes(status);
  return `<span class="badge-status ${status}"><span class="dot${pulse ? ' pulse' : ''}"></span>${status}</span>`;
}

function shortModel(m) {
  return m.split('/').pop().replace('TinyLlama-', '').replace('-Chat-v1.0', '');
}

function relTime(iso) {
  const d = new Date(iso + 'Z');
  const diff = (Date.now() - d) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
  return d.toLocaleTimeString();
}

function toast(msg, type = 'info') {
  const c = document.getElementById('toasts');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  c.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// Init
pollJobs();
