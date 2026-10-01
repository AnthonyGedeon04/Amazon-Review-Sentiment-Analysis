// Sends the review to /api/predict and draws the label with one bar per class.
const LABELS = ['not satisfied', 'neutral', 'satisfied'];
 
const form = document.getElementById('review-form');
const titleInput = document.getElementById('title');
const textInput = document.getElementById('text');
const counter = document.getElementById('counter');
const submitBtn = document.getElementById('submit');
const resultBox = document.getElementById('result');
const resultLabel = document.getElementById('result-label');
const bars = document.getElementById('bars');
const meta = document.getElementById('result-meta');
const errorBox = document.getElementById('error');
const maxChars = Number(textInput.getAttribute('maxlength'));
 
const slug = (label) => label.replace(/\s+/g, '-');
 
function updateCounter() {
  counter.textContent = `${textInput.value.length.toLocaleString()} / ${maxChars.toLocaleString()}`;
}
 
function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}
 
function setLoading(loading) {
  submitBtn.disabled = loading;
  submitBtn.classList.toggle('loading', loading);
}
 
function render({ label, scores, ms }) {
  resultBox.hidden = false;
  resultBox.dataset.label = slug(label);
  resultLabel.textContent = label;
  resultLabel.className = `result-label tag-${slug(label)}`;
 
  bars.innerHTML = '';
  for (const name of LABELS) {
    const pct = Math.round(scores[name] * 100);
    const row = document.createElement('div');
    row.className = `bar-row bar-${slug(name)}${name === label ? ' is-top' : ''}`;
    row.innerHTML = `
      <span class="bar-name"></span>
      <span class="bar-track"><span class="bar-fill"></span></span>
      <span class="bar-value">${pct}%</span>`;
    row.querySelector('.bar-name').textContent = name;
    bars.appendChild(row);
    // set the width on the next frame so the bar animates from 0
    requestAnimationFrame(() => { row.querySelector('.bar-fill').style.width = `${Math.max(pct, 1)}%`; });
  }
  meta.textContent = `Answered in ${ms} ms`;
}
 
async function analyze() {
  const text = textInput.value.trim();
  if (!text) {
    showError('Write a review first.');
    textInput.focus();
    return;
  }
  showError('');
  setLoading(true);
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, title: titleInput.value.trim() }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `The server answered ${res.status}.`);
    render(data);
  } catch (err) {
    showError(err.message === 'Failed to fetch'
      ? 'Could not reach the server. Is the app still running?'
      : err.message);
  } finally {
    setLoading(false);
  }
}
 
form.addEventListener('submit', (e) => { e.preventDefault(); analyze(); });
textInput.addEventListener('input', updateCounter);
form.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); analyze(); }
});
 
document.getElementById('clear').addEventListener('click', () => {
  titleInput.value = '';
  textInput.value = '';
  updateCounter();
  resultBox.hidden = true;
  showError('');
  textInput.focus();
});
 
document.querySelectorAll('.chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    titleInput.value = chip.dataset.title;
    textInput.value = chip.dataset.text;
    updateCounter();
    analyze();
  });
});
 
updateCounter();