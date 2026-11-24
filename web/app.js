"use strict";

const $ = (id) => document.getElementById(id);
const form = $("ask-form");
const questionInput = $("question");
const askBtn = $("ask-btn");
const statusEl = $("status");
const answerPanel = $("answer-panel");
const sourcesPanel = $("sources-panel");
const modeSelect = $("mode");
const kSelect = $("k");
const docsSelect = $("docs");

let currentTraceId = null;

/* ---------------------------------------------------------------- helpers */

function setStatus(message, isError) {
  statusEl.textContent = message || "";
  statusEl.hidden = !message;
  statusEl.classList.toggle("error", Boolean(isError));
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/** Turn "[1]" markers into links that scroll to the matching source card. */
function renderAnswer(text) {
  return escapeHtml(text).replace(/\[(\d+)\]/g, (match, n) =>
    `<a class="cite" href="#source-${n}" title="Jump to source ${n}">${n}</a>`
  );
}

function badge(el, text, kind) {
  el.textContent = text;
  el.className = "badge" + (kind ? " " + kind : "");
}

/* ------------------------------------------------------------------ setup */

async function loadExamples() {
  try {
    const data = await (await fetch("/api/examples")).json();
    $("examples").innerHTML = "";
    for (const example of data.examples) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = example;
      chip.addEventListener("click", () => {
        questionInput.value = example;
        form.requestSubmit();
      });
      $("examples").appendChild(chip);
    }
  } catch (_) { /* examples are a convenience, not a requirement */ }
}

async function loadDocuments() {
  try {
    const docs = await (await fetch("/api/documents")).json();
    for (const doc of docs) {
      const option = document.createElement("option");
      option.value = doc.doc_id;
      option.textContent = `${doc.title} (${doc.pages} pages)`;
      docsSelect.appendChild(option);
    }
  } catch (_) { /* filter stays on "all" */ }
}

async function loadMeta() {
  try {
    const stats = await (await fetch("/api/stats")).json();
    $("disclaimer").textContent = stats.disclaimer;
  } catch (_) { /* non-fatal */ }
}

function metric(value, label) {
  return `<div class="metric"><b>${value}</b><span>${label}</span></div>`;
}

async function loadEvaluation() {
  try {
    const report = await (await fetch("/api/evaluation")).json();
    const p = report.pipeline;
    const pct = (x) => `${(x * 100).toFixed(1)}%`;

    let html = '<div class="metric-grid">';
    html += metric(pct(p.refusal_rate_out_of_scope), "Out-of-scope questions correctly refused");
    html += metric(pct(p.mean_groundedness), "Answer sentences carrying a citation");
    html += metric(pct(p.cited_correct_page_rate), "Answers citing a correct page");
    html += metric(`${p.median_latency_ms.toFixed(0)} ms`, "Median time to answer");
    html += "</div>";

    const modes = { lexical: "Keyword only", dense: "Semantic only", hybrid: "Hybrid (shipped)" };
    const best = Object.keys(modes).reduce((a, b) =>
      report.retrieval[a].mrr_at_10 >= report.retrieval[b].mrr_at_10 ? a : b);
    html += '<div class="metrics-scroll"><table class="metrics"><thead><tr>' +
      "<th>Retrieval mode</th><th>Recall@1</th><th>Recall@5</th><th>Recall@10</th><th>MRR@10</th>" +
      "</tr></thead><tbody>";
    for (const [key, label] of Object.entries(modes)) {
      const m = report.retrieval[key];
      html += `<tr class="${key === best ? "best" : ""}"><td>${label}</td>` +
        `<td>${pct(m.recall_at["1"])}</td><td>${pct(m.recall_at["5"])}</td>` +
        `<td>${pct(m.recall_at["10"])}</td><td>${m.mrr_at_10.toFixed(3)}</td></tr>`;
    }
    html += "</tbody></table></div>";

    const n = p.num_questions, ins = p.num_in_scope, out = p.num_out_of_scope;
    $("eval-hint").textContent =
      `Measured on ${n} held-out questions (${ins} answerable, ${out} deliberately not), ` +
      `never used for tuning. Generated ${report.generated_at.slice(0, 10)}.`;
    $("evaluation").innerHTML = html;
  } catch (_) {
    $("evaluation").innerHTML = '<p class="section-hint">Evaluation report not available.</p>';
  }
}

/* ------------------------------------------------------------- rendering  */

function renderTrace(data) {
  const steps = (data.steps || [])
    .filter((s) => s.node !== "graph")
    .map((s) => {
      const detail = s.detail && s.detail.detail ? `:${s.detail.detail}` : "";
      return `<span class="step">${escapeHtml(s.node + detail)}</span>`;
    })
    .join('<span class="arrow">→</span>');

  let meta = `trace <code>${escapeHtml(data.trace_id)}</code> · ${data.total_ms} ms total`;
  if (data.attempts > 1) meta += ` · ${data.attempts} retrieval attempts`;
  if (data.llm_model) meta += ` · ${escapeHtml(data.llm_model)} (${data.input_tokens}→${data.output_tokens} tokens)`;
  $("trace").innerHTML = `<div class="trace-steps">${steps}</div><div class="trace-meta">${meta}</div>`;
}

function renderSources(sources) {
  const list = $("sources");
  list.innerHTML = "";
  for (const s of sources) {
    const li = document.createElement("li");
    li.className = "source";
    li.id = `source-${s.number}`;
    const arms = [];
    if (s.dense_rank) arms.push(`<span class="tag">semantic #${s.dense_rank}</span>`);
    if (s.lexical_rank) arms.push(`<span class="tag">keyword #${s.lexical_rank}</span>`);
    if (s.matched_terms && s.matched_terms.length) {
      arms.push(`<span class="tag">matched: ${escapeHtml(s.matched_terms.slice(0, 5).join(", "))}</span>`);
    }
    li.innerHTML =
      `<div class="source-head">
         <span class="source-num">${s.number}</span>
         <span class="source-title">${escapeHtml(s.doc_title)} — page ${s.page}</span>
       </div>
       ${s.section ? `<div class="source-section">${escapeHtml(s.section)}</div>` : ""}
       <p class="source-text">${escapeHtml(s.text)}</p>
       <div class="source-foot">
         <a href="${s.url}" target="_blank" rel="noopener">Open page ${s.page} of the official PDF ↗</a>
         ${arms.join("")}
       </div>`;
    list.appendChild(li);
  }
  sourcesPanel.hidden = sources.length === 0;
}

function renderResult(data) {
  currentTraceId = data.trace_id;
  $("fb-thanks").hidden = true;

  const strategyLabel = {
    extractive: "quoted from sources",
    generated: "generated + citations verified",
    refused: "refused",
  }[data.strategy] || data.strategy;
  badge($("badge-strategy"), strategyLabel, data.strategy === "refused" ? "warn" : "accent");

  const confKind = { high: "good", medium: "warn", low: "bad" }[data.confidence] || "";
  badge($("badge-confidence"), `${data.confidence} confidence`, confKind);
  badge($("badge-grounded"), `${(data.groundedness * 100).toFixed(0)}% of sentences cited`,
        data.groundedness >= 0.999 ? "good" : "warn");

  $("answer").innerHTML = renderAnswer(data.answer);
  $("notes").innerHTML = (data.notes || []).map((n) => `<li>${escapeHtml(n)}</li>`).join("");
  answerPanel.hidden = false;
  renderSources(data.sources || []);

  fetch(`/api/traces/${data.trace_id}`)
    .then((r) => (r.ok ? r.json() : null))
    .then((t) => { if (t) renderTrace(t); })
    .catch(() => {});
}

/* --------------------------------------------------------------- actions  */

async function ask(question) {
  const params = new URLSearchParams({ q: question, k: kSelect.value, mode: modeSelect.value });
  if (docsSelect.value) params.set("docs", docsSelect.value);

  askBtn.disabled = true;
  setStatus("Searching the Medicare publications…");
  answerPanel.hidden = true;
  sourcesPanel.hidden = true;

  try {
    const response = await fetch(`/api/ask?${params}`);
    if (!response.ok) {
      let detail = `Request failed (${response.status}).`;
      try {
        const body = await response.json();
        if (response.status === 422) detail = "That question could not be processed. Please rephrase it.";
        else if (body && body.detail) detail = typeof body.detail === "string" ? body.detail : detail;
      } catch (_) { /* keep the generic message */ }
      setStatus(detail, true);
      return;
    }
    const data = await response.json();
    setStatus("");
    renderResult(data);
    const url = new URL(window.location);
    url.searchParams.set("q", question);
    url.searchParams.set("mode", modeSelect.value);
    if (docsSelect.value) url.searchParams.set("docs", docsSelect.value); else url.searchParams.delete("docs");
    window.history.replaceState({}, "", url);
  } catch (error) {
    setStatus("Could not reach the service. It may still be waking up — try again in a moment.", true);
  } finally {
    askBtn.disabled = false;
  }
}

async function sendFeedback(helpful) {
  if (!currentTraceId) return;
  try {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trace_id: currentTraceId, helpful, comment: "" }),
    });
    $("fb-thanks").hidden = false;
  } catch (_) { /* feedback is best-effort */ }
}

function restoreFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const q = params.get("q");
  const mode = params.get("mode");
  const docs = params.get("docs");
  if (mode && Array.from(modeSelect.options).some((o) => o.value === mode)) modeSelect.value = mode;
  if (docs && docsSelect.querySelector(`option[value="${CSS.escape(docs)}"]`)) docsSelect.value = docs;
  if (q) { questionInput.value = q; ask(q); return true; }
  return false;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question) ask(question);
});
modeSelect.addEventListener("change", () => {
  const q = questionInput.value.trim();
  if (q && !answerPanel.hidden) ask(q);
});
docsSelect.addEventListener("change", () => {
  const q = questionInput.value.trim();
  if (q && !answerPanel.hidden) ask(q);
});
$("fb-yes").addEventListener("click", () => sendFeedback(true));
$("fb-no").addEventListener("click", () => sendFeedback(false));

const themeBtn = $("theme-btn");
themeBtn.addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("theme", next); } catch (_) {}
});
try {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
} catch (_) {}

loadExamples();
loadEvaluation();
loadMeta();
loadDocuments().then(() => {
  if (!restoreFromUrl()) questionInput.focus();
});
