"use strict";

const $ = (sel) => document.querySelector(sel);
const state = { config: null, example: null };

const SOURCE_NAMES = { mmlu: "MMLU", mmlu_pro: "MMLU-Pro", arc: "ARC", gsm8k: "GSM8K" };
const MODE_TEXT = {
  cascade: {
    label: "Escalate when p(wrong) is at least",
    help: "The small model answers. Jev estimates the probability that its answer is wrong. At or above the threshold, the large model answers instead. Lower thresholds escalate more often.",
  },
  preroute: {
    label: "Use the large model when p(small is enough) is below",
    help: "Jev reads only the prompt and estimates whether the small model will get it right. Below the threshold, the prompt goes straight to the large model. In the evaluation this did worse than random routing.",
  },
};

function usd(x) {
  if (x === 0) return "$0";
  if (x < 0.0001) return "$" + x.toFixed(7);
  return "$" + x.toFixed(5);
}
const secs = (s) => s.toFixed(2) + " s";

function finalAnswer(text) {
  const matches = [...(text || "").matchAll(/ANSWER:\s*\**\s*([^\s*]+)/gi)];
  return matches.length ? matches[matches.length - 1][1].replace(/[.)]$/, "") : null;
}

function sameAnswer(a, gold) {
  if (a == null) return false;
  const na = parseFloat(a.replace(/[$,]/g, ""));
  const ng = parseFloat(gold.replace(/[$,]/g, ""));
  if (!isNaN(na) && !isNaN(ng)) return Math.abs(na - ng) < 1e-6;
  return a.trim().toUpperCase().startsWith(gold.trim().toUpperCase());
}

function mode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function syncMode() {
  const m = mode();
  $("#threshold-label").textContent = MODE_TEXT[m].label;
  $("#mode-help").textContent = MODE_TEXT[m].help;
  if (state.config) {
    const t = m === "cascade" ? state.config.cascade_threshold : state.config.preroute_threshold;
    $("#threshold").value = t;
    $("#threshold-out").value = Number(t).toFixed(2);
  }
}

function station({ kind, title, meta, taken, on }) {
  const li = $("#station").content.firstElementChild.cloneNode(true);
  li.classList.add(kind);
  if (taken) li.classList.add("taken");
  if (on) li.classList.add("on");
  if (!on) li.classList.add("skipped");
  li.querySelector("h3").textContent = title;
  li.querySelector(".meta").textContent = meta || "";
  return li;
}

function para(text, cls) {
  const p = document.createElement("p");
  if (cls) p.className = cls;
  p.textContent = text;
  return p;
}

function output(text, open) {
  const d = document.createElement("details");
  if (open) d.open = true;
  const s = document.createElement("summary");
  const ans = finalAnswer(text);
  s.textContent = ans ? `Answered ${ans}. Show full response` : "Show full response";
  const pre = document.createElement("pre");
  pre.className = "output";
  pre.textContent = text || "(empty response)";
  d.append(s, pre);
  return d;
}

// zone: "above" means values at or above the threshold escalate; "below" means values below it do.
function gauge(p, threshold, zone, lowLabel, highLabel) {
  const wrap = document.createElement("div");
  wrap.className = "gauge";
  const track = document.createElement("div");
  track.className = "gauge-track";
  const z = document.createElement("div");
  z.className = "gauge-zone";
  if (zone === "above") { z.style.left = threshold * 100 + "%"; }
  else { z.style.left = "0"; z.style.right = (1 - threshold) * 100 + "%"; z.style.borderRadius = "7px 0 0 7px"; }
  const tick = document.createElement("div");
  tick.className = "gauge-tick";
  tick.style.left = `calc(${threshold * 100}% - 1px)`;
  tick.title = "threshold " + threshold.toFixed(2);
  const dot = document.createElement("div");
  dot.className = "gauge-dot";
  dot.style.left = "0%";
  track.append(z, tick, dot);
  const scale = document.createElement("div");
  scale.className = "gauge-scale";
  scale.innerHTML = "<span></span><span></span>";
  scale.children[0].textContent = lowLabel;
  scale.children[1].textContent = highLabel;
  wrap.append(track, scale);
  requestAnimationFrame(() => requestAnimationFrame(() => { dot.style.left = p * 100 + "%"; }));
  return wrap;
}

function modelMeta(step) {
  return `${step.model}  ${secs(step.seconds)}  ${step.input_tokens + step.output_tokens} tokens  ${usd(step.usd)}`;
}

function renderCascade(r) {
  const trace = [];
  const s = station({ kind: "small", title: "Small model answers", meta: modelMeta(r.small), taken: true, on: true });
  s.querySelector(".content").append(output(r.small.text, !r.check.escalated));
  trace.push(s);

  const c = r.check;
  const j = station({
    kind: "jev", title: "Jev checks the answer", taken: c.escalated, on: c.p_wrong != null,
    meta: c.p_wrong != null ? `${secs(c.seconds)}  ${usd(c.usd)}` : "not called",
  });
  const jc = j.querySelector(".content");
  if (c.p_wrong != null) {
    jc.append(para(`Probability the answer is wrong: ${c.p_wrong.toFixed(2)}. Threshold ${c.threshold.toFixed(2)}.`));
    jc.append(gauge(c.p_wrong, c.threshold, "above", "Looks right", "Looks wrong"));
    jc.append(para(c.escalated ? "Escalated to the large model." : "Kept the small model's answer.", "verdict"));
  } else {
    jc.append(para(`Escalated without a check: ${c.fallback}.`, "verdict"));
  }
  trace.push(j);

  const l = station({
    kind: "large", title: r.large ? "Large model answers" : "Large model not needed",
    meta: r.large ? modelMeta(r.large) : state.config.large.id, on: !!r.large,
  });
  if (r.large) l.querySelector(".content").append(output(r.large.text, true));
  trace.push(l);
  return trace;
}

function renderPreroute(r) {
  const rt = r.route;
  const trace = [];
  const j = station({
    kind: "jev", title: "Jev reads the prompt", taken: true, on: rt.p_small_ok != null,
    meta: rt.p_small_ok != null ? `${secs(rt.seconds)}  ${usd(rt.usd)}` : "failed",
  });
  const jc = j.querySelector(".content");
  if (rt.p_small_ok != null) {
    jc.append(para(`Probability the small model is enough: ${rt.p_small_ok.toFixed(2)}. Threshold ${rt.threshold.toFixed(2)}.`));
    jc.append(gauge(rt.p_small_ok, rt.threshold, "below", "Needs the large model", "Small is enough"));
    jc.append(para(rt.role === "large" ? "Sent to the large model." : "Sent to the small model.", "verdict"));
  } else {
    jc.append(para(`Sent to the large model because Jev was unavailable: ${rt.fallback}.`, "verdict"));
  }
  trace.push(j);

  for (const role of ["small", "large"]) {
    const used = rt.role === role;
    const name = role === "small" ? "Small model" : "Large model";
    const st = station({
      kind: role, title: used ? `${name} answers` : `${name} not used`,
      meta: used ? modelMeta(r.answered) : state.config[role].id, on: used, taken: used && role === "small",
    });
    if (used) st.querySelector(".content").append(output(r.answered.text, true));
    trace.push(st);
  }
  return trace;
}

function renderSummary(r) {
  const box = $("#summary");
  box.innerHTML = "";
  const dl = document.createElement("dl");
  dl.style.display = "contents";
  const add = (label, value, cls) => {
    const d = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    if (cls) dd.className = cls;
    d.append(dt, dd);
    dl.append(d);
  };
  const ans = finalAnswer(r.answer.text);
  add(`Final answer, from the ${r.answer.role} model`, ans ?? "No ANSWER line");
  if (state.example) {
    const ok = sameAnswer(ans, state.example.gold);
    add(`Gold answer is ${state.example.gold}`, ok ? "Correct" : "Wrong", ok ? "grade-ok" : "grade-bad");
  }
  add("Cost at list prices, including Jev", usd(r.total.usd));
  add("Large model alone would cost about", usd(r.always_large_usd));
  add("Total time", secs(r.total.seconds));
  box.append(dl);
  box.hidden = false;
}

async function run(e) {
  e.preventDefault();
  const prompt = $("#prompt").value.trim();
  if (!prompt) return;
  const btn = $("#run");
  const status = $("#status");
  btn.disabled = true;
  btn.textContent = "Running";
  status.className = "status";
  status.textContent = mode() === "cascade" ? "The small model is answering, then Jev checks it." : "Jev is reading the prompt.";
  $("#trace").hidden = true;
  $("#summary").hidden = true;
  try {
    const res = await fetch("api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, mode: mode(), threshold: parseFloat($("#threshold").value) }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(data.detail) ? "The prompt is empty or too long." : data.detail;
      throw new Error(detail || `The server returned ${res.status}.`);
    }
    const trace = $("#trace");
    trace.replaceChildren(...(data.mode === "cascade" ? renderCascade(data) : renderPreroute(data)));
    trace.classList.remove("reveal");
    void trace.offsetWidth;
    trace.classList.add("reveal");
    trace.hidden = false;
    renderSummary(data);
    status.textContent = "";
  } catch (err) {
    status.className = "status error";
    status.textContent = err.message === "Failed to fetch" ? "The server could not be reached. It may be waking up; try again in a few seconds." : err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run";
  }
}

function renderExamples(examples) {
  const box = $("#examples");
  const order = [...new Set(examples.map((ex) => ex.group))];
  for (const g of order) {
    const members = examples.filter((x) => x.group === g);
    const group = document.createElement("div");
    group.className = "example-group " + g;
    const head = document.createElement("p");
    head.className = "example-head";
    const strong = document.createElement("strong");
    strong.textContent = members[0].group_title;
    head.append(strong, " " + members[0].group_hint);
    const chips = document.createElement("div");
    chips.className = "chips";
    for (const ex of members) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = ex.label;
      const src = document.createElement("span");
      src.className = "chip-src";
      src.textContent = SOURCE_NAMES[ex.source] || ex.source;
      b.append(" ", src);
      b.setAttribute("aria-pressed", "false");
      b.title = ex.prompt.slice(0, 160);
      b.addEventListener("click", () => {
        $("#prompt").value = ex.prompt;
        state.example = ex;
        box.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      });
      chips.append(b);
    }
    group.append(head, chips);
    box.append(group);
  }
}

function renderFooter(c) {
  const f = $("#foot");
  const lines = [
    `Small model: ${c.small.id}. Large model: ${c.large.id}. Both are open-weight models served on the free tier of ${c.small.provider === "groq" ? "Groq" : c.small.provider}; costs shown use the provider's paid list prices ($${c.small.usd_per_m_input} and $${c.large.usd_per_m_input} per million input tokens).`,
    `Checker: ${c.jev_model}. Each visitor can send ${c.rate_per_minute} requests a minute.`,
  ];
  if (!c.jev_ready) lines.push("Jev is not configured on this server, so every request falls back to the large model.");
  if (!c.llm_ready) lines.push(`The models are not configured on this server: ${c.llm_problem}.`);
  f.replaceChildren(...lines.map((t) => para(t)));
}

async function init() {
  document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", syncMode));
  $("#threshold").addEventListener("input", (e) => { $("#threshold-out").value = Number(e.target.value).toFixed(2); });
  $("#prompt").addEventListener("input", () => {
    if (state.example && $("#prompt").value !== state.example.prompt) {
      state.example = null;
      $("#examples").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", "false"));
    }
  });
  $("#ask").addEventListener("submit", run);
  syncMode();
  try {
    const [config, examples] = await Promise.all([
      fetch("api/config").then((r) => r.json()),
      fetch("api/examples").then((r) => r.json()),
    ]);
    state.config = config;
    $("#prompt").maxLength = config.max_prompt_chars;
    syncMode();
    renderExamples(examples);
    renderFooter(config);
  } catch {
    $("#status").className = "status error";
    $("#status").textContent = "The server could not be reached. It may be waking up; reload in a few seconds.";
  }
}

init();
