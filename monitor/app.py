"""Live training monitor: parses a plugin's rank-0 log lines and plots them.

Every ``key=value`` on a ``step=N`` line becomes a series, and every
``group[k=v ...]`` becomes series ``group/k``. For the reconstructor +
forecaster plugin (fieldspace_RF_plugin.py) that gives one panel per model,
each showing the per-level loss the model is trained on and its reference:

* Reconstructor: ``recon/<level>``, with ``state_ms/<level>`` (dotted, the
  loss of outputting zero).
* Forecaster: ``latent_mse/<level>``, the latent forecast MSE per latent level,
  mean over its batch (newest + replayed windows), with
  ``latent_persist/<level>`` (dotted, the loss of persistence). Older runs
  logged the field increment loss ``incre`` with ``persist_mse`` instead.
* Forecast error: ``fc_rmse`` vs ``persistence`` in the variable's units.

The "÷ reference" switch of a panel divides each curve by its reference (both
smoothed first), so incre then reads 1 - skill against persistence. Each panel
has checkboxes to choose its curves and a log/linear switch; a sample-step
window (from/to) limits all panels to part of the run.

    bash scripts/monitor.sh LOG_FILE PORT
"""
import argparse
import os
import re

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
_log_file = None

_TIME = re.compile(r"^\s*0:\s+Time step:\s+\d+,?\s+model time:?\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_STEP = re.compile(r"^\s*0:.*\bstep=(\d+)\s")
_GROUP = re.compile(r"(\w+)\[([^\]]*)\]")
_VALUE = re.compile(r"(\w+)=(-?[\d.]+(?:[eE][-+]?\d+)?)")


def parse_log():
    """One point per rank-0 training line: step, the last ICON model time
    printed before it, and all its numeric values."""
    if not os.path.exists(_log_file):
        return None, f"Log file not found: {_log_file}"
    points, model_time = [], None
    with open(_log_file, errors="ignore") as f:
        for line in f:
            if m := _TIME.match(line):
                model_time = m.group(1)
            elif m := _STEP.match(line):
                values = {
                    f"{group}/{k}": float(v)
                    for group, body in _GROUP.findall(line)
                    for k, v in _VALUE.findall(body)
                }
                values.update((k, float(v)) for k, v in _VALUE.findall(_GROUP.sub("", line)) if k != "step")
                points.append({"step": int(m.group(1)), "time": model_time, "values": values})
    return {"points": points}, None


HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MEssE monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: monospace; background: #fff; color: #333; padding: 20px; }
  #header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 6px; }
  #logo { font-size: 1.6em; font-weight: bold; color: #6ab0d4; letter-spacing: 1px; }
  h3 { font-size: 1em; color: #666; }
  #stats { font-size: 0.82em; color: #999; margin-bottom: 12px; }
  #controls { font-size: 0.85em; margin-bottom: 14px; }
  #controls input { font: inherit; width: 7em; }
  #controls button { font: inherit; padding: 0 6px; cursor: pointer; }
  .panel { max-width: 1100px; margin-bottom: 26px; }
  .panel h4 { font-size: 0.95em; margin-bottom: 2px; }
  .panel p { font-size: 0.78em; color: #888; margin-bottom: 6px; }
  .wrap { height: 340px; }
  .bar { font-size: 0.8em; margin-bottom: 6px; display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: center; }
  .bar label { cursor: pointer; white-space: nowrap; }
  .bar .swatch { display: inline-block; width: 22px; height: 0; border-top: 3px solid; vertical-align: middle; margin: 0 3px; }
  .bar button { font: inherit; font-size: 0.95em; padding: 0 6px; cursor: pointer; }
  #err { color: #c33; font-size: 0.85em; margin-top: 10px; }
</style>
</head>
<body>
<div id="header"><span id="logo">MEssE</span><h3>{{ job_id }}</h3></div>
<div id="stats">Loading&hellip;</div>
<div id="controls">smoothing (running mean over steps):
  <select id="smooth"><option>1</option><option selected>10</option><option>50</option></select>
  &nbsp;&nbsp;sample steps from <input id="step-min" type="number" min="0" placeholder="first">
  to <input id="step-max" type="number" min="0" placeholder="last">
  <button id="step-all">whole run</button></div>

<div class="panel"><h4>Reconstructor loss</h4>
  <p>Per level: the trained reconstruction MSE (solid); dotted = the MSE of outputting zero.
     "÷ reference": 0 = perfect, 1 = outputs nothing.</p>
  <div class="bar" id="recon-bar"></div>
  <div class="wrap"><canvas id="recon"></canvas></div></div>
<div class="panel"><h4>Forecaster loss</h4>
  <p>Per latent level: the trained latent forecast MSE, mean over the batch of newest + replayed windows (solid);
     dotted = the persistence MSE of the same batch (older runs: field increment MSE).
     "÷ reference": 1 = persistence, 0 = perfect.</p>
  <div class="bar" id="forecast-bar"></div>
  <div class="wrap"><canvas id="forecast"></canvas></div></div>
<div class="panel"><h4>Forecast error of var_predict</h4>
  <p>RMSE over one lead in the variable's units, model vs persistence.</p>
  <div class="bar" id="rmse-bar"></div>
  <div class="wrap"><canvas id="rmse"></canvas></div></div>
<div id="err"></div>

<script>
const LEVEL_COLORS = ["#000000", "#1e88e5", "#fb8c00", "#43a047", "#8e24aa", "#e53935"];
// ref: the reference curve a curve is divided by under "÷ reference".
const PANELS = {
  recon:    { match: n => /^(recon|state_ms)[/]/.test(n), log: true,
              ref: n => n.startsWith("recon/") ? "state_ms/" + n.split("/")[1] : null },
  forecast: { match: n => /^(latent_mse|latent_persist|incre|persist_mse)[/]/.test(n), log: false,
              ref: n => { const [group, key] = n.split("/");
                          const r = { latent_mse: "latent_persist/", incre: "persist_mse/" }[group];
                          return r ? r + key : null; } },
  rmse:     { match: n => n === "fc_rmse" || n === "persistence", log: false },
};
const REFERENCE = /^(state_ms|persist_mse|latent_persist)[/]/;
const charts = {};
let points = [];
let countdown = 5;

// Curves the user unchecked and per-panel log scales, kept across reloads.
function stored(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch (e) { return fallback; }
}
function store(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) {}
}
const hidden = new Set(stored("messe-hidden", []));
const logScale = stored("messe-log", Object.fromEntries(Object.entries(PANELS).map(([id, p]) => [id, p.log])));
const relative = stored("messe-relative", {});

function smooth(ys, n) {
  return ys.map((_, i) => {
    const w = ys.slice(Math.max(0, i - n + 1), i + 1).filter(v => v !== null);
    return w.length ? w.reduce((a, b) => a + b, 0) / w.length : null;
  });
}

function style(name, names) {
  const key = name.split("/").pop();
  const keys = [...new Set(names.map(n => n.split("/").pop()))];
  const fixed = { fc_rmse: "#1e88e5", persistence: "#e53935" };
  return {
    borderColor: fixed[name] || LEVEL_COLORS[keys.indexOf(key) % LEVEL_COLORS.length],
    borderDash: REFERENCE.test(name) ? [2, 3] : [],
    borderWidth: REFERENCE.test(name) ? 1.5 : 2,
  };
}

function redraw() { Object.keys(PANELS).forEach(draw); }

// Sample-step window; an empty box means no limit. Kept across reloads.
const stepBox = { min: document.getElementById("step-min"), max: document.getElementById("step-max") };
const savedWindow = stored("messe-window", {});
Object.entries(stepBox).forEach(([k, box]) => {
  box.value = savedWindow[k] ?? "";
  box.onchange = () => { store("messe-window", { min: stepBox.min.value, max: stepBox.max.value }); redraw(); };
});
document.getElementById("step-all").onclick = () => {
  stepBox.min.value = stepBox.max.value = "";
  store("messe-window", {});
  redraw();
};
function inWindow(step) {
  const lo = stepBox.min.value === "" ? -Infinity : Number(stepBox.min.value);
  const hi = stepBox.max.value === "" ? Infinity : Number(stepBox.max.value);
  return step >= lo && step <= hi;
}

function setHidden(names, hide) {
  names.forEach(n => hide ? hidden.add(n) : hidden.delete(n));
  store("messe-hidden", [...hidden]);
  redraw();
}

// One checkbox per curve, plus all/none and the y-scale switch.
function drawBar(id, names) {
  const bar = document.getElementById(id + "-bar");
  if (bar.dataset.names === names.join()) {
    bar.querySelectorAll("input[data-name]").forEach(c => c.checked = !hidden.has(c.dataset.name));
    return;
  }
  bar.dataset.names = names.join();
  bar.innerHTML = "";
  names.forEach(name => {
    const s = style(name, names);
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" data-name="${name}"><span class="swatch" style="border-top-color:${s.borderColor};border-top-style:${s.borderDash.length ? "dashed" : "solid"}"></span>${name}`;
    const box = label.querySelector("input");
    box.checked = !hidden.has(name);
    box.onchange = () => setHidden([name], !box.checked);
    bar.appendChild(label);
  });
  const all = document.createElement("button"); all.textContent = "all"; all.onclick = () => setHidden(names, false);
  const none = document.createElement("button"); none.textContent = "none"; none.onclick = () => setHidden(names, true);
  const log = document.createElement("label");
  log.innerHTML = `<input type="checkbox"> log y`;
  log.querySelector("input").checked = logScale[id];
  log.querySelector("input").onchange = e => { logScale[id] = e.target.checked; store("messe-log", logScale); redraw(); };
  bar.append(all, none, log);
  if (PANELS[id].ref && names.some(n => REFERENCE.test(n))) {
    const rel = document.createElement("label");
    rel.innerHTML = `<input type="checkbox"> \u00f7 reference`;
    rel.querySelector("input").checked = !!relative[id];
    rel.querySelector("input").onchange = e => { relative[id] = e.target.checked; store("messe-relative", relative); redraw(); };
    bar.append(rel);
  }
}

function draw(id) {
  const panel = PANELS[id];
  const names = [...new Set(points.flatMap(p => Object.keys(p.values)))].filter(panel.match).sort();
  drawBar(id, names);
  const n = parseInt(document.getElementById("smooth").value);
  const hasRef = names.some(nm => REFERENCE.test(nm));
  const rel = panel.ref && relative[id] && hasRef;
  const series = name => smooth(points.map(p => p.values[name] ?? null), n);
  const datasets = names.filter(name => !(rel && REFERENCE.test(name))).map(name => {
    let ys = series(name);
    const ref = rel && panel.ref(name);
    if (ref) {
      const r = series(ref);
      ys = ys.map((y, i) => (y === null || !r[i]) ? null : y / r[i]);
    }
    return {
      label: name,
      data: ys.map((y, i) => ({ x: points[i].step, y })).filter(d => inWindow(d.x)),
      hidden: hidden.has(name),
      pointRadius: 0, tension: 0, ...style(name, names),
    };
  });
  // The 1-line is the reference once divided.
  if (names.length && rel) {
    datasets.push({ label: "reference (1)", data: points.filter(p => inWindow(p.step)).map(p => ({ x: p.step, y: 1 })),
                    borderColor: "#bbb", borderWidth: 1, pointRadius: 0 });
  }
  const yType = logScale[id] ? "logarithmic" : "linear";
  if (charts[id]) {
    charts[id].data.datasets = datasets;
    charts[id].options.scales.y.type = yType;
    charts[id].update("none");
    return;
  }
  charts[id] = new Chart(document.getElementById(id), {
    type: "line",
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, parsing: true,
      interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: {
        legend: { display: false },  // the checkboxes above are the legend
        tooltip: { callbacks: {
          title: items => {
            const p = points.find(q => q.step === items[0].raw.x);
            return "step " + items[0].raw.x + (p && p.time ? "  ·  " + p.time : "");
          },
          label: item => item.dataset.label + ": " + (item.raw.y === null ? "-" : item.raw.y.toPrecision(4)),
        } },
      },
      scales: {
        x: { type: "linear", title: { display: true, text: "sample step" }, grid: { color: "#eee" } },
        y: { type: yType, grid: { color: "#eee" } },
      },
    },
  });
}

async function load() {
  const resp = await fetch("/data").catch(() => null);
  if (!resp) return;
  const data = await resp.json();
  document.getElementById("err").textContent = data.error || "";
  points = data.points || [];
  const last = points[points.length - 1];
  document.getElementById("stats").textContent = last
    ? points.length + " steps | last step " + last.step + " | model time " + (last.time || "?")
    : "No training lines yet.";
  redraw();
}

document.getElementById("smooth").onchange = redraw;
load();
setInterval(() => { if (--countdown <= 0) { countdown = 5; load(); } }, 1000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML, job_id=os.path.basename(_log_file))


@app.route("/data")
def data():
    result, error = parse_log()
    return jsonify(result or {"error": error})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICON training loss monitor")
    parser.add_argument("log_file", help="Path to the ICON log file")
    parser.add_argument("port", type=int, help="Port to serve on")
    args = parser.parse_args()
    _log_file = os.path.abspath(args.log_file)
    print(f"Monitoring log: {_log_file}")
    print(f"Open: http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
