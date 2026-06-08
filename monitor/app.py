import re
import argparse
from datetime import datetime
from flask import Flask, jsonify, render_template_string

LOG_DIR = "/work/mh0033/m300883/Project_week_global/MEssE/build_atm/messe_env/build_dir/icon-model/run"
LOG_NAME = "LOG.exp.atm_tracer_Hadley_comin_portability.run.{job_id}.o"

app = Flask(__name__)
_job_id = None


def parse_log():
    log_path = f"{LOG_DIR}/{LOG_NAME.format(job_id=_job_id)}"
    step_times = {}
    losses = []

    try:
        with open(log_path) as f:
            for line in f:
                m = re.match(
                    r"0:\s+Time step:\s+(\d+) model time (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
                    line,
                )
                if m:
                    step_times[int(m.group(1))] = m.group(2)
                    continue
                m = re.search(r"\[rank=0\] step=(\d+) loss=([\d.]+)", line)
                if m:
                    losses.append((int(m.group(1)), float(m.group(2))))
    except FileNotFoundError:
        return None, f"Log file not found: {log_path}"

    points = []
    for step, loss in losses:
        time_str = step_times.get(step)
        ts = None
        if time_str:
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            ts = int(dt.timestamp() * 1000)
        points.append({"step": step, "time": time_str, "loss": loss, "ts": ts})

    return {"points": points}, None


HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Loss — Job {{ job_id }}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: monospace; background: #fff; color: #333; padding: 20px; }
  #header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 6px; }
  #logo { font-size: 1.6em; font-weight: bold; color: #6ab0d4; letter-spacing: 1px; }
  h3  { font-size: 1em; color: #666; }
  #stats { font-size: 0.82em; color: #999; margin-bottom: 18px; }
  #wrap  { width: 100%; max-width: 1100px; height: 480px; }
  #err   { color: #c33; font-size: 0.85em; margin-top: 10px; }
</style>
</head>
<body>
<div id="header"><span id="logo">MEssE</span><h3>Training Loss &mdash; Job {{ job_id }}</h3></div>
<div id="stats">Loading&hellip;</div>
<div id="wrap"><canvas id="chart"></canvas></div>
<div id="err"></div>
<script>
let chart = null;
let allPts = [];
let cd = 2;

function iso(ms) {
  return new Date(ms).toISOString().replace("T"," ").slice(0,19);
}
function shortLabel(ms) {
  const s = new Date(ms).toISOString();
  return s.slice(5,10) + " " + s.slice(11,16);
}

async function load() {
  let resp;
  try { resp = await fetch("/data"); } catch(e) { return; }
  if (!resp.ok) {
    const j = await resp.json().catch(() => ({}));
    document.getElementById("err").textContent = j.error || "fetch error";
    return;
  }
  document.getElementById("err").textContent = "";
  const data = await resp.json();
  allPts = data.points || [];
  render();
}

function render() {
  const pts = allPts;
  if (!pts.length) { document.getElementById("stats").textContent = "No data yet."; return; }

  const last = pts[pts.length - 1];
  const xKey = pts.some(p => p.ts) ? "ts" : "step";
  const xs = pts.map(p => xKey === "ts" ? p.ts : p.step);
  const ys = pts.map(p => p.loss);

  document.getElementById("stats").textContent =
    pts.length + " points\u2002|\u2002last loss: " + last.loss.toExponential(4) +
    "\u2002|\u2002model time: " + (last.time || "step " + last.step) +
    "\u2002|\u2002refresh in " + cd + "s";

  if (!chart) {
    const ctx = document.getElementById("chart").getContext("2d");
    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: xs,
        datasets: [{
          data: ys,
          borderColor: "#000",
          borderWidth: 3,
          pointRadius: pts.length > 200 ? 0 : 2.5,
          pointHoverRadius: 5,
          fill: false,
          tension: 0,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: items => xKey === "ts" ? iso(items[0].label) : "step " + items[0].label,
              label: item => "loss: " + item.raw.toExponential(6) + "  (step " + pts[item.dataIndex].step + ")"
            }
          }
        },
        scales: {
          x: {
            type: "linear",
            ticks: {
              color: "#999", maxTicksLimit: 8,
              callback: v => xKey === "ts" ? shortLabel(v) : "step " + v
            },
            grid: { color: "#eee" }
          },
          y: {
            type: "logarithmic",
            ticks: { color: "#999" },
            grid: { color: "#eee" }
          }
        }
      }
    });
  } else {
    chart.data.labels = xs;
    chart.data.datasets[0].data = ys;
    chart.data.datasets[0].pointRadius = pts.length > 200 ? 0 : 2.5;
    chart.update("none");
  }
}

function tick() {
  cd--;
  if (cd <= 0) { cd = 2; load(); }
  else {
    const el = document.getElementById("stats");
    el.textContent = el.textContent.replace(/refresh in \\d+s/, "refresh in " + cd + "s");
  }
}

load();
setInterval(tick, 1000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML, job_id=_job_id)


@app.route("/data")
def data():
    result, error = parse_log()
    if error:
        return jsonify({"error": error}), 404
    return jsonify(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICON training loss monitor")
    parser.add_argument("job_id", help="SLURM job ID")
    parser.add_argument("port", type=int, help="Port to serve on")
    args = parser.parse_args()
    _job_id = args.job_id
    print(f"Monitoring job {args.job_id}")
    print(f"Open: http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
