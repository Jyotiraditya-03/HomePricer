const API = "http://localhost:8000";

// ── Theme ────────────────────────────────────────────────────────────────────
function toggleTheme() {
  const root = document.documentElement;
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  document.querySelector(".theme-toggle").textContent =
    root.dataset.theme === "dark" ? "☀️" : "🌙";
  redrawCharts();
}

function getTheme() { return document.documentElement.dataset.theme || "dark"; }
function paperBg()  { return getTheme() === "dark" ? "#1a1d27" : "#ffffff"; }
function plotBg()   { return getTheme() === "dark" ? "#1a1d27" : "#ffffff"; }
function fontColor(){ return getTheme() === "dark" ? "#e8eaf0" : "#1a1d27"; }

// ── Slider ↔ Number sync ─────────────────────────────────────────────────────
function syncNum(id) {
  document.getElementById(id + "-num").value = document.getElementById(id).value;
}
function syncSlider(id) {
  document.getElementById(id).value = document.getElementById(id + "-num").value;
}

// ── Steppers ─────────────────────────────────────────────────────────────────
const STEPPER_LIMITS = { bedrooms:[1,10], bathrooms:[1,6], stories:[1,4], parking:[0,5] };
function step(id, delta) {
  const inp = document.getElementById(id);
  const [min, max] = STEPPER_LIMITS[id];
  const v = Math.min(max, Math.max(min, parseInt(inp.value) + delta));
  inp.value = v;
  document.getElementById(id + "-val").textContent = v;
}

// ── Toggle group ─────────────────────────────────────────────────────────────
function setTog(groupId, btn) {
  document.querySelectorAll(`#${groupId} .tog`).forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
}

// ── Bool tiles ───────────────────────────────────────────────────────────────
function toggleBool(el) {
  const isOn = el.dataset.val === "yes";
  el.dataset.val = isOn ? "no" : "yes";
  el.classList.toggle("on", !isOn);
  el.querySelector(".bool-dot").classList.toggle("active", !isOn);
}

// Initialise bool tiles from their data-val
document.querySelectorAll(".bool-item").forEach(el => {
  if (el.dataset.val === "yes") {
    el.classList.add("on");
    el.querySelector(".bool-dot").classList.add("active");
  }
});

// ── Build payload ─────────────────────────────────────────────────────────────
function getPayload() {
  return {
    area:             parseFloat(document.getElementById("area").value),
    bedrooms:         parseInt(document.getElementById("bedrooms").value),
    bathrooms:        parseInt(document.getElementById("bathrooms").value),
    stories:          parseInt(document.getElementById("stories").value),
    parking:          parseInt(document.getElementById("parking").value),
    furnishingstatus: document.querySelector("#furnishing .tog.active").dataset.val,
    mainroad:         document.querySelector('[data-field="mainroad"]').dataset.val,
    guestroom:        document.querySelector('[data-field="guestroom"]').dataset.val,
    basement:         document.querySelector('[data-field="basement"]').dataset.val,
    hotwaterheating:  document.querySelector('[data-field="hotwaterheating"]').dataset.val,
    airconditioning:  document.querySelector('[data-field="airconditioning"]').dataset.val,
    prefarea:         document.querySelector('[data-field="prefarea"]').dataset.val,
  };
}

// ── Format currency ──────────────────────────────────────────────────────────
function fmt(n) {
  if (n >= 1e7) return "₹" + (n/1e7).toFixed(2) + "Cr";
  if (n >= 1e5) return "₹" + (n/1e5).toFixed(1) + "L";
  return "₹" + n.toLocaleString("en-IN");
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
function showTab(name, btn) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.add("hidden"));
  btn.classList.add("active");
  document.getElementById("tab-" + name).classList.remove("hidden");
}

// ── Last data store for redraw on theme change ───────────────────────────────
let lastShap = null, lastLime = null, lastImportance = null;

function redrawCharts() {
  if (lastShap)       drawShap(lastShap);
  if (lastLime)       drawLime(lastLime);
  if (lastImportance) drawImportance(lastImportance);
}

// ── Main predict ─────────────────────────────────────────────────────────────
async function predict() {
  const btn  = document.getElementById("predict-btn");
  const text = btn.querySelector(".btn-text");
  const spin = btn.querySelector(".btn-spinner");

  btn.disabled = true;
  text.hidden  = true;
  spin.hidden  = false;

  const payload = getPayload();

  try {
    // 1. Predict
    const predRes = await fetch(`${API}/predict`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload),
    });
    if (!predRes.ok) throw new Error(await predRes.text());
    const pred = await predRes.json();

    // Show price
    document.getElementById("placeholder").classList.add("hidden");
    document.getElementById("tabs").hidden = false;
    document.getElementById("price-display").textContent = fmt(pred.price);
    document.getElementById("price-range").textContent =
      `Range: ${fmt(pred.low)} – ${fmt(pred.high)}`;
    document.getElementById("price-bar-wrap").hidden = false;

    // Update header metric
    if (pred.metrics) {
      document.getElementById("header-r2").textContent =
        `R² ${pred.metrics.r2}  |  CV ${pred.metrics.cv_r2}`;
    }

    // Confidence bar: map price within low–high
    const pct = ((pred.price - pred.low) / (pred.high - pred.low)) * 100;
    document.getElementById("price-bar").style.width  = pct + "%";
    document.getElementById("price-marker").style.left = pct + "%";
    document.getElementById("bar-low").textContent  = fmt(pred.low);
    document.getElementById("bar-high").textContent = fmt(pred.high);

    // 2. SHAP
    const shapRes = await fetch(`${API}/explain/shap`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload),
    });
    if (shapRes.ok) {
      const shap = await shapRes.json();
      lastShap = shap;
      drawShap(shap);
    }

    // 3. LIME
    const limeRes = await fetch(`${API}/explain/lime`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload),
    });
    if (limeRes.ok) {
      const lime = await limeRes.json();
      lastLime = lime;
      drawLime(lime);
    }

    // 4. Feature Importance
    const fiRes = await fetch(`${API}/features`);
    if (fiRes.ok) {
      const fi = await fiRes.json();
      lastImportance = fi;
      drawImportance(fi);
    }

  } catch (err) {
    alert("Error: " + err.message + "\n\nMake sure the HomePricer API server is running on port 8000.");
  } finally {
    btn.disabled = false;
    text.hidden  = false;
    spin.hidden  = true;
  }
}

// ── SHAP Waterfall Chart ─────────────────────────────────────────────────────
function drawShap(data) {
  const items = data.contributions.slice(0, 10);
  const labels   = items.map(d => `${d.feature} = ${d.input}`);
  const values   = items.map(d => d.value);
  const colors   = values.map(v => v >= 0 ? "#10b981" : "#ef4444");

  Plotly.newPlot("shap-chart", [{
    type: "bar",
    orientation: "h",
    x: values,
    y: labels,
    marker: { color: colors },
    text: values.map(v => (v >= 0 ? "+" : "") + (v/1000).toFixed(0) + "K"),
    textposition: "outside",
    hovertemplate: "%{y}<br>SHAP: %{x:,.0f}<extra></extra>",
  }], {
    paper_bgcolor: paperBg(),
    plot_bgcolor:  plotBg(),
    font: { family: "Inter", color: fontColor(), size: 12 },
    xaxis: { title: "SHAP value (price impact ₹)", gridcolor: "rgba(139,146,168,.15)", zeroline: true, zerolinecolor: "rgba(139,146,168,.4)" },
    yaxis: { automargin: true },
    margin: { l: 10, r: 60, t: 10, b: 50 },
    height: 340,
  }, { responsive: true, displayModeBar: false });
}

// ── LIME Chart ───────────────────────────────────────────────────────────────
function drawLime(data) {
  const items  = data.contributions.slice(0, 10);
  const labels = items.map(d => d.rule);
  const values = items.map(d => d.value);
  const colors = values.map(v => v >= 0 ? "#6366f1" : "#f59e0b");

  Plotly.newPlot("lime-chart", [{
    type: "bar",
    orientation: "h",
    x: values,
    y: labels,
    marker: { color: colors },
    hovertemplate: "%{y}<br>Weight: %{x:.4f}<extra></extra>",
  }], {
    paper_bgcolor: paperBg(),
    plot_bgcolor:  plotBg(),
    font: { family: "Inter", color: fontColor(), size: 12 },
    xaxis: { title: "LIME weight", gridcolor: "rgba(139,146,168,.15)", zeroline: true, zerolinecolor: "rgba(139,146,168,.4)" },
    yaxis: { automargin: true },
    margin: { l: 10, r: 20, t: 10, b: 50 },
    height: 340,
  }, { responsive: true, displayModeBar: false });
}

// ── Feature Importance Chart ──────────────────────────────────────────────────
function drawImportance(data) {
  const items  = data.features;
  const labels = items.map(d => d.feature);
  const values = items.map(d => d.importance);

  Plotly.newPlot("importance-chart", [{
    type: "bar",
    orientation: "h",
    x: values,
    y: labels,
    marker: {
      color: values,
      colorscale: [[0,"#1a2040"],[0.5,"#6366f1"],[1,"#10b981"]],
    },
    hovertemplate: "%{y}<br>Importance: %{x:.4f}<extra></extra>",
  }], {
    paper_bgcolor: paperBg(),
    plot_bgcolor:  plotBg(),
    font: { family: "Inter", color: fontColor(), size: 12 },
    xaxis: { title: "Gain importance", gridcolor: "rgba(139,146,168,.15)" },
    yaxis: { automargin: true },
    margin: { l: 10, r: 20, t: 10, b: 50 },
    height: 340,
  }, { responsive: true, displayModeBar: false });
}

// ── Load metrics on startup ───────────────────────────────────────────────────
(async () => {
  try {
    const r = await fetch(`${API}/metrics`);
    if (r.ok) {
      const m = await r.json();
      document.getElementById("header-r2").textContent =
        `R² ${m.r2}  |  CV ${m.cv_r2}`;
    }
  } catch(_) { /* server may not be running yet */ }
})();
