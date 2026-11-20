/* Shared client state and HiDPI Canvas scientific renderers for Wireless-Signal Coverage Mapper. */

const SignalApp = (() => {
  const STORAGE_CSV = "wcm_custom_csv_v1";
  const STORAGE_LABEL = "wcm_custom_label_v1";
  const STORAGE_WEAK = "wcm_weak_thr_v1";
  const STORAGE_STRONG = "wcm_strong_thr_v1";

  const COLORS = {
    strong: "#157A6E",
    fair: "#C27803",
    weak: "#C83E2B",
    ink: "#171614",
    grid: "#E5DFD3",
    muted: "#78716A",
    bg: "#FFFFFF",
  };

  function getThresholds() {
    const weak = parseFloat(sessionStorage.getItem(STORAGE_WEAK) || "-75");
    const strong = parseFloat(sessionStorage.getItem(STORAGE_STRONG) || "-60");
    return { weak, strong };
  }

  function setThresholds(weak, strong) {
    sessionStorage.setItem(STORAGE_WEAK, String(weak));
    sessionStorage.setItem(STORAGE_STRONG, String(strong));
  }

  function saveCustomCSV(csvText, label) {
    sessionStorage.setItem(STORAGE_CSV, csvText);
    sessionStorage.setItem(STORAGE_LABEL, label || "Uploaded CSV");
  }

  function clearCustomCSV() {
    sessionStorage.removeItem(STORAGE_CSV);
    sessionStorage.removeItem(STORAGE_LABEL);
  }

  function hasCustomCSV() {
    return Boolean(sessionStorage.getItem(STORAGE_CSV));
  }

  async function fetchAnalysis(overrides = {}) {
    const thr = getThresholds();
    const weak_threshold = overrides.weak_threshold ?? thr.weak;
    const strong_threshold = overrides.strong_threshold ?? thr.strong;
    const customCsv = sessionStorage.getItem(STORAGE_CSV);
    const customLabel = sessionStorage.getItem(STORAGE_LABEL) || "Uploaded CSV";

    let resp;
    if (customCsv) {
      resp = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          csv_text: customCsv,
          source_label: customLabel,
          is_synthetic: false,
          weak_threshold,
          strong_threshold,
        }),
      });
    } else {
      const qs = new URLSearchParams({
        weak_threshold: String(weak_threshold),
        strong_threshold: String(strong_threshold),
      });
      resp = await fetch(`/api/sample?${qs.toString()}`);
    }

    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || "Failed to analyze dataset.");
    }
    updateHeaderPill(data);
    return data;
  }

  function updateHeaderPill(data) {
    const pill = document.getElementById("nav-dataset-pill");
    if (!pill || !data || !data.meta) return;
    const kind = data.meta.is_synthetic ? "Synthetic" : "Custom CSV";
    pill.textContent = `${kind} · ${data.meta.valid_rows} pts`;
  }

  async function downloadCleanedCSV() {
    const thr = getThresholds();
    const customCsv = sessionStorage.getItem(STORAGE_CSV) || "";
    const resp = await fetch("/api/export/csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        csv_text: customCsv,
        weak_threshold: thr.weak,
        strong_threshold: thr.strong,
      }),
    });
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cleaned_wireless_survey.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function setupCanvas(canvas, aspectRatio = 0.62) {
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.parentElement.clientWidth || 680;
    const height = Math.max(260, Math.round(width * aspectRatio));
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function rssiToColor(rssi) {
    // Grounded scientific ramp: vermilion (-92) -> warm ochre (-73) -> pine teal (-48)
    const t = Math.max(0, Math.min(1, (rssi - (-92)) / ((-48) - (-92))));
    let r, g, b;
    if (t < 0.5) {
      const k = t / 0.5;
      r = Math.round(200 + (214 - 200) * k);
      g = Math.round(62 + (148 - 62) * k);
      b = Math.round(43 + (38 - 43) * k);
    } else {
      const k = (t - 0.5) / 0.5;
      r = Math.round(214 + (21 - 214) * k);
      g = Math.round(148 + (122 - 148) * k);
      b = Math.round(38 + (110 - 38) * k);
    }
    return `rgb(${r},${g},${b})`;
  }

  function drawScatterMap(canvas, data, options = {}) {
    const { ctx, width, height } = setupCanvas(canvas, options.aspectRatio || 0.64);
    const pad = { top: 24, right: 24, bottom: 42, left: 48 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const xMax = data.heatmap.bounds.x_max || 40;
    const yMax = data.heatmap.bounds.y_max || 30;
    const locFilter = options.locationFilter || "All";

    const sx = (x) => pad.left + (x / xMax) * plotW;
    const sy = (y) => pad.top + plotH - (y / yMax) * plotH;

    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, width, height);

    // Subtle gridlines
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.font = "11px 'IBM Plex Mono', monospace";
    ctx.fillStyle = COLORS.muted;

    for (let x = 0; x <= xMax; x += 10) {
      const px = sx(x);
      ctx.beginPath();
      ctx.moveTo(px, pad.top);
      ctx.lineTo(px, pad.top + plotH);
      ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(`${x}m`, px, height - 20);
    }
    for (let y = 0; y <= yMax; y += 5) {
      const py = sy(y);
      ctx.beginPath();
      ctx.moveTo(pad.left, py);
      ctx.lineTo(pad.left + plotW, py);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(`${y}m`, pad.left - 8, py + 4);
    }

    // Axes border
    ctx.strokeStyle = "#CFC7B8";
    ctx.strokeRect(pad.left, pad.top, plotW, plotH);

    // Highlight unsurveyed gap bounding box gently on map if requested
    if (options.showGapBox !== false) {
      ctx.save();
      ctx.strokeStyle = "#9C9488";
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(sx(28), sy(29), sx(39) - sx(28), sy(18) - sy(29));
      ctx.fillStyle = "#78716A";
      ctx.font = "10px 'IBM Plex Mono', monospace";
      ctx.textAlign = "left";
      ctx.fillText("Unsurveyed gap", sx(28.5), sy(27.5));
      ctx.restore();
    }

    // Draw survey points
    const rendered = [];
    for (const p of data.points) {
      if (locFilter !== "All" && p.location !== locFilter) continue;
      const px = sx(p.x);
      const py = sy(p.y);
      const color =
        p.zone === "Strong" ? COLORS.strong : p.zone === "Fair" ? COLORS.fair : COLORS.weak;

      ctx.beginPath();
      ctx.arc(px, py, 5.2, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = "#FFFFFF";
      ctx.stroke();
      rendered.push({ px, py, p });
    }

    // Draw AP origin triangle
    const ap = data.model.ap_origin;
    const apx = sx(ap.x);
    const apy = sy(ap.y);
    ctx.beginPath();
    ctx.moveTo(apx, apy - 8);
    ctx.lineTo(apx - 7, apy + 6);
    ctx.lineTo(apx + 7, apy + 6);
    ctx.closePath();
    ctx.fillStyle = COLORS.ink;
    ctx.fill();
    ctx.font = "600 11px 'IBM Plex Mono', monospace";
    ctx.fillStyle = COLORS.ink;
    ctx.textAlign = "left";
    ctx.fillText("AP-01", apx + 10, apy + 4);

    // Axis titles
    ctx.font = "11px 'IBM Plex Sans', sans-serif";
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "center";
    ctx.fillText("Floor X Coordinate (meters)", pad.left + plotW / 2, height - 4);

    // Attach hover tooltip if element exists
    if (options.tooltipEl) {
      canvas.onmousemove = (ev) => {
        const rect = canvas.getBoundingClientRect();
        const mx = ev.clientX - rect.left;
        const my = ev.clientY - rect.top;
        let hit = null;
        let best = 12;
        for (const item of rendered) {
          const d = Math.hypot(item.px - mx, item.py - my);
          if (d < best) {
            best = d;
            hit = item;
          }
        }
        if (hit) {
          const p = hit.p;
          options.tooltipEl.textContent = `${p.point_id} · ${p.location} · (${p.x}m, ${p.y}m) · ${p.signal_dbm} dBm (${p.zone})`;
          options.tooltipEl.style.left = `${Math.min(width - 240, Math.max(8, hit.px + 12))}px`;
          options.tooltipEl.style.top = `${Math.max(8, hit.py - 32)}px`;
          options.tooltipEl.style.opacity = "1";
        } else {
          options.tooltipEl.style.opacity = "0";
        }
      };
      canvas.onmouseleave = () => {
        options.tooltipEl.style.opacity = "0";
      };
    }
  }

  function drawHeatmap(canvas, data, options = {}) {
    const { ctx, width, height } = setupCanvas(canvas, 0.64);
    const pad = { top: 24, right: 24, bottom: 42, left: 48 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const hm = data.heatmap;
    const xMax = hm.bounds.x_max;
    const yMax = hm.bounds.y_max;
    const cellW = plotW / hm.nx;
    const cellH = plotH / hm.ny;

    const sx = (x) => pad.left + (x / xMax) * plotW;
    const sy = (y) => pad.top + plotH - (y / yMax) * plotH;

    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, width, height);

    const showGaps = options.showGaps !== false;
    const showPoints = options.showPoints !== false;

    for (const c of hm.cells) {
      const x0 = pad.left + c.col * cellW;
      const y0 = pad.top + plotH - (c.row + 1) * cellH;
      ctx.fillStyle = rssiToColor(c.rssi_dbm);
      ctx.fillRect(x0, y0, cellW + 0.6, cellH + 0.6);

      if (showGaps && c.is_gap) {
        ctx.strokeStyle = "rgba(23,22,20,0.45)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x0, y0 + cellH);
        ctx.lineTo(x0 + cellW, y0);
        ctx.stroke();
      }
      if (c.zone === "Weak") {
        ctx.strokeStyle = "rgba(255,255,255,0.22)";
        ctx.lineWidth = 0.6;
        ctx.strokeRect(x0, y0, cellW, cellH);
      }
    }

    if (showPoints) {
      for (const p of data.points) {
        ctx.beginPath();
        ctx.arc(sx(p.x), sy(p.y), 2.8, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(23,22,20,0.72)";
        ctx.fill();
      }
    }

    // AP marker
    const ap = data.model.ap_origin;
    const apx = sx(ap.x);
    const apy = sy(ap.y);
    ctx.beginPath();
    ctx.moveTo(apx, apy - 8);
    ctx.lineTo(apx - 7, apy + 6);
    ctx.lineTo(apx + 7, apy + 6);
    ctx.closePath();
    ctx.fillStyle = "#FFFFFF";
    ctx.fill();
    ctx.lineWidth = 1.8;
    ctx.strokeStyle = COLORS.ink;
    ctx.stroke();

    // Axes ticks
    ctx.strokeStyle = COLORS.ink;
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.left, pad.top, plotW, plotH);
    ctx.font = "11px 'IBM Plex Mono', monospace";
    ctx.fillStyle = COLORS.muted;
    for (let x = 0; x <= xMax; x += 10) {
      ctx.textAlign = "center";
      ctx.fillText(`${x}m`, sx(x), height - 20);
    }
    for (let y = 0; y <= yMax; y += 5) {
      ctx.textAlign = "right";
      ctx.fillText(`${y}m`, pad.left - 8, sy(y) + 4);
    }

    if (options.tooltipEl) {
      canvas.onmousemove = (ev) => {
        const rect = canvas.getBoundingClientRect();
        const mx = ev.clientX - rect.left;
        const my = ev.clientY - rect.top;
        const col = Math.floor((mx - pad.left) / cellW);
        const row = Math.floor((pad.top + plotH - my) / cellH);
        if (col >= 0 && col < hm.nx && row >= 0 && row < hm.ny) {
          const cell = hm.cells[row * hm.nx + col];
          const gapTxt = cell.is_gap ? " · UNSURVEYED GAP" : "";
          options.tooltipEl.textContent = `(${cell.cx}m, ${cell.cy}m) · Est ${cell.rssi_dbm} dBm (${cell.zone}) · Nearest ${cell.nearest_sample_m}m${gapTxt}`;
          options.tooltipEl.style.left = `${Math.min(width - 280, Math.max(8, mx + 12))}px`;
          options.tooltipEl.style.top = `${Math.max(8, my - 32)}px`;
          options.tooltipEl.style.opacity = "1";
        } else {
          options.tooltipEl.style.opacity = "0";
        }
      };
      canvas.onmouseleave = () => {
        options.tooltipEl.style.opacity = "0";
      };
    }
  }

  function drawModelPlot(canvas, data) {
    const { ctx, width, height } = setupCanvas(canvas, 0.58);
    const pad = { top: 24, right: 24, bottom: 44, left: 54 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const pts = data.model.points;
    const curve = data.model.curve;
    const dMax = Math.ceil(Math.max(...pts.map((p) => p.distance_m)) / 5) * 5;
    const yMin = -100;
    const yMax = -30;

    const sx = (d) => pad.left + (d / dMax) * plotW;
    const sy = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = COLORS.grid;
    ctx.font = "11px 'IBM Plex Mono', monospace";
    ctx.fillStyle = COLORS.muted;
    for (let d = 0; d <= dMax; d += 5) {
      const px = sx(d);
      ctx.beginPath();
      ctx.moveTo(px, pad.top);
      ctx.lineTo(px, pad.top + plotH);
      ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(`${d}m`, px, height - 22);
    }
    for (let v = yMin; v <= yMax; v += 10) {
      const py = sy(v);
      ctx.beginPath();
      ctx.moveTo(pad.left, py);
      ctx.lineTo(pad.left + plotW, py);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(`${v}`, pad.left - 8, py + 4);
    }

    // Weak threshold line
    const thr = data.thresholds.weak_dbm;
    ctx.save();
    ctx.strokeStyle = COLORS.weak;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, sy(thr));
    ctx.lineTo(pad.left + plotW, sy(thr));
    ctx.stroke();
    ctx.restore();

    // Measured scatter
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(sx(p.distance_m), sy(p.measured_dbm), 4, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(21, 122, 110, 0.72)";
      ctx.fill();
    }

    // Fitted log-distance curve
    ctx.beginPath();
    curve.forEach((pt, idx) => {
      const px = sx(pt.distance_m);
      const py = sy(pt.predicted_dbm);
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = COLORS.weak;
    ctx.lineWidth = 2.4;
    ctx.stroke();

    ctx.strokeStyle = "#CFC7B8";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.left, pad.top, plotW, plotH);

    ctx.font = "11px 'IBM Plex Sans', sans-serif";
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "center";
    ctx.fillText("Distance from Access Point (meters)", pad.left + plotW / 2, height - 4);
  }

  function drawHistogram(canvas, data) {
    const { ctx, width, height } = setupCanvas(canvas, 0.56);
    const pad = { top: 24, right: 24, bottom: 46, left: 46 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const bins = data.histogram;
    const maxCnt = Math.max(1, ...bins.map((b) => b.count));
    const barW = plotW / bins.length;

    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, width, height);

    const thr = data.thresholds.weak_dbm;
    const strongThr = data.thresholds.strong_dbm;

    bins.forEach((b, i) => {
      const h = (b.count / maxCnt) * (plotH - 10);
      const x = pad.left + i * barW + 3;
      const y = pad.top + plotH - h;
      const mid = (b.min_dbm + b.max_dbm) / 2;

      ctx.fillStyle = mid < thr ? COLORS.weak : mid >= strongThr ? COLORS.strong : COLORS.fair;
      ctx.fillRect(x, y, Math.max(4, barW - 6), h);

      ctx.font = "10px 'IBM Plex Mono', monospace";
      ctx.fillStyle = COLORS.ink;
      ctx.textAlign = "center";
      if (b.count > 0) {
        ctx.fillText(String(b.count), x + (barW - 6) / 2, y - 5);
      }
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(`${b.min_dbm}`, x + (barW - 6) / 2, height - 22);
    });

    ctx.strokeStyle = "#CFC7B8";
    ctx.strokeRect(pad.left, pad.top, plotW, plotH);
    ctx.font = "11px 'IBM Plex Sans', sans-serif";
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "center";
    ctx.fillText("Measured RSSI Bin Lower Edge (dBm)", pad.left + plotW / 2, height - 4);
  }

  return {
    getThresholds,
    setThresholds,
    saveCustomCSV,
    clearCustomCSV,
    hasCustomCSV,
    fetchAnalysis,
    downloadCleanedCSV,
    drawScatterMap,
    drawHeatmap,
    drawModelPlot,
    drawHistogram,
  };
})();
