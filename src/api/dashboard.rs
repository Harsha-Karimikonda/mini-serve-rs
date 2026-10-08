pub const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mini-Serve Rust Core Dashboard</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #111827;
      --card-border: #1f293d;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --primary: #f97316;
      --primary-hover: #ea580c;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --accent: #38bdf8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
    }
    .title-group h1 { font-size: 24px; font-weight: 700; color: #fff; }
    .title-group p { font-size: 14px; color: var(--text-muted); }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 600;
      background: rgba(16, 185, 129, 0.15);
      color: var(--success);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; }
    .card-title { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 8px; }
    .card-value { font-size: 28px; font-weight: 700; color: #fff; }
    .card-subtext { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
    .progress-bar { width: 100%; height: 8px; background: #1f293d; border-radius: 4px; margin-top: 12px; overflow: hidden; }
    .progress-fill { height: 100%; background: var(--primary); width: 0%; transition: width 0.3s ease; }
    .playground { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 24px; }
    .playground h2 { font-size: 18px; margin-bottom: 16px; }
    .input-group { margin-bottom: 16px; }
    label { display: block; font-size: 13px; font-weight: 500; color: var(--text-muted); margin-bottom: 6px; }
    textarea { width: 100%; height: 90px; background: #0b0f19; border: 1px solid var(--card-border); border-radius: 8px; color: #fff; padding: 12px; font-family: inherit; font-size: 14px; resize: vertical; }
    textarea:focus { outline: none; border-color: var(--primary); }
    .btn { background: var(--primary); color: #fff; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    .btn:hover { background: var(--primary-hover); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .output-box { margin-top: 16px; background: #0b0f19; border: 1px solid var(--card-border); border-radius: 8px; padding: 16px; min-height: 100px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; white-space: pre-wrap; word-break: break-word; color: #38bdf8; }
    .rust-pill { background: rgba(249, 115, 22, 0.2); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title-group">
        <h1>Mini-Serve <span class="rust-pill">RUST CORE v0.2.0</span></h1>
        <p>Continuous Batching & Paged KV-Cache Engine</p>
      </div>
      <div class="status-badge">
        <span class="status-dot"></span>
        <span id="system-status">HEALTHY</span>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="card-title">Token Throughput</div>
        <div class="card-value" id="val-tps">0.0 <span style="font-size: 16px; color: var(--text-muted)">tok/s</span></div>
        <div class="card-subtext" id="sub-tokens">Total Tokens: 0</div>
      </div>
      <div class="card">
        <div class="card-title">Logical KV-Cache Utilization</div>
        <div class="card-value" id="val-cache-util">0.0%</div>
        <div class="progress-bar"><div class="progress-fill" id="fill-cache"></div></div>
        <div class="card-subtext" id="sub-cache">Free Blocks: 1024 / 1024</div>
      </div>
      <div class="card">
        <div class="card-title">Queue & Active Sequences</div>
        <div class="card-value" id="val-active">0 <span style="font-size: 16px; color: var(--text-muted)">active</span></div>
        <div class="card-subtext" id="sub-queue">Queue Backlog: 0 reqs</div>
      </div>
      <div class="card">
        <div class="card-title">Rolling Latency (p50)</div>
        <div class="card-value" id="val-lat">0.0 <span style="font-size: 16px; color: var(--text-muted)">ms</span></div>
        <div class="card-subtext" id="sub-workers">Active Workers: 2</div>
      </div>
    </div>

    <div class="playground">
      <h2>Interactive Streaming Playground</h2>
      <div class="input-group">
        <label for="prompt">Prompt</label>
        <textarea id="prompt">Explain continuous batching in high-throughput LLM serving systems.</textarea>
      </div>
      <div style="display: flex; gap: 12px; align-items: center;">
        <button class="btn" id="btn-generate" onclick="streamGenerate()">Stream Completion</button>
        <button class="btn" id="btn-cancel" onclick="cancelStream()" style="background: var(--danger); display: none;">Cancel</button>
        <span id="gen-stats" style="font-size: 12px; color: var(--text-muted)"></span>
      </div>
      <div class="output-box" id="output">Output will stream here in real time...</div>
    </div>
  </div>

  <script>
    let abortCtrl = null;

    async function updateStatus() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();
        
        document.getElementById('val-tps').innerHTML = `${data.tokens_per_sec.toFixed(1)} <span style="font-size: 16px; color: var(--text-muted)">tok/s</span>`;
        document.getElementById('sub-tokens').textContent = `Total Tokens: ${data.total_tokens} (${data.total_requests} reqs)`;

        const cache = data.cache || {};
        const utilPct = ((cache.utilization || 0) * 100).toFixed(1);
        document.getElementById('val-cache-util').textContent = `${utilPct}%`;
        document.getElementById('fill-cache').style.width = `${utilPct}%`;
        document.getElementById('sub-cache').textContent = `Free Blocks: ${cache.free_blocks || 0} / ${cache.total_blocks || 0}`;

        let totalQueue = 0, totalActive = 0, avgLat = 0;
        if (data.workers) {
          data.workers.forEach(w => {
            totalQueue += w.queue_depth || 0;
            totalActive += w.active_count || 0;
            avgLat = Math.max(avgLat, w.avg_latency_ms || 0);
          });
        }
        document.getElementById('val-active').innerHTML = `${totalActive} <span style="font-size: 16px; color: var(--text-muted)">active</span>`;
        document.getElementById('sub-queue').textContent = `Queue Backlog: ${totalQueue} reqs`;
        document.getElementById('val-lat').innerHTML = `${avgLat.toFixed(1)} <span style="font-size: 16px; color: var(--text-muted)">ms</span>`;
        document.getElementById('sub-workers').textContent = `Workers: ${(data.workers || []).length} registered`;
      } catch (e) {
        console.error("status fetch error", e);
      }
    }

    setInterval(updateStatus, 1000);
    updateStatus();

    async function streamGenerate() {
      const prompt = document.getElementById('prompt').value;
      const output = document.getElementById('output');
      const btnGen = document.getElementById('btn-generate');
      const btnCancel = document.getElementById('btn-cancel');
      const stats = document.getElementById('gen-stats');

      output.textContent = "";
      btnGen.disabled = true;
      btnCancel.style.display = "inline-block";
      stats.textContent = "Connecting...";

      abortCtrl = new AbortController();
      const t0 = performance.now();
      let tokenCount = 0;

      try {
        const res = await fetch('/v1/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'mock',
            prompt: prompt,
            max_tokens: 128,
            stream: true
          }),
          signal: abortCtrl.signal
        });

        if (!res.ok) {
          const err = await res.json();
          output.textContent = "Error: " + JSON.stringify(err);
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop();

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const dataStr = line.slice(6).trim();
              if (dataStr === "[DONE]") break;
              try {
                const chunk = JSON.parse(dataStr);
                const text = chunk.choices[0].text;
                output.textContent += text;
                tokenCount++;
                const elapsed = (performance.now() - t0) / 1000;
                stats.textContent = `${tokenCount} tokens (${(tokenCount / elapsed).toFixed(1)} tok/s)`;
              } catch (_) {}
            }
          }
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          output.textContent += "\n\n[Generation cancelled by user]";
        } else {
          output.textContent += "\n\n[Error: " + err.message + "]";
        }
      } finally {
        btnGen.disabled = false;
        btnCancel.style.display = "none";
        abortCtrl = null;
      }
    }

    function cancelStream() {
      if (abortCtrl) {
        abortCtrl.abort();
      }
    }
  </script>
</body>
</html>
"#;
