#!/usr/bin/env python3
"""Generates an interactive, standalone HTML dashboard for feature exploration and benchmark comparison."""

import json
import os
import sys
import pandas as pd

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_html_dashboard(
    interpret_json_path: str = "results/feature_interpretability_report.json",
    metrics_csv_path: str = "results/evaluation_metrics.csv",
    output_html_path: str = "results/dashboard.html",
) -> str:
    # Load interpretation data
    features = []
    if os.path.exists(interpret_json_path):
        with open(interpret_json_path, "r") as f:
            features = json.load(f)

    # Load evaluation data
    eval_rows = []
    if os.path.exists(metrics_csv_path):
        df = pd.read_csv(metrics_csv_path)
        eval_rows = df.to_dict(orient="records")

    total_feats = len(features)
    interpretable_count = sum(1 for f in features if f.get("status") == "interpretable")
    poly_count = sum(1 for f in features if f.get("status") == "polysemantic_or_noisy")
    dead_count = sum(1 for f in features if f.get("status") == "dead")
    interpretable_pct = round((interpretable_count / total_feats * 100) if total_feats else 0, 1)

    # Generate feature cards HTML
    feature_cards_html = []
    for f in features:
        feat_id = f.get("feature_idx")
        status = f.get("status", "unknown")
        sample_type = f.get("sample_type", "unknown")
        max_act = f.get("max_activation", 0.0)
        hypothesis = f.get("hypothesis", "N/A")
        spec = f.get("validation_metrics", {}).get("specificity", 0.0)
        top_tokens = ", ".join(f.get("top_tokens", [])) or "None"

        badge_color = "#10b981" if status == "interpretable" else "#f59e0b" if status == "polysemantic_or_noisy" else "#ef4444"

        snippets_html = []
        for s in f.get("snippets", [])[:2]:
            act = s.get("activation", 0.0)
            tok = s.get("token", "")
            snip = s.get("snippet", "")
            # Highlight token in snippet
            highlighted = snip.replace(tok, f"<span class='hl' style='background: rgba(59, 130, 246, 0.4); padding: 1px 4px; border-radius: 3px; font-weight: 600;'>{tok}</span>") if tok else snip
            snippets_html.append(f"""
            <div class="snippet-box">
                <span class="act-tag">act={act:.2f}</span>
                <span class="snip-text">{highlighted}</span>
            </div>
            """)

        snippets_rendered = "".join(snippets_html) if snippets_html else "<div class='snippet-empty'>No activations recorded</div>"

        card = f"""
        <div class="feature-card" data-status="{status}" data-search="{feat_id} {hypothesis.lower()} {top_tokens.lower()}">
            <div class="card-header">
                <span class="feat-id">Feature #{feat_id}</span>
                <span class="badge" style="background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}44;">{status.replace('_', ' ').title()}</span>
            </div>
            <div class="card-body">
                <div class="meta-row">
                    <span class="meta-label">Sample:</span> <span class="meta-val">{sample_type.replace('_', ' ').title()}</span>
                    <span class="meta-label" style="margin-left: 12px;">Max Act:</span> <span class="meta-val">{max_act:.2f}</span>
                    <span class="meta-label" style="margin-left: 12px;">Specificity:</span> <span class="meta-val">{spec:+.2f}</span>
                </div>
                <div class="hypothesis-box">
                    <strong>Hypothesis:</strong> {hypothesis}
                </div>
                <div class="snippets-container">
                    {snippets_rendered}
                </div>
            </div>
        </div>
        """
        feature_cards_html.append(card)

    cards_rendered = "\n".join(feature_cards_html)

    # Generate Benchmark Rows HTML
    eval_table_rows = []
    for r in eval_rows:
        method = r.get("method", "")
        alpha = r.get("alpha", 0.0)
        judge = r.get("judge_score", 0.0)
        ppl = r.get("perplexity", 0.0)
        ppl_delta = r.get("perplexity_delta", 0.0)

        method_badge = "#3b82f6" if "SAE" in method else "#8b5cf6" if "Diff" in method else "#64748b"

        eval_table_rows.append(f"""
        <tr>
            <td><span class="method-tag" style="background: {method_badge}22; color: {method_badge};">{method}</span></td>
            <td><strong>{alpha:+.1f}</strong></td>
            <td>{judge:.4f}</td>
            <td>{ppl:.1f}</td>
            <td style="color: {'#ef4444' if ppl_delta > 10 else '#10b981' if ppl_delta <= 0 else '#f59e0b'}; font-weight: 600;">{ppl_delta:+.2f}</td>
        </tr>
        """)
    eval_table_rendered = "\n".join(eval_table_rows)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interpretable Concept Steering via Sparse Autoencoders - Dashboard</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #3b82f6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background: var(--bg); color: var(--text); padding: 32px 24px; line-height: 1.5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ margin-bottom: 32px; border-bottom: 1px solid var(--border); padding-bottom: 24px; }}
        h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; color: #fff; }}
        p.subtitle {{ color: var(--text-muted); font-size: 15px; }}

        /* Stat Grid */
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }}
        .stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }}
        .stat-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 6px; font-weight: 600; }}
        .stat-val {{ font-size: 26px; font-weight: 700; color: #fff; }}
        .stat-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}

        /* Sections */
        .section-title {{ font-size: 20px; font-weight: 600; margin: 32px 0 16px 0; color: #fff; }}

        /* Table */
        .table-container {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow-x: auto; margin-bottom: 32px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
        th, td {{ padding: 14px 18px; border-bottom: 1px solid var(--border); }}
        th {{ background: rgba(0,0,0,0.2); color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; }}
        tr:last-child td {{ border-bottom: none; }}
        .method-tag {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}

        /* Filter Controls */
        .controls {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
        .search-input {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; color: #fff; font-size: 14px; flex: 1; min-width: 260px; outline: none; }}
        .search-input:focus {{ border-color: var(--primary); }}
        .filter-btn {{ background: var(--surface); border: 1px solid var(--border); color: var(--text-muted); padding: 10px 16px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.15s; }}
        .filter-btn.active {{ background: var(--primary); color: #fff; border-color: var(--primary); }}

        /* Feature Grid */
        .feature-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
        .feature-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px; display: flex; flex-direction: column; transition: transform 0.15s, border-color 0.15s; }}
        .feature-card:hover {{ border-color: #475569; transform: translateY(-2px); }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .feat-id {{ font-size: 16px; font-weight: 700; color: #fff; }}
        .badge {{ font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 20px; }}
        .meta-row {{ font-size: 12px; color: var(--text-muted); margin-bottom: 12px; }}
        .meta-label {{ color: var(--text-muted); }}
        .meta-val {{ color: #fff; font-weight: 600; }}
        .hypothesis-box {{ background: rgba(0,0,0,0.25); border-left: 3px solid var(--primary); padding: 10px 12px; font-size: 13px; border-radius: 0 6px 6px 0; margin-bottom: 12px; color: #e2e8f0; }}
        .snippet-box {{ background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 8px 10px; font-size: 12px; font-family: monospace; margin-top: 6px; display: flex; gap: 8px; align-items: flex-start; }}
        .act-tag {{ background: rgba(59,130,246,0.2); color: #93c5fd; padding: 1px 6px; border-radius: 4px; font-size: 10px; white-space: nowrap; font-weight: 600; }}
        .snip-text {{ color: #cbd5e1; word-break: break-all; }}
        .snippet-empty {{ font-size: 12px; color: var(--text-muted); font-style: italic; padding: 6px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Interpretable Concept Steering via Sparse Autoencoders</h1>
            <p class="subtitle">Mechanistic interpretability dashboard for GPT-2 small (Layer 6 residual stream) &middot; Dictionary size: 6,144 latents</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Model & Hook</div>
                <div class="stat-val" style="font-size: 20px;">GPT-2 Small</div>
                <div class="stat-sub">Layer 6 Residual Stream (d=768)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">SAE Expansion</div>
                <div class="stat-val">8&times;</div>
                <div class="stat-sub">6,144 Overcomplete Latents</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Interpretable Rate</div>
                <div class="stat-val" style="color: var(--success);">{interpretable_pct}%</div>
                <div class="stat-sub">{interpretable_count} clean / {total_feats} analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Dead Feature Rate</div>
                <div class="stat-val" style="color: var(--danger);">{round((dead_count / total_feats * 100) if total_feats else 0, 1)}%</div>
                <div class="stat-sub">{dead_count} inactive across corpus</div>
            </div>
        </div>

        <h2 class="section-title">Quantitative Steering Benchmark: SAE vs Difference-of-Means</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Steering Method</th>
                        <th>Alpha (&alpha;)</th>
                        <th>Judge Score (Efficacy &uarr;)</th>
                        <th>Perplexity (PPL &darr;)</th>
                        <th>Perplexity Delta (&Delta;PPL)</th>
                    </tr>
                </thead>
                <tbody>
                    {eval_table_rendered}
                </tbody>
            </table>
        </div>

        <h2 class="section-title">Systematic Feature Explorer ({total_feats} Sampled Latents)</h2>
        <div class="controls">
            <input type="text" id="searchInput" class="search-input" placeholder="Search by feature ID, concept keyword, or token (e.g., 'python', 'def', 'sentiment')...">
            <button class="filter-btn active" onclick="filterStatus('all')">All ({total_feats})</button>
            <button class="filter-btn" onclick="filterStatus('interpretable')">Interpretable ({interpretable_count})</button>
            <button class="filter-btn" onclick="filterStatus('polysemantic_or_noisy')">Polysemantic ({poly_count})</button>
            <button class="filter-btn" onclick="filterStatus('dead')">Dead ({dead_count})</button>
        </div>

        <div class="feature-grid" id="featureGrid">
            {cards_rendered}
        </div>
    </div>

    <script>
        let currentFilter = 'all';

        function filterStatus(status) {{
            currentFilter = status;
            document.querySelectorAll('.filter-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.textContent.toLowerCase().startsWith(status));
            }});
            applyFilters();
        }}

        function applyFilters() {{
            const query = document.getElementById('searchInput').value.toLowerCase().trim();
            const cards = document.querySelectorAll('.feature-card');

            cards.forEach(card => {{
                const status = card.getAttribute('data-status');
                const text = card.getAttribute('data-search');

                const matchesStatus = (currentFilter === 'all' || status === currentFilter);
                const matchesQuery = (!query || text.includes(query));

                card.style.display = (matchesStatus && matchesQuery) ? 'flex' : 'none';
            }});
        }}

        document.getElementById('searchInput').addEventListener('input', applyFilters);
    </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w") as f:
        f.write(html_content)

    print(f"Generated standalone interactive dashboard: {output_html_path}")
    return output_html_path


if __name__ == "__main__":
    generate_html_dashboard()
