# # # """
# # # FINAL MODEL COMPARISON (Robust + Paper Ready)
# # # ============================================

# # # Compares multiple detox models on:
# # # 1. Overall toxic vs non-toxic %
# # # 2. Subreddit-level comparison (Top 10)

# # # Uses:
# # #     chain_avg_old_toxicity as threshold source

# # # Outputs:
# # #     model_comparison_table.csv
# # #     model_comparison_bar.png
# # #     model_comparison_stacked.png
# # #     subreddit_top10_comparison.png
# # # """

# # # from pathlib import Path
# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import matplotlib
# # # matplotlib.use("Agg")


# # # # ─────────────────────────────────────────────
# # # # CONFIG
# # # # ─────────────────────────────────────────────
# # # CONFIG = {
# # #     "models": [
# # #         {"name": "Model_A", "file": "reddit_detoxified.parquet"},
# # #         {"name": "Model_B", "file": "LLM_reddit_detoxified.parquet"},
# # #         {"name": "Model_C", "file": "PTSD_reddit_detoxified.parquet"},
# # #     ],

# # #     "group_col": "subreddit",
# # #     "comments_col": "comments",

# # #     "toxic_label_key": "toxic",
# # #     "score_key": "toxicity_score",

# # #     # KEY REQUIREMENT
# # #     "threshold_col": "chain_avg_old_toxicity",
# # #     "score_threshold": 0.5,

# # #     "top_n": 10,
# # #     "output_dir": "outputs_model_compare"
# # # }


# # # # ─────────────────────────────────────────────
# # # # Resolve toxic
# # # # ─────────────────────────────────────────────
# # # def resolve_toxic(comment, cfg):
# # #     lk = cfg["toxic_label_key"]
# # #     sk = cfg["score_key"]
# # #     th = cfg["score_threshold"]

# # #     if lk and lk in comment:
# # #         val = comment[lk]
# # #         if isinstance(val, bool):
# # #             return val
# # #         if isinstance(val, (int, float)):
# # #             return bool(val)
# # #         if isinstance(val, str):
# # #             return val.lower() in ("true", "1", "yes")

# # #     if sk and sk in comment:
# # #         try:
# # #             return float(comment[sk]) >= th
# # #         except:
# # #             pass

# # #     return None


# # # # ─────────────────────────────────────────────
# # # # Flatten
# # # # ─────────────────────────────────────────────
# # # def flatten(df, cfg):
# # #     rows = []

# # #     for _, row in df.iterrows():
# # #         group = row.get(cfg["group_col"], "unknown")
# # #         comments = row.get(cfg["comments_col"])

# # #         if comments is None:
# # #             continue

# # #         try:
# # #             comments = list(comments)
# # #         except:
# # #             continue

# # #         threshold_val = row.get(cfg["threshold_col"], np.nan)

# # #         for c in comments:
# # #             if not isinstance(c, dict):
# # #                 continue

# # #             toxic = resolve_toxic(c, cfg)
# # #             if toxic is None:
# # #                 continue

# # #             rows.append({
# # #                 "subreddit": str(group),
# # #                 "toxic": bool(toxic),
# # #                 "threshold": threshold_val
# # #             })

# # #     return pd.DataFrame(rows)


# # # # ─────────────────────────────────────────────
# # # # Stats
# # # # ─────────────────────────────────────────────
# # # def compute_overall(flat):
# # #     total = len(flat)
# # #     toxic = int(flat["toxic"].sum())

# # #     return {
# # #         "total_comments": total,
# # #         "toxic_pct": round(100 * toxic / total, 2),
# # #         "nontoxic_pct": round(100 - (100 * toxic / total), 2)
# # #     }


# # # def compute_by_subreddit(flat):
# # #     g = flat.groupby("subreddit")

# # #     df = pd.DataFrame({
# # #         "total": g["toxic"].count(),
# # #         "toxic": g["toxic"].sum()
# # #     }).reset_index()

# # #     df["toxic_pct"] = (100 * df["toxic"] / df["total"]).round(2)

# # #     return df.sort_values("total", ascending=False)


# # # # ─────────────────────────────────────────────
# # # # Charts
# # # # ─────────────────────────────────────────────
# # # def plot_overall(df, out_dir):
# # #     x = np.arange(len(df))

# # #     plt.figure(figsize=(8, 5))
# # #     plt.bar(x - 0.2, df["toxic_pct"], width=0.4, label="Toxic")
# # #     plt.bar(x + 0.2, df["nontoxic_pct"], width=0.4, label="Non-toxic")

# # #     plt.xticks(x, df["model"])
# # #     plt.ylabel("%")
# # #     plt.title("Model Comparison — Toxic vs Non-Toxic")
# # #     plt.legend()

# # #     plt.savefig(out_dir / "model_comparison_bar.png", dpi=150)
# # #     plt.close()


# # # def plot_stacked(df, out_dir):
# # #     x = np.arange(len(df))

# # #     plt.figure(figsize=(8, 5))
# # #     plt.bar(x, df["toxic_pct"], label="Toxic")
# # #     plt.bar(x, df["nontoxic_pct"], bottom=df["toxic_pct"], label="Non-toxic")

# # #     plt.xticks(x, df["model"])
# # #     plt.ylabel("%")
# # #     plt.title("Model Comparison — Stacked")
# # #     plt.legend()

# # #     plt.savefig(out_dir / "model_comparison_stacked.png", dpi=150)
# # #     plt.close()


# # # def plot_top10_subreddits(sub_dict, cfg, out_dir):
# # #     """
# # #     Compare models on SAME top 10 subreddits (based on first model)
# # #     """

# # #     base_model = list(sub_dict.keys())[0]
# # #     top10 = sub_dict[base_model].head(cfg["top_n"])["subreddit"]

# # #     plt.figure(figsize=(12, 6))

# # #     for name, df in sub_dict.items():
# # #         df = df[df["subreddit"].isin(top10)]
# # #         df = df.set_index("subreddit").reindex(top10)

# # #         plt.plot(df.index, df["toxic_pct"], marker='o', label=name)

# # #     plt.xticks(rotation=35, ha="right")
# # #     plt.ylabel("Toxic %")
# # #     plt.title("Top 10 Subreddits — Model Comparison")
# # #     plt.legend()

# # #     plt.savefig(out_dir / "subreddit_top10_comparison.png", dpi=150)
# # #     plt.close()


# # # # ─────────────────────────────────────────────
# # # # MAIN
# # # # ─────────────────────────────────────────────
# # # def run(cfg):
# # #     results = []
# # #     sub_results = {}

# # #     for model in cfg["models"]:
# # #         name = model["name"]
# # #         file = Path(model["file"])

# # #         if not file.exists():
# # #             print(f"Missing: {file}")
# # #             continue

# # #         print(f"\nProcessing {name}")

# # #         df = pd.read_parquet(file)
# # #         flat = flatten(df, cfg)

# # #         overall = compute_overall(flat)
# # #         overall["model"] = name
# # #         results.append(overall)

# # #         sub_results[name] = compute_by_subreddit(flat)

# # #     comp_df = pd.DataFrame(results)

# # #     out_dir = Path(cfg["output_dir"])
# # #     out_dir.mkdir(exist_ok=True)

# # #     comp_df.to_csv(out_dir / "model_comparison_table.csv", index=False)

# # #     plot_overall(comp_df, out_dir)
# # #     plot_stacked(comp_df, out_dir)
# # #     plot_top10_subreddits(sub_results, cfg, out_dir)

# # #     print("\nSaved all outputs.")
# # #     # print(comp_df)


# # # if __name__ == "__main__":
# # #     run(CONFIG)


# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as mpatches
# # from matplotlib.gridspec import GridSpec

# # # ─── Load data ────────────────────────────────────────────────────────────────
# # base = pd.read_parquet("reddit_detoxified.parquet")
# # ptsd = pd.read_parquet("PTSD_reddit_detoxified.parquet")
# # llm  = pd.read_parquet("LLM_reddit_detoxified.parquet")

# # MODELS   = ["T5 Detoxifier\n(Baseline)", "PTSD Model\n(Fine-tuned)", "LLM Judge\n(Prompted)"]
# # COLORS   = ["#378ADD", "#1D9E75", "#D85A30"]   # blue, teal, coral
# # ALPHAS   = [0.85, 0.85, 0.85]

# # # ─── Compute per-model mean stats ─────────────────────────────────────────────
# # def stats(df, prefix="avg_"):
# #     """Return a dict of mean values, handling both column name conventions."""
# #     cols = list(df.columns)
# #     def mean(col):
# #         return df[col].mean() if col in cols else np.nan

# #     return {
# #         "cosine_similarity":        mean(f"{prefix}cosine_similarity")
# #                                     if f"{prefix}cosine_similarity" in cols
# #                                     else mean("chain_cosine_similarity_avg"),
# #         "toxicity_change":          mean(f"{prefix}toxicity_change")
# #                                     if f"{prefix}toxicity_change" in cols
# #                                     else mean("chain_toxicity_change_avg"),
# #         "severe_toxicity_change":   mean(f"{prefix}severe_toxicity_change")
# #                                     if f"{prefix}severe_toxicity_change" in cols
# #                                     else mean("chain_severe_toxicity_change_avg"),
# #         "avg_old_toxicity":         mean("chain_avg_old_toxicity"),
# #         "avg_new_toxicity":         mean("chain_avg_new_toxicity"),
# #         "length_ratio":             mean(f"{prefix}length_ratio"),
# #         "bleu":                     mean(f"{prefix}bleu"),
# #         "rougeL":                   mean(f"{prefix}rougeL"),
# #         "llm_toxicity_removal":     mean(f"{prefix}llm_toxicity_removal"),
# #         "llm_meaning_preservation": mean(f"{prefix}llm_meaning_preservation"),
# #         "llm_fluency":              mean(f"{prefix}llm_fluency"),
# #         "llm_refusal":              mean(f"{prefix}llm_refusal"),
# #         "llm_overall":              mean(f"{prefix}llm_overall"),
# #     }

# # s = [stats(base), stats(ptsd), stats(llm)]

# # # ─── Helper ───────────────────────────────────────────────────────────────────
# # def bar_chart(ax, labels, values, title, ylabel, color_list, annotate_fmt="{:.4f}"):
# #     x = np.arange(len(labels))
# #     bars = ax.bar(x, values, color=color_list, width=0.55, zorder=3)
# #     ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
# #     ax.set_ylabel(ylabel, fontsize=9)
# #     ax.set_xticks(x)
# #     ax.set_xticklabels(labels, fontsize=9)
# #     ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
# #     ax.spines[["top","right"]].set_visible(False)
# #     for bar, v in zip(bars, values):
# #         if not np.isnan(v):
# #             ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(bar.get_height())*0.02,
# #                     annotate_fmt.format(v), ha="center", va="bottom", fontsize=8)

# # def group_bar(ax, group_labels, model_values_list, title, ylabel, annotate_fmt="{:.4f}"):
# #     """
# #     group_labels  : list of metric names
# #     model_values_list : list of 3 lists (one per model), each matching group_labels length
# #     """
# #     n_groups = len(group_labels)
# #     n_models = len(MODELS)
# #     width = 0.22
# #     x = np.arange(n_groups)
# #     for i, (model_name, values) in enumerate(zip(MODELS, model_values_list)):
# #         offset = (i - 1) * width
# #         bars = ax.bar(x + offset, values, width=width, color=COLORS[i],
# #                       alpha=ALPHAS[i], label=model_name.replace("\n", " "), zorder=3)
# #     ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
# #     ax.set_ylabel(ylabel, fontsize=9)
# #     ax.set_xticks(x)
# #     ax.set_xticklabels(group_labels, fontsize=9)
# #     ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
# #     ax.spines[["top","right"]].set_visible(False)
# #     ax.legend(fontsize=8, framealpha=0.6)


# # # ══════════════════════════════════════════════════════════════════════════════
# # # Figure 1 — Toxicity Reduction Overview
# # # ══════════════════════════════════════════════════════════════════════════════
# # fig1, axes = plt.subplots(1, 3, figsize=(14, 4.5))
# # fig1.suptitle("Figure 1 — Toxicity Reduction", fontsize=14, fontweight="bold", y=1.02)

# # # 1a: Before vs After toxicity
# # ax = axes[0]
# # x = np.arange(len(MODELS))
# # w = 0.35
# # bars_old = ax.bar(x - w/2, [s[i]["avg_old_toxicity"] for i in range(3)],
# #                   width=w, color="#888780", label="Before", zorder=3)
# # bars_new = ax.bar(x + w/2, [s[i]["avg_new_toxicity"] for i in range(3)],
# #                   width=w, color=COLORS, label="After", zorder=3)
# # ax.set_title("Avg toxicity before vs after", fontsize=11, fontweight="bold", pad=8)
# # ax.set_ylabel("Toxicity score", fontsize=9)
# # ax.set_xticks(x)
# # ax.set_xticklabels(MODELS, fontsize=9)
# # ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
# # ax.spines[["top","right"]].set_visible(False)
# # ax.legend(fontsize=9, framealpha=0.6)

# # # 1b: Toxicity change (negative = better)
# # bar_chart(axes[1],
# #           MODELS,
# #           [s[i]["toxicity_change"] for i in range(3)],
# #           "Avg toxicity change\n(more negative = better)",
# #           "Delta toxicity",
# #           COLORS, annotate_fmt="{:.4f}")

# # # 1c: Severe toxicity change
# # bar_chart(axes[2],
# #           MODELS,
# #           [s[i]["severe_toxicity_change"] for i in range(3)],
# #           "Avg severe toxicity change\n(more negative = better)",
# #           "Delta severe toxicity",
# #           COLORS, annotate_fmt="{:.4f}")

# # fig1.tight_layout()
# # fig1.savefig("fig1_toxicity.png", dpi=150, bbox_inches="tight")
# # print("Saved fig1_toxicity.png")


# # # ══════════════════════════════════════════════════════════════════════════════
# # # Figure 2 — Meaning Preservation
# # # ══════════════════════════════════════════════════════════════════════════════
# # fig2, axes = plt.subplots(1, 4, figsize=(16, 4.5))
# # fig2.suptitle("Figure 2 — Meaning Preservation Metrics", fontsize=14, fontweight="bold", y=1.02)

# # metrics = [
# #     ("cosine_similarity", "Cosine similarity\n(embedding space)", "Score"),
# #     ("length_ratio",      "Length ratio\n(closer to 1.0 = better)", "Ratio"),
# #     ("bleu",              "BLEU score\n(higher = better)", "Score"),
# #     ("rougeL",            "ROUGE-L\n(higher = better)", "Score"),
# # ]
# # for ax, (key, title, ylabel) in zip(axes, metrics):
# #     values = [s[i][key] for i in range(3)]
# #     bar_chart(ax, MODELS, values, title, ylabel, COLORS, annotate_fmt="{:.4f}")

# # fig2.tight_layout()
# # fig2.savefig("fig2_preservation.png", dpi=150, bbox_inches="tight")
# # print("Saved fig2_preservation.png")


# # # ══════════════════════════════════════════════════════════════════════════════
# # # Figure 3 — LLM Judge Scores (PTSD vs LLM only; Baseline has no judge scores)
# # # ══════════════════════════════════════════════════════════════════════════════
# # judge_metrics = ["Toxicity\nRemoval", "Meaning\nPreserv.", "Fluency", "Overall"]
# # judge_keys    = ["llm_toxicity_removal", "llm_meaning_preservation", "llm_fluency", "llm_overall"]

# # ptsd_vals = [s[1][k] for k in judge_keys]
# # llm_vals  = [s[2][k] for k in judge_keys]

# # fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
# # fig3.suptitle("Figure 3 — LLM Judge Scores (1–5 scale)", fontsize=14, fontweight="bold", y=1.02)

# # # 3a: Grouped bar
# # ax = axes[0]
# # x = np.arange(len(judge_metrics))
# # w = 0.35
# # ax.bar(x - w/2, ptsd_vals, width=w, color=COLORS[1], alpha=0.85, label="PTSD Model", zorder=3)
# # ax.bar(x + w/2, llm_vals,  width=w, color=COLORS[2], alpha=0.85, label="LLM Judge",  zorder=3)
# # ax.set_xticks(x); ax.set_xticklabels(judge_metrics, fontsize=9)
# # ax.set_ylim(0, 5.5); ax.set_ylabel("Score (max 5)", fontsize=9)
# # ax.set_title("LLM judge scores by category", fontsize=11, fontweight="bold", pad=8)
# # ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
# # ax.spines[["top","right"]].set_visible(False)
# # ax.legend(fontsize=9, framealpha=0.6)
# # for i, (pv, lv) in enumerate(zip(ptsd_vals, llm_vals)):
# #     ax.text(x[i]-w/2, pv+0.05, f"{pv:.2f}", ha="center", va="bottom", fontsize=8)
# #     ax.text(x[i]+w/2, lv+0.05, f"{lv:.2f}", ha="center", va="bottom", fontsize=8)

# # # 3b: Radar / spider chart
# # ax2 = axes[1]
# # categories = ["Toxicity\nRemoval", "Meaning\nPreserv.", "Fluency", "Overall", "Low\nRefusal"]
# # ptsd_radar = [s[1]["llm_toxicity_removal"], s[1]["llm_meaning_preservation"],
# #               s[1]["llm_fluency"], s[1]["llm_overall"], 5 - s[1]["llm_refusal"]*100]
# # llm_radar  = [s[2]["llm_toxicity_removal"], s[2]["llm_meaning_preservation"],
# #               s[2]["llm_fluency"], s[2]["llm_overall"], 5 - s[2]["llm_refusal"]*100]

# # n = len(categories)
# # angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
# # angles += angles[:1]
# # ptsd_radar += ptsd_radar[:1]
# # llm_radar  += llm_radar[:1]

# # ax2 = fig3.add_subplot(122, polar=True)
# # ax2.plot(angles, ptsd_radar, color=COLORS[1], linewidth=2, label="PTSD Model")
# # ax2.fill(angles, ptsd_radar, alpha=0.2, color=COLORS[1])
# # ax2.plot(angles, llm_radar,  color=COLORS[2], linewidth=2, linestyle="--", label="LLM Judge")
# # ax2.fill(angles, llm_radar,  alpha=0.2, color=COLORS[2])
# # ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(categories, fontsize=9)
# # ax2.set_ylim(0, 5); ax2.set_yticks([1,2,3,4,5]); ax2.set_yticklabels(["1","2","3","4","5"], fontsize=7)
# # ax2.set_title("Radar: judge dimensions", fontsize=11, fontweight="bold", pad=14)
# # ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

# # fig3.tight_layout()
# # fig3.savefig("fig3_judge_scores.png", dpi=150, bbox_inches="tight")
# # print("Saved fig3_judge_scores.png")


# # # ══════════════════════════════════════════════════════════════════════════════
# # # Figure 4 — Summary dashboard (all key metrics side by side)
# # # ══════════════════════════════════════════════════════════════════════════════
# # fig4, axes = plt.subplots(2, 3, figsize=(15, 9))
# # fig4.suptitle("Figure 4 — Full Model Comparison Dashboard", fontsize=15, fontweight="bold", y=1.01)

# # specs = [
# #     # (key, title, ylabel, annotate_fmt, lower_is_better_note)
# #     ("cosine_similarity",        "Cosine similarity",           "Score",         "{:.4f}"),
# #     ("toxicity_change",          "Toxicity change (Δ)",         "Delta",         "{:.4f}"),
# #     ("severe_toxicity_change",   "Severe toxicity change (Δ)",  "Delta",         "{:.4f}"),
# #     ("bleu",                     "BLEU score",                  "Score",         "{:.4f}"),
# #     ("rougeL",                   "ROUGE-L",                     "Score",         "{:.4f}"),
# #     ("avg_new_toxicity",         "Post-detox toxicity",         "Score",         "{:.4f}"),
# # ]

# # for ax, (key, title, ylabel, fmt) in zip(axes.flat, specs):
# #     values = [s[i][key] for i in range(3)]
# #     bar_chart(ax, MODELS, values, title, ylabel, COLORS, annotate_fmt=fmt)

# # fig4.tight_layout()
# # fig4.savefig("fig4_dashboard.png", dpi=150, bbox_inches="tight")
# # print("Saved fig4_dashboard.png")


# # # ══════════════════════════════════════════════════════════════════════════════
# # # Figure 5 — Full Metrics Table (all metrics, all models)
# # # ══════════════════════════════════════════════════════════════════════════════

# # MODEL_LABELS = ["T5 Detoxifier (Baseline)", "PTSD Model (Fine-tuned)", "LLM Judge (Prompted)"]

# # # Define all rows: (display label, stat key, format, lower_is_better)
# # ALL_ROWS = [
# #     # --- Toxicity Reduction ---
# #     ("── Toxicity Reduction ──",          None,                        None,     None),
# #     ("Avg toxicity before",               "avg_old_toxicity",          ".4f",    True),
# #     ("Avg toxicity after",                "avg_new_toxicity",          ".4f",    True),
# #     ("Toxicity change (Δ)",               "toxicity_change",           ".4f",    True),
# #     ("Severe toxicity change (Δ)",        "severe_toxicity_change",    ".4f",    True),
# #     # --- Meaning Preservation ---
# #     ("── Meaning Preservation ──",        None,                        None,     None),
# #     ("Cosine similarity",                 "cosine_similarity",         ".4f",    False),
# #     ("Length ratio",                      "length_ratio",              ".4f",    None),   # closer to 1
# #     ("BLEU score",                        "bleu",                      ".4f",    False),
# #     ("ROUGE-L",                           "rougeL",                    ".4f",    False),
# #     # --- LLM Judge (1–5) ---
# #     ("── LLM Judge Scores (1–5) ──",      None,                        None,     None),
# #     ("Toxicity removal",                  "llm_toxicity_removal",      ".2f",    False),
# #     ("Meaning preservation",              "llm_meaning_preservation",  ".2f",    False),
# #     ("Fluency",                           "llm_fluency",               ".2f",    False),
# #     ("Overall",                           "llm_overall",               ".2f",    False),
# #     ("Refusal rate",                      "llm_refusal",               ".4f",    True),
# # ]

# # BEST_COLOR  = "#d4edda"   # light green
# # WORST_COLOR = "#f8d7da"   # light red
# # HEAD_COLOR  = "#dce8f5"   # light blue — column headers
# # CAT_COLOR   = "#f0f0f0"   # light gray — category separator rows
# # NA_STR      = "—"

# # def fmt_val(v, fmt):
# #     if v is None or (isinstance(v, float) and np.isnan(v)):
# #         return NA_STR
# #     return f"{v:{fmt}}"

# # def best_worst_indices(vals, lower_is_better):
# #     """Return (best_idx, worst_idx) among non-NaN values; None if tie or N/A."""
# #     valid = [(i, v) for i, v in enumerate(vals)
# #              if v is not None and not (isinstance(v, float) and np.isnan(v))]
# #     if len(valid) < 2:
# #         return None, None
# #     if lower_is_better is None:          # e.g. length ratio — highlight closest to 1
# #         valid_sorted = sorted(valid, key=lambda x: abs(x[1] - 1.0))
# #     elif lower_is_better:
# #         valid_sorted = sorted(valid, key=lambda x: x[1])
# #     else:
# #         valid_sorted = sorted(valid, key=lambda x: x[1], reverse=True)
# #     best  = valid_sorted[0][0]
# #     worst = valid_sorted[-1][0]
# #     return (best, worst) if best != worst else (None, None)

# # # Build table data
# # col_headers = ["Metric"] + MODEL_LABELS
# # table_data  = []
# # cell_colors = []

# # for label, key, fmt, lib in ALL_ROWS:
# #     if key is None:
# #         # Category separator
# #         row   = [label, "", "", ""]
# #         color = [CAT_COLOR] * 4
# #     else:
# #         vals = [s[i].get(key, np.nan) for i in range(3)]
# #         best_i, worst_i = best_worst_indices(vals, lib)
# #         row   = [label] + [fmt_val(v, fmt) for v in vals]
# #         color = ["white"]
# #         for i, v in enumerate(vals):
# #             if fmt_val(v, fmt) == NA_STR:
# #                 color.append("#fafafa")
# #             elif i == best_i:
# #                 color.append(BEST_COLOR)
# #             elif i == worst_i:
# #                 color.append(WORST_COLOR)
# #             else:
# #                 color.append("white")
# #     table_data.append(row)
# #     cell_colors.append(color)

# # n_rows = len(table_data)
# # fig5, ax5 = plt.subplots(figsize=(13, n_rows * 0.42 + 1.2))
# # ax5.axis("off")
# # fig5.suptitle("Figure 5 — Full Metrics Comparison Table", fontsize=14, fontweight="bold", y=0.98)

# # tbl = ax5.table(
# #     cellText=table_data,
# #     colLabels=col_headers,
# #     cellColours=cell_colors,
# #     cellLoc="center",
# #     loc="center",
# # )
# # tbl.auto_set_font_size(False)
# # tbl.set_fontsize(9)
# # tbl.scale(1, 1.35)

# # # Style header row
# # for j in range(len(col_headers)):
# #     tbl[0, j].set_facecolor(HEAD_COLOR)
# #     tbl[0, j].set_text_props(fontweight="bold", fontsize=9)

# # # Left-align metric name column; bold category rows
# # for i, (label, key, *_) in enumerate(ALL_ROWS):
# #     tbl[i + 1, 0].set_text_props(ha="left",
# #                                   fontweight="bold" if key is None else "normal",
# #                                   color="#333333" if key is None else "black")
# #     if key is None:
# #         for j in range(len(col_headers)):
# #             tbl[i + 1, j].set_facecolor(CAT_COLOR)
# #             tbl[i + 1, j].set_text_props(fontweight="bold", color="#444444")

# # # Legend
# # legend_elements = [
# #     mpatches.Patch(facecolor=BEST_COLOR,  edgecolor="#aaa", label="Best value"),
# #     mpatches.Patch(facecolor=WORST_COLOR, edgecolor="#aaa", label="Worst value"),
# #     mpatches.Patch(facecolor="#fafafa",   edgecolor="#aaa", label="N/A (not computed for this model)"),
# # ]
# # fig5.legend(handles=legend_elements, loc="lower center", ncol=3,
# #             fontsize=8, framealpha=0.7, bbox_to_anchor=(0.5, 0.01))

# # fig5.tight_layout(rect=[0, 0.04, 1, 0.97])
# # fig5.savefig("fig5_full_table.png", dpi=150, bbox_inches="tight")
# # print("Saved fig5_full_table.png")


# # # ══════════════════════════════════════════════════════════════════════════════
# # # Figure 6 — Side-by-side category tables (toxicity / preservation / judge)
# # # ══════════════════════════════════════════════════════════════════════════════

# # SECTION_DEFS = [
# #     {
# #         "title": "Toxicity Reduction",
# #         "rows": [
# #             ("Avg toxicity before",    "avg_old_toxicity",       ".4f", True),
# #             ("Avg toxicity after",     "avg_new_toxicity",       ".4f", True),
# #             ("Toxicity change (Δ)",    "toxicity_change",        ".4f", True),
# #             ("Severe tox. change (Δ)", "severe_toxicity_change", ".4f", True),
# #         ],
# #     },
# #     {
# #         "title": "Meaning Preservation",
# #         "rows": [
# #             ("Cosine similarity",  "cosine_similarity", ".4f", False),
# #             ("Length ratio",       "length_ratio",      ".4f", None),
# #             ("BLEU score",         "bleu",              ".4f", False),
# #             ("ROUGE-L",            "rougeL",            ".4f", False),
# #         ],
# #     },
# #     {
# #         "title": "LLM Judge Scores (1–5)",
# #         "rows": [
# #             ("Toxicity removal",      "llm_toxicity_removal",      ".2f", False),
# #             ("Meaning preservation",  "llm_meaning_preservation",  ".2f", False),
# #             ("Fluency",               "llm_fluency",               ".2f", False),
# #             ("Overall",               "llm_overall",               ".2f", False),
# #             ("Refusal rate",          "llm_refusal",               ".4f", True),
# #         ],
# #     },
# # ]

# # fig6, axes6 = plt.subplots(1, 3, figsize=(18, 5))
# # fig6.suptitle("Figure 6 — Metrics by Category", fontsize=14, fontweight="bold", y=1.01)

# # short_model = ["T5 Baseline", "PTSD Model", "LLM Model"]

# # for ax, section in zip(axes6, SECTION_DEFS):
# #     ax.axis("off")
# #     ax.set_title(section["title"], fontsize=11, fontweight="bold", pad=10)

# #     t_data   = []
# #     t_colors = []
# #     for label, key, fmt, lib in section["rows"]:
# #         vals = [s[i].get(key, np.nan) for i in range(3)]
# #         best_i, worst_i = best_worst_indices(vals, lib)
# #         row   = [label] + [fmt_val(v, fmt) for v in vals]
# #         color = ["white"]
# #         for i, v in enumerate(vals):
# #             if fmt_val(v, fmt) == NA_STR:
# #                 color.append("#fafafa")
# #             elif i == best_i:
# #                 color.append(BEST_COLOR)
# #             elif i == worst_i:
# #                 color.append(WORST_COLOR)
# #             else:
# #                 color.append("white")
# #         t_data.append(row)
# #         t_colors.append(color)

# #     col_lbl = ["Metric"] + short_model
# #     tbl6 = ax.table(
# #         cellText=t_data,
# #         colLabels=col_lbl,
# #         cellColours=t_colors,
# #         cellLoc="center",
# #         loc="center",
# #     )
# #     tbl6.auto_set_font_size(False)
# #     tbl6.set_fontsize(9)
# #     tbl6.scale(1, 1.6)

# #     for j in range(len(col_lbl)):
# #         tbl6[0, j].set_facecolor(HEAD_COLOR)
# #         tbl6[0, j].set_text_props(fontweight="bold", fontsize=9)
# #     for i in range(len(section["rows"])):
# #         tbl6[i + 1, 0].set_text_props(ha="left")

# # legend_elements6 = [
# #     mpatches.Patch(facecolor=BEST_COLOR,  edgecolor="#aaa", label="Best"),
# #     mpatches.Patch(facecolor=WORST_COLOR, edgecolor="#aaa", label="Worst"),
# #     mpatches.Patch(facecolor="#fafafa",   edgecolor="#aaa", label="N/A"),
# # ]
# # fig6.legend(handles=legend_elements6, loc="lower center", ncol=3,
# #             fontsize=8, framealpha=0.7, bbox_to_anchor=(0.5, -0.02))

# # fig6.tight_layout()
# # fig6.savefig("fig6_category_tables.png", dpi=150, bbox_inches="tight")
# # print("Saved fig6_category_tables.png")


# # # ══════════════════════════════════════════════════════════════════════════════
# # # Figure 7 — Heatmap-style comparison table
# # # ══════════════════════════════════════════════════════════════════════════════
# # import matplotlib.colors as mcolors

# # # Metrics to include in heatmap (only numeric, no category rows)
# # HEATMAP_ROWS = [
# #     ("Avg tox. before",       "avg_old_toxicity",          True),
# #     ("Avg tox. after",        "avg_new_toxicity",           True),
# #     ("Toxicity Δ",            "toxicity_change",            True),
# #     ("Severe tox. Δ",         "severe_toxicity_change",     True),
# #     ("Cosine similarity",     "cosine_similarity",          False),
# #     ("Length ratio",          "length_ratio",               None),
# #     ("BLEU",                  "bleu",                       False),
# #     ("ROUGE-L",               "rougeL",                     False),
# #     ("Judge: tox. removal",   "llm_toxicity_removal",       False),
# #     ("Judge: meaning pres.",  "llm_meaning_preservation",   False),
# #     ("Judge: fluency",        "llm_fluency",                False),
# #     ("Judge: overall",        "llm_overall",                False),
# #     ("Judge: refusal",        "llm_refusal",                True),
# # ]

# # hm_labels  = [r[0] for r in HEATMAP_ROWS]
# # hm_matrix  = np.array([[s[i].get(r[1], np.nan) for i in range(3)] for r in HEATMAP_ROWS])

# # # Normalise each row 0→1 so the heatmap shows relative ranking
# # # For "lower is better" metrics, invert so green always = better
# # hm_norm = np.full_like(hm_matrix, np.nan)
# # for ri, (_, _, lib) in enumerate(HEATMAP_ROWS):
# #     row = hm_matrix[ri]
# #     valid_mask = ~np.isnan(row)
# #     if valid_mask.sum() < 2:
# #         hm_norm[ri] = row
# #         continue
# #     vmin, vmax = np.nanmin(row), np.nanmax(row)
# #     if vmax == vmin:
# #         hm_norm[ri, valid_mask] = 0.5
# #     else:
# #         normed = (row - vmin) / (vmax - vmin)
# #         if lib is True:      # lower is better → invert so green = lower
# #             normed = 1 - normed
# #         elif lib is None:    # closer to 1.0 = better
# #             normed = 1 - np.abs(row - 1.0) / max(np.nanmax(np.abs(hm_matrix[ri] - 1.0)), 1e-9)
# #         hm_norm[ri] = normed

# # fig7, ax7 = plt.subplots(figsize=(9, len(HEATMAP_ROWS) * 0.52 + 1.5))
# # ax7.set_title("Figure 7 — Heatmap: Relative Performance\n(green = best, red = worst per row)",
# #               fontsize=12, fontweight="bold", pad=10)

# # cmap = plt.cm.RdYlGn
# # im = ax7.imshow(hm_norm, aspect="auto", cmap=cmap, vmin=0, vmax=1)

# # ax7.set_xticks(range(3))
# # ax7.set_xticklabels(short_model, fontsize=10, fontweight="bold")
# # ax7.set_yticks(range(len(hm_labels)))
# # ax7.set_yticklabels(hm_labels, fontsize=9)
# # ax7.xaxis.tick_top()
# # ax7.xaxis.set_label_position("top")

# # # Annotate each cell with the raw value
# # for ri in range(len(HEATMAP_ROWS)):
# #     _, key, _ = HEATMAP_ROWS[ri]
# #     fmt_str = ".2f" if "llm_" in key else ".4f"
# #     for ci in range(3):
# #         raw = hm_matrix[ri, ci]
# #         txt = NA_STR if np.isnan(raw) else f"{raw:{fmt_str}}"
# #         norm_v = hm_norm[ri, ci]
# #         text_color = "black" if (np.isnan(norm_v) or 0.25 < norm_v < 0.75) else "white"
# #         ax7.text(ci, ri, txt, ha="center", va="center", fontsize=8, color=text_color)

# # plt.colorbar(im, ax=ax7, orientation="vertical", label="Relative score (0=worst, 1=best)",
# #              shrink=0.6, pad=0.02)

# # fig7.tight_layout()
# # fig7.savefig("fig7_heatmap_table.png", dpi=150, bbox_inches="tight")
# # print("Saved fig7_heatmap_table.png")


# # plt.show()
# # print("\nAll figures saved successfully.")
# # print("Output files: fig1_toxicity.png, fig2_preservation.png, fig3_judge_scores.png,")
# # print("              fig4_dashboard.png, fig5_full_table.png, fig6_category_tables.png,")
# # print("              fig7_heatmap_table.png")
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.gridspec import GridSpec

# # ─── Load data ────────────────────────────────────────────────────────────────
# base = pd.read_parquet("reddit_detoxified.parquet")
# ptsd = pd.read_parquet("PTSD_reddit_detoxified.parquet")
# llm  = pd.read_parquet("LLM_reddit_detoxified.parquet")

# MODELS   = ["T5 Detoxifier\n(Baseline)", "PTSD Model\n(Fine-tuned)", "LLM Judge\n(Prompted)"]
# COLORS   = ["#378ADD", "#1D9E75", "#D85A30"]   # blue, teal, coral
# ALPHAS   = [0.85, 0.85, 0.85]

# # ─── Compute per-model mean stats ─────────────────────────────────────────────
# def stats(df, prefix="avg_"):
#     """Return a dict of mean values, handling both column name conventions."""
#     cols = list(df.columns)
#     def mean(col):
#         return df[col].mean() if col in cols else np.nan

#     return {
#         "cosine_similarity":        mean(f"{prefix}cosine_similarity")
#                                     if f"{prefix}cosine_similarity" in cols
#                                     else mean("chain_cosine_similarity_avg"),
#         "toxicity_change":          mean(f"{prefix}toxicity_change")
#                                     if f"{prefix}toxicity_change" in cols
#                                     else mean("chain_toxicity_change_avg"),
#         "severe_toxicity_change":   mean(f"{prefix}severe_toxicity_change")
#                                     if f"{prefix}severe_toxicity_change" in cols
#                                     else mean("chain_severe_toxicity_change_avg"),
#         "avg_old_toxicity":         mean("chain_avg_old_toxicity"),
#         "avg_new_toxicity":         mean("chain_avg_new_toxicity"),
#         "length_ratio":             mean(f"{prefix}length_ratio"),
#         "bleu":                     mean(f"{prefix}bleu"),
#         "rougeL":                   mean(f"{prefix}rougeL"),
#         "llm_toxicity_removal":     mean(f"{prefix}llm_toxicity_removal"),
#         "llm_meaning_preservation": mean(f"{prefix}llm_meaning_preservation"),
#         "llm_fluency":              mean(f"{prefix}llm_fluency"),
#         "llm_refusal":              mean(f"{prefix}llm_refusal"),
#         "llm_overall":              mean(f"{prefix}llm_overall"),
#     }

# s = [stats(base), stats(ptsd), stats(llm)]

# # ─── Helper ───────────────────────────────────────────────────────────────────
# def bar_chart(ax, labels, values, title, ylabel, color_list, annotate_fmt="{:.4f}"):
#     x = np.arange(len(labels))
#     bars = ax.bar(x, values, color=color_list, width=0.55, zorder=3)
#     ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
#     ax.set_ylabel(ylabel, fontsize=9)
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, fontsize=9)
#     ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
#     ax.spines[["top","right"]].set_visible(False)
#     for bar, v in zip(bars, values):
#         if not np.isnan(v):
#             ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(bar.get_height())*0.02,
#                     annotate_fmt.format(v), ha="center", va="bottom", fontsize=8)

# def group_bar(ax, group_labels, model_values_list, title, ylabel, annotate_fmt="{:.4f}"):
#     """
#     group_labels  : list of metric names
#     model_values_list : list of 3 lists (one per model), each matching group_labels length
#     """
#     n_groups = len(group_labels)
#     n_models = len(MODELS)
#     width = 0.22
#     x = np.arange(n_groups)
#     for i, (model_name, values) in enumerate(zip(MODELS, model_values_list)):
#         offset = (i - 1) * width
#         bars = ax.bar(x + offset, values, width=width, color=COLORS[i],
#                       alpha=ALPHAS[i], label=model_name.replace("\n", " "), zorder=3)
#     ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
#     ax.set_ylabel(ylabel, fontsize=9)
#     ax.set_xticks(x)
#     ax.set_xticklabels(group_labels, fontsize=9)
#     ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
#     ax.spines[["top","right"]].set_visible(False)
#     ax.legend(fontsize=8, framealpha=0.6)


# # ══════════════════════════════════════════════════════════════════════════════
# # Figure 1 — Toxicity Reduction Overview
# # ══════════════════════════════════════════════════════════════════════════════
# fig1, axes = plt.subplots(1, 3, figsize=(14, 4.5))
# fig1.suptitle("Figure 1 — Toxicity Reduction", fontsize=14, fontweight="bold", y=1.02)

# # 1a: Before vs After toxicity
# ax = axes[0]
# x = np.arange(len(MODELS))
# w = 0.35
# bars_old = ax.bar(x - w/2, [s[i]["avg_old_toxicity"] for i in range(3)],
#                   width=w, color="#888780", label="Before", zorder=3)
# bars_new = ax.bar(x + w/2, [s[i]["avg_new_toxicity"] for i in range(3)],
#                   width=w, color=COLORS, label="After", zorder=3)
# ax.set_title("Avg toxicity before vs after", fontsize=11, fontweight="bold", pad=8)
# ax.set_ylabel("Toxicity score", fontsize=9)
# ax.set_xticks(x)
# ax.set_xticklabels(MODELS, fontsize=9)
# ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
# ax.spines[["top","right"]].set_visible(False)
# ax.legend(fontsize=9, framealpha=0.6)

# # 1b: Toxicity change (negative = better)
# bar_chart(axes[1],
#           MODELS,
#           [s[i]["toxicity_change"] for i in range(3)],
#           "Avg toxicity change\n(more negative = better)",
#           "Delta toxicity",
#           COLORS, annotate_fmt="{:.4f}")

# # 1c: Severe toxicity change
# bar_chart(axes[2],
#           MODELS,
#           [s[i]["severe_toxicity_change"] for i in range(3)],
#           "Avg severe toxicity change\n(more negative = better)",
#           "Delta severe toxicity",
#           COLORS, annotate_fmt="{:.4f}")

# fig1.tight_layout()
# fig1.savefig("fig1_toxicity.png", dpi=150, bbox_inches="tight")
# print("Saved fig1_toxicity.png")


# # ══════════════════════════════════════════════════════════════════════════════
# # Figure 2 — Meaning Preservation
# # ══════════════════════════════════════════════════════════════════════════════
# fig2, axes = plt.subplots(1, 4, figsize=(16, 4.5))
# fig2.suptitle("Figure 2 — Meaning Preservation Metrics", fontsize=14, fontweight="bold", y=1.02)

# metrics = [
#     ("cosine_similarity", "Cosine similarity\n(embedding space)", "Score"),
#     ("length_ratio",      "Length ratio\n(closer to 1.0 = better)", "Ratio"),
#     ("bleu",              "BLEU score\n(higher = better)", "Score"),
#     ("rougeL",            "ROUGE-L\n(higher = better)", "Score"),
# ]
# for ax, (key, title, ylabel) in zip(axes, metrics):
#     values = [s[i][key] for i in range(3)]
#     bar_chart(ax, MODELS, values, title, ylabel, COLORS, annotate_fmt="{:.4f}")

# fig2.tight_layout()
# fig2.savefig("fig2_preservation.png", dpi=150, bbox_inches="tight")
# print("Saved fig2_preservation.png")


# # ══════════════════════════════════════════════════════════════════════════════
# # Figure 3 — LLM Judge Scores (PTSD vs LLM only; Baseline has no judge scores)
# # ══════════════════════════════════════════════════════════════════════════════
# judge_metrics = ["Toxicity\nRemoval", "Meaning\nPreserv.", "Fluency", "Overall"]
# judge_keys    = ["llm_toxicity_removal", "llm_meaning_preservation", "llm_fluency", "llm_overall"]

# ptsd_vals = [s[1][k] for k in judge_keys]
# llm_vals  = [s[2][k] for k in judge_keys]

# fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
# fig3.suptitle("Figure 3 — LLM Judge Scores (1–5 scale)", fontsize=14, fontweight="bold", y=1.02)

# # 3a: Grouped bar
# ax = axes[0]
# x = np.arange(len(judge_metrics))
# w = 0.35
# ax.bar(x - w/2, ptsd_vals, width=w, color=COLORS[1], alpha=0.85, label="PTSD Model", zorder=3)
# ax.bar(x + w/2, llm_vals,  width=w, color=COLORS[2], alpha=0.85, label="LLM Judge",  zorder=3)
# ax.set_xticks(x); ax.set_xticklabels(judge_metrics, fontsize=9)
# ax.set_ylim(0, 5.5); ax.set_ylabel("Score (max 5)", fontsize=9)
# ax.set_title("LLM judge scores by category", fontsize=11, fontweight="bold", pad=8)
# ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
# ax.spines[["top","right"]].set_visible(False)
# ax.legend(fontsize=9, framealpha=0.6)
# for i, (pv, lv) in enumerate(zip(ptsd_vals, llm_vals)):
#     ax.text(x[i]-w/2, pv+0.05, f"{pv:.2f}", ha="center", va="bottom", fontsize=8)
#     ax.text(x[i]+w/2, lv+0.05, f"{lv:.2f}", ha="center", va="bottom", fontsize=8)

# # 3b: Radar / spider chart
# ax2 = axes[1]
# categories = ["Toxicity\nRemoval", "Meaning\nPreserv.", "Fluency", "Overall", "Low\nRefusal"]
# ptsd_radar = [s[1]["llm_toxicity_removal"], s[1]["llm_meaning_preservation"],
#               s[1]["llm_fluency"], s[1]["llm_overall"], 5 - s[1]["llm_refusal"]*100]
# llm_radar  = [s[2]["llm_toxicity_removal"], s[2]["llm_meaning_preservation"],
#               s[2]["llm_fluency"], s[2]["llm_overall"], 5 - s[2]["llm_refusal"]*100]

# n = len(categories)
# angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
# angles += angles[:1]
# ptsd_radar += ptsd_radar[:1]
# llm_radar  += llm_radar[:1]

# ax2 = fig3.add_subplot(122, polar=True)
# ax2.plot(angles, ptsd_radar, color=COLORS[1], linewidth=2, label="PTSD Model")
# ax2.fill(angles, ptsd_radar, alpha=0.2, color=COLORS[1])
# ax2.plot(angles, llm_radar,  color=COLORS[2], linewidth=2, linestyle="--", label="LLM Judge")
# ax2.fill(angles, llm_radar,  alpha=0.2, color=COLORS[2])
# ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(categories, fontsize=9)
# ax2.set_ylim(0, 5); ax2.set_yticks([1,2,3,4,5]); ax2.set_yticklabels(["1","2","3","4","5"], fontsize=7)
# ax2.set_title("Radar: judge dimensions", fontsize=11, fontweight="bold", pad=14)
# ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

# fig3.tight_layout()
# fig3.savefig("fig3_judge_scores.png", dpi=150, bbox_inches="tight")
# print("Saved fig3_judge_scores.png")


# # ══════════════════════════════════════════════════════════════════════════════
# # Figure 4 — Summary dashboard (all key metrics side by side)
# # ══════════════════════════════════════════════════════════════════════════════
# fig4, axes = plt.subplots(2, 3, figsize=(15, 9))
# fig4.suptitle("Figure 4 — Full Model Comparison Dashboard", fontsize=15, fontweight="bold", y=1.01)

# specs = [
#     # (key, title, ylabel, annotate_fmt, lower_is_better_note)
#     ("cosine_similarity",        "Cosine similarity",           "Score",         "{:.4f}"),
#     ("toxicity_change",          "Toxicity change (Δ)",         "Delta",         "{:.4f}"),
#     ("severe_toxicity_change",   "Severe toxicity change (Δ)",  "Delta",         "{:.4f}"),
#     ("bleu",                     "BLEU score",                  "Score",         "{:.4f}"),
#     ("rougeL",                   "ROUGE-L",                     "Score",         "{:.4f}"),
#     ("avg_new_toxicity",         "Post-detox toxicity",         "Score",         "{:.4f}"),
# ]

# for ax, (key, title, ylabel, fmt) in zip(axes.flat, specs):
#     values = [s[i][key] for i in range(3)]
#     bar_chart(ax, MODELS, values, title, ylabel, COLORS, annotate_fmt=fmt)

# fig4.tight_layout()
# fig4.savefig("fig4_dashboard.png", dpi=150, bbox_inches="tight")
# print("Saved fig4_dashboard.png")


# # ══════════════════════════════════════════════════════════════════════════════
# # Figure 5 — Full Metrics Table (all metrics, all models)
# # ══════════════════════════════════════════════════════════════════════════════

# MODEL_LABELS = ["T5 Detoxifier (Baseline)", "PTSD Model (Fine-tuned)", "LLM Judge (Prompted)"]

# # Define all rows: (display label, stat key, format, lower_is_better)
# ALL_ROWS = [
#     # --- Toxicity Reduction ---
#     ("── Toxicity Reduction ──",          None,                        None,     None),
#     ("Avg toxicity before",               "avg_old_toxicity",          ".4f",    True),
#     ("Avg toxicity after",                "avg_new_toxicity",          ".4f",    True),
#     ("Toxicity change (Δ)",               "toxicity_change",           ".4f",    True),
#     ("Severe toxicity change (Δ)",        "severe_toxicity_change",    ".4f",    True),
#     # --- Meaning Preservation ---
#     ("── Meaning Preservation ──",        None,                        None,     None),
#     ("Cosine similarity",                 "cosine_similarity",         ".4f",    False),
#     ("Length ratio",                      "length_ratio",              ".4f",    None),   # closer to 1
#     ("BLEU score",                        "bleu",                      ".4f",    False),
#     ("ROUGE-L",                           "rougeL",                    ".4f",    False),
#     # --- LLM Judge (1–5) ---
#     ("── LLM Judge Scores (1–5) ──",      None,                        None,     None),
#     ("Toxicity removal",                  "llm_toxicity_removal",      ".2f",    False),
#     ("Meaning preservation",              "llm_meaning_preservation",  ".2f",    False),
#     ("Fluency",                           "llm_fluency",               ".2f",    False),
#     ("Overall",                           "llm_overall",               ".2f",    False),
#     ("Refusal rate",                      "llm_refusal",               ".4f",    True),
# ]

# BEST_COLOR  = "#d4edda"   # light green
# WORST_COLOR = "#f8d7da"   # light red
# HEAD_COLOR  = "#dce8f5"   # light blue — column headers
# CAT_COLOR   = "#f0f0f0"   # light gray — category separator rows
# NA_STR      = "—"

# def fmt_val(v, fmt):
#     if v is None or (isinstance(v, float) and np.isnan(v)):
#         return NA_STR
#     return f"{v:{fmt}}"

# MIN_REL_DIFF = 0.005   # suppress highlight if best/worst differ by less than 0.5%

# def best_worst_indices(vals, lower_is_better):
#     """Return (best_idx, worst_idx) among non-NaN values.
#     Returns (None, None) when values are identical or differ by < MIN_REL_DIFF."""
#     valid = [(i, v) for i, v in enumerate(vals)
#              if v is not None and not (isinstance(v, float) and np.isnan(v))]
#     if len(valid) < 2:
#         return None, None
#     if lower_is_better is None:
#         valid_sorted = sorted(valid, key=lambda x: abs(x[1] - 1.0))
#     elif lower_is_better:
#         valid_sorted = sorted(valid, key=lambda x: x[1])
#     else:
#         valid_sorted = sorted(valid, key=lambda x: x[1], reverse=True)
#     best_i  = valid_sorted[0][0]
#     worst_i = valid_sorted[-1][0]
#     if best_i == worst_i:
#         return None, None
#     best_v, worst_v = valid_sorted[0][1], valid_sorted[-1][1]
#     denom = max(abs(best_v), abs(worst_v), 1e-9)
#     if abs(best_v - worst_v) / denom < MIN_REL_DIFF:
#         return None, None
#     return best_i, worst_i

# # Build table data
# col_headers = ["Metric"] + MODEL_LABELS
# table_data  = []
# cell_colors = []

# for label, key, fmt, lib in ALL_ROWS:
#     if key is None:
#         # Category separator
#         row   = [label, "", "", ""]
#         color = [CAT_COLOR] * 4
#     else:
#         vals = [s[i].get(key, np.nan) for i in range(3)]
#         best_i, worst_i = best_worst_indices(vals, lib)
#         row   = [label] + [fmt_val(v, fmt) for v in vals]
#         color = ["white"]
#         for i, v in enumerate(vals):
#             if fmt_val(v, fmt) == NA_STR:
#                 color.append("#fafafa")
#             elif i == best_i:
#                 color.append(BEST_COLOR)
#             elif i == worst_i:
#                 color.append(WORST_COLOR)
#             else:
#                 color.append("white")
#     table_data.append(row)
#     cell_colors.append(color)

# n_rows = len(table_data)
# fig5, ax5 = plt.subplots(figsize=(13, n_rows * 0.42 + 1.2))
# ax5.axis("off")
# fig5.suptitle("Figure 5 — Full Metrics Comparison Table", fontsize=14, fontweight="bold", y=0.98)

# tbl = ax5.table(
#     cellText=table_data,
#     colLabels=col_headers,
#     cellColours=cell_colors,
#     cellLoc="center",
#     loc="center",
# )
# tbl.auto_set_font_size(False)
# tbl.set_fontsize(9)
# tbl.scale(1, 1.35)

# # Style header row
# for j in range(len(col_headers)):
#     tbl[0, j].set_facecolor(HEAD_COLOR)
#     tbl[0, j].set_text_props(fontweight="bold", fontsize=9)

# # Left-align metric name column; bold category rows
# for i, (label, key, *_) in enumerate(ALL_ROWS):
#     tbl[i + 1, 0].set_text_props(ha="left",
#                                   fontweight="bold" if key is None else "normal",
#                                   color="#333333" if key is None else "black")
#     if key is None:
#         for j in range(len(col_headers)):
#             tbl[i + 1, j].set_facecolor(CAT_COLOR)
#             tbl[i + 1, j].set_text_props(fontweight="bold", color="#444444")

# # Legend
# legend_elements = [
#     mpatches.Patch(facecolor=BEST_COLOR,  edgecolor="#aaa", label="Best value"),
#     mpatches.Patch(facecolor=WORST_COLOR, edgecolor="#aaa", label="Worst value"),
#     mpatches.Patch(facecolor="#fafafa",   edgecolor="#aaa", label="N/A (not computed for this model)"),
# ]
# fig5.legend(handles=legend_elements, loc="lower center", ncol=3,
#             fontsize=8, framealpha=0.7, bbox_to_anchor=(0.5, 0.01))

# fig5.tight_layout(rect=[0, 0.04, 1, 0.97])
# fig5.savefig("fig5_full_table.png", dpi=150, bbox_inches="tight")
# print("Saved fig5_full_table.png")


# # ══════════════════════════════════════════════════════════════════════════════
# # Figure 6 — Side-by-side category tables (plain, no color highlighting)
# # Each category gets its own clean table for easy reading.
# # ══════════════════════════════════════════════════════════════════════════════

# SECTION_DEFS = [
#     {
#         "title": "Toxicity Reduction",
#         "rows": [
#             ("Avg toxicity before",    "avg_old_toxicity",       ".4f"),
#             ("Avg toxicity after",     "avg_new_toxicity",       ".4f"),
#             ("Toxicity change (Δ)",    "toxicity_change",        ".4f"),
#             ("Severe tox. change (Δ)", "severe_toxicity_change", ".4f"),
#         ],
#         "note": "↓ lower is better for all metrics in this section",
#     },
#     {
#         "title": "Meaning Preservation",
#         "rows": [
#             ("Cosine similarity",  "cosine_similarity", ".4f"),
#             ("Length ratio",       "length_ratio",      ".4f"),
#             ("BLEU score",         "bleu",              ".4f"),
#             ("ROUGE-L",            "rougeL",            ".4f"),
#         ],
#         "note": "↑ higher is better  |  length ratio: closer to 1.0 is better",
#     },
#     {
#         "title": "LLM Judge Scores (1–5)",
#         "rows": [
#             ("Toxicity removal",      "llm_toxicity_removal",      ".2f"),
#             ("Meaning preservation",  "llm_meaning_preservation",  ".2f"),
#             ("Fluency",               "llm_fluency",               ".2f"),
#             ("Overall",               "llm_overall",               ".2f"),
#             ("Refusal rate",          "llm_refusal",               ".4f"),
#         ],
#         "note": "↑ higher is better  |  refusal rate: ↓ lower is better",
#     },
# ]

# short_model = ["T5 Baseline", "PTSD Model", "LLM Model"]
# PLAIN_WHITE = "white"
# PLAIN_HEAD  = HEAD_COLOR   # keep the same blue header as fig5

# fig6, axes6 = plt.subplots(1, 3, figsize=(18, 5.5))
# fig6.suptitle("Figure 6 — Metrics by Category", fontsize=14, fontweight="bold", y=1.02)

# for ax, section in zip(axes6, SECTION_DEFS):
#     ax.axis("off")
#     ax.set_title(section["title"], fontsize=11, fontweight="bold", pad=6)

#     t_data   = []
#     t_colors = []
#     for label, key, fmt in section["rows"]:
#         vals = [s[i].get(key, np.nan) for i in range(3)]
#         row   = [label] + [fmt_val(v, fmt) for v in vals]
#         # All data cells plain white; N/A gets a very light gray
#         color = ["white"] + [
#             "#f5f5f5" if fmt_val(v, fmt) == NA_STR else PLAIN_WHITE
#             for v in vals
#         ]
#         t_data.append(row)
#         t_colors.append(color)

#     col_lbl = ["Metric"] + short_model
#     tbl6 = ax.table(
#         cellText=t_data,
#         colLabels=col_lbl,
#         cellColours=t_colors,
#         cellLoc="center",
#         loc="center",
#     )
#     tbl6.auto_set_font_size(False)
#     tbl6.set_fontsize(9)
#     tbl6.scale(1, 1.7)

#     # Style header
#     for j in range(len(col_lbl)):
#         tbl6[0, j].set_facecolor(PLAIN_HEAD)
#         tbl6[0, j].set_text_props(fontweight="bold", fontsize=9)
#     # Left-align metric name column
#     for i in range(len(section["rows"])):
#         tbl6[i + 1, 0].set_text_props(ha="left")

#     # Add a small note below each sub-table
#     ax.text(0.5, 0.01, section["note"],
#             transform=ax.transAxes, ha="center", va="bottom",
#             fontsize=7.5, color="#555555", style="italic")

# # Single legend: just explain N/A
# na_patch = mpatches.Patch(facecolor="#f5f5f5", edgecolor="#aaa",
#                            label="— = metric not computed for this model")
# fig6.legend(handles=[na_patch], loc="lower center", fontsize=8,
#             framealpha=0.7, bbox_to_anchor=(0.5, -0.03))

# fig6.tight_layout()
# fig6.savefig("fig6_category_tables.png", dpi=150, bbox_inches="tight")
# print("Saved fig6_category_tables.png")


# plt.show()
# print("\nAll figures saved successfully.")
# print("Output files: fig1_toxicity.png, fig2_preservation.png, fig3_judge_scores.png,")
# print("              fig4_dashboard.png, fig5_full_table.png, fig6_category_tables.png")
"""
compare_models_final.py
=======================
✔ Debugs data loading
✔ Compares 3 models
✔ Saves clean table
✔ Generates ONLY 2 graphs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
MODEL_FILES = [
    "reddit_detoxified.parquet",
    "PTSD_reddit_detoxified.parquet",
    "LLM_reddit_detoxified.parquet",
]

MODEL_NAMES = [
    "T5 Baseline",
    "PTSD Model",
    "LLM Model",
]

COLORS = ["#378ADD", "#1D9E75", "#D85A30"]

CONFIG = {
    "old_toxicity_col": "chain_avg_old_toxicity",
    "new_toxicity_col": "chain_avg_new_toxicity",
    "toxicity_threshold": 0.5,
}

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("📥 Loading data...\n")
dfs = [pd.read_parquet(f) for f in MODEL_FILES]

# ═══════════════════════════════════════════════════════════════
# 🔍 DEBUG CHECK (VERY IMPORTANT)
# ═══════════════════════════════════════════════════════════════
print("\n🔍 DEBUGGING DATA:\n")

for i, (name, df) in enumerate(zip(MODEL_NAMES, dfs)):
    print(f"--- {name} ---")
    print("Shape:", df.shape)

    if CONFIG["new_toxicity_col"] in df.columns:
        print("Mean toxicity:", df[CONFIG["new_toxicity_col"]].mean())
    else:
        print("❌ Missing column:", CONFIG["new_toxicity_col"])

    print("Preview:")
    print(df.head(2))
    print("\n")

print("⚠️ If all values above look identical → your files are the same!\n")

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def safe_mean(df, col):
    return df[col].mean() if col in df.columns else np.nan

def toxic_pct(df, col):
    if col not in df.columns:
        return np.nan
    return (df[col] >= CONFIG["toxicity_threshold"]).mean() * 100

def pick(df, a, b):
    if a in df.columns:
        return df[a].mean()
    if b in df.columns:
        return df[b].mean()
    return np.nan

# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════
def compute_stats(df):
    return {
        "Input Toxicity": safe_mean(df, CONFIG["old_toxicity_col"]),
        "Output Toxicity": safe_mean(df, CONFIG["new_toxicity_col"]),
        "% Toxic Output": toxic_pct(df, CONFIG["new_toxicity_col"]),
        "Toxicity Reduction Δ": pick(df, "avg_toxicity_change", "chain_toxicity_change_avg"),
        "Cosine Similarity": pick(df, "avg_cosine_similarity", "chain_cosine_similarity_avg"),
        "BLEU": safe_mean(df, "avg_bleu"),
        "ROUGE-L": safe_mean(df, "avg_rougeL"),
    }

print("📊 Computing metrics...\n")
stats_list = [compute_stats(df) for df in dfs]

# ═══════════════════════════════════════════════════════════════
# SAVE TABLE
# ═══════════════════════════════════════════════════════════════
comparison_df = pd.DataFrame(stats_list, index=MODEL_NAMES)
comparison_df = comparison_df.round(4)

comparison_df.to_csv("model_comparison_table.csv")
comparison_df.to_excel("model_comparison_table.xlsx")

print("\n📊 MODEL COMPARISON TABLE:\n")
print(comparison_df)

print("\n✅ Saved:")
print(" - model_comparison_table.csv")
print(" - model_comparison_table.xlsx")

# ═══════════════════════════════════════════════════════════════
# PLOTTING FUNCTION
# ═══════════════════════════════════════════════════════════════
def bar_plot(ax, values, title, ylabel):
    x = np.arange(len(MODEL_NAMES))
    ax.bar(x, values, color=COLORS[:len(values)])
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    for i, v in enumerate(values):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.3f}", ha="center", fontsize=8)

# ═══════════════════════════════════════════════════════════════
# FIGURE 1 — TOXICITY
# ═══════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
fig1.suptitle("Toxicity Evaluation", fontsize=14, fontweight="bold")

bar_plot(axes[0],
         comparison_df["Output Toxicity"],
         "Output Toxicity", "Score")

bar_plot(axes[1],
         comparison_df["% Toxic Output"],
         "% Toxic Output", "%")

bar_plot(axes[2],
         comparison_df["Toxicity Reduction Δ"],
         "Toxicity Reduction", "Delta")

plt.tight_layout()
plt.savefig("fig1_toxicity.png", dpi=150)

# ═══════════════════════════════════════════════════════════════
# FIGURE 2 — MEANING
# ═══════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle("Meaning Preservation", fontsize=14, fontweight="bold")

bar_plot(axes[0],
         comparison_df["Cosine Similarity"],
         "Cosine Similarity", "Score")

bar_plot(axes[1],
         comparison_df["BLEU"],
         "BLEU", "Score")

bar_plot(axes[2],
         comparison_df["ROUGE-L"],
         "ROUGE-L", "Score")

plt.tight_layout()
plt.savefig("fig2_meaning.png", dpi=150)

plt.show()

print("\n🎯 DONE — Debug + Table + 2 Graphs ready!")