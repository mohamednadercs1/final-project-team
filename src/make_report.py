"""
Task 5 helper: turns results/metrics.csv + results/summary.json into text.

Run AFTER `python src/modeling.py`:
    python src/make_report.py

It (1) fills the AUTO block in report/05_modeling_results.md,
(2) fills the results block in README.md,
(3) fills the numbers in docs/linkedin_post.md (as docs/linkedin_post_filled.md),
(4) compiles report/final_report.md from report/01..05.
No numbers are typed by hand - everything comes from the real results files.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics.csv"
SUMMARY = ROOT / "results" / "summary.json"
START, END = "<!-- AUTO:START -->", "<!-- AUTO:END -->"
R_START, R_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

SECTIONS = [
    "01_dataset_cleaning_eda.md", "02_external_data.md", "03_scraping_ethics.md",
    "04_features_integration.md", "05_modeling_results.md",
]


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def best_per_set(m):
    m = m[m.Model != "Baseline (always fail)"]
    return m.sort_values("ROC-AUC", ascending=False).groupby("Feature set").first()


def lift_sentence(a, b):
    d = b - a
    if d >= 0.02:
        return (f"Adding critic and audience scores **meaningfully improved** the best ROC-AUC "
                f"from {a:.3f} to {b:.3f} (+{d:.3f}).")
    if d > 0.005:
        return (f"Adding critic and audience scores gave a **small improvement** in the best "
                f"ROC-AUC, from {a:.3f} to {b:.3f} (+{d:.3f}).")
    return (f"Adding critic and audience scores **did not materially change** the best "
            f"ROC-AUC ({a:.3f} -> {b:.3f}); budget and franchise information carry most of the signal.")


def build_blocks():
    m = pd.read_csv(METRICS)
    s = json.loads(SUMMARY.read_text())
    best = best_per_set(m)
    a = best.loc["A_pre_release", "ROC-AUC"] if "A_pre_release" in best.index else None
    b = best.loc["B_pre_release+critics", "ROC-AUC"] if "B_pre_release+critics" in best.index else None
    c = best.loc["C_post_release(leaky)", "ROC-AUC"] if "C_post_release(leaky)" in best.index else None
    bm = m[(m["Feature set"] == s["best_feature_set"]) & (m.Model == s["best_model"])].iloc[0]
    base_acc = m[m.Model == "Baseline (always fail)"]["Accuracy"].iloc[0]

    setup = (f"- Movies used: **{s['n_total']}** (train: {s['n_train']} released before {s['split_year']}, "
             f"test: {s['n_test']} released {s['split_year']} or later)\n"
             f"- Success rate (revenue >= ${s['threshold']/1e6:.0f}M): train "
             f"{s['train_success_rate']:.1%}, test {s['test_success_rate']:.1%}\n"
             f"- Features per set: " + ", ".join(f"{k} = {v}" for k, v in s["feature_counts"].items()))

    table = md_table(m)
    findings = [
        f"**Best model:** {s['best_model']} on feature set `{s['best_feature_set']}` - "
        f"ROC-AUC {bm['ROC-AUC']:.3f}, F1 {bm['F1']:.3f}, Precision {bm['Precision']:.3f}, "
        f"Recall {bm['Recall']:.3f}.",
        f"**Baseline check:** always predicting 'flop' gives {base_acc:.1%} accuracy but 0 recall and "
        f"ROC-AUC 0.5, so accuracy alone is misleading here.",
    ]
    if a is not None and b is not None:
        findings.append("**Do critics matter?** " + lift_sentence(a, b))
    if c is not None and b is not None:
        findings.append(f"**Leakage check:** with post-release features (votes, popularity) the best "
                        f"ROC-AUC jumps to {c:.3f} (vs {b:.3f}). That gap is information the studio "
                        f"would not have before release, so Set C is shown only as a warning.")
    findings.append("**Top drivers:** " + ", ".join(f"`{f}`" for f in s["top_features"][:6]) + ".")
    interp = "\n".join(f"- {f}" for f in findings)

    figs = "\n".join(f"![{n}](../results/figures/{n})" for n in
                     ["featureset_comparison.png", "roc_curves.png", "confusion_matrix.png",
                      "feature_importance.png", "shap_summary.png"]
                     if (ROOT / "results" / "figures" / n).exists())

    auto05 = (f"### Experimental setup\n{setup}\n\n### Metrics (test set)\n{table}\n\n"
              f"### Key findings\n{interp}\n\n### Figures\n{figs}")
    readme = f"{table}\n\n**Key findings**\n{interp}"
    return auto05, readme, dict(model=s["best_model"], auc=bm["ROC-AUC"], f1=bm["F1"],
                                a=a, b=b, c=c, top=s["top_features"][:3])


def replace_block(text, start, end, content):
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pat.search(text):
        raise ValueError(f"Marker {start} not found")
    return pat.sub(lambda _: f"{start}\n{content}\n{end}", text)


def main():
    if not METRICS.exists() or not SUMMARY.exists():
        raise SystemExit("Run `python src/modeling.py` first (results files not found).")
    auto05, readme_block, k = build_blocks()

    p05 = ROOT / "report" / "05_modeling_results.md"
    p05.write_text(replace_block(p05.read_text(), START, END, auto05))

    pr = ROOT / "README.md"
    pr.write_text(replace_block(pr.read_text(), R_START, R_END, readme_block))

    post_path = ROOT / "docs" / "linkedin_post.md"
    post = post_path.read_text() if post_path.exists() else ""
    verdict = ("improved" if (k["b"] and k["a"] and k["b"] - k["a"] >= 0.005) else "barely changed")
    post = (post.replace("[MODEL]", k["model"]).replace("[X], F1 [Y]", f"{k['auc']:.2f}, F1 {k['f1']:.2f}")
            .replace("[improved / barely changed]", verdict)
            .replace("from [X] to [Y]", f"from {k['a']:.2f} to {k['b']:.2f}" if k["a"] and k["b"] else "")
            .replace("[budget, sequel, director history]", ", ".join(k["top"])))
    if post:
        (ROOT / "docs" / "linkedin_post_filled.md").write_text(post)

    parts = ["# Final Report - Predicting Movie Commercial Success\n"]
    for name in SECTIONS:
        f = ROOT / "report" / name
        if f.exists():
            parts.append(f.read_text().strip())
    (ROOT / "report" / "final_report.md").write_text("\n\n---\n\n".join(parts) + "\n")
    print("Updated: report/05_modeling_results.md, README.md, "
          "docs/linkedin_post_filled.md, report/final_report.md")


if __name__ == "__main__":
    main()
