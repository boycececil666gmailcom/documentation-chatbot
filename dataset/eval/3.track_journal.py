# region Tracking
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env")

CONFIG_PATH = CURRENT_DIR / "eval_config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    eval_config = json.load(f)

journal_cfg = eval_config.get("journal", {})
viz_cfg = eval_config.get("visualization", {})
JOURNAL_FILE = CURRENT_DIR / journal_cfg.get("journal_path", "JOURNAL.md")
CHART_FILE = CURRENT_DIR / "3.track_journal.png"
SUMMARY_FILE = CURRENT_DIR / "3.track_journal.json"


def sync_latest_eval_to_journal(
    eval_dir: Path,
    milestone_name: str,
    journal_path: Path,
) -> dict[str, Any] | None:
    """Appends the latest evaluation CSV summary row to JOURNAL.md."""
    if not journal_path.exists():
        print(f"[JournalTracker-sync] Journal file not found: {journal_path}")
        return None

    csv_pattern = viz_cfg.get("csv_pattern", "2.run_eval_*.csv")
    csv_files = sorted(eval_dir.glob(csv_pattern))
    if not csv_files:
        print(f"[JournalTracker-sync] No evaluation CSV files found in {eval_dir}")
        return None

    latest_csv = csv_files[-1]
    match = re.search(r"2\.run_eval_(\d{8})_(\d{6})\.csv", latest_csv.name)
    if not match:
        print(f"[JournalTracker-sync] Latest file format unexpected: {latest_csv.name}")
        return None

    date_str = f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}"
    time_str = f"{match.group(2)[:2]}:{match.group(2)[2:4]}:{match.group(2)[4:]}"

    df = pd.read_csv(latest_csv)
    summary_mask = df["user_input"].astype(str).str.contains("AVERAGE", case=False, na=False)
    summary_row = (
        df[summary_mask].iloc[0]
        if summary_mask.any()
        else df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean()
    )

    f = float(summary_row.get("faithfulness", 0.0))
    ar = float(summary_row.get("answer_relevancy", 0.0))
    cp = float(summary_row.get("context_precision", 0.0))
    cr = float(summary_row.get("context_recall", 0.0))

    new_row = f"| **{date_str}** (`{time_str}`) | {milestone_name} | `{f:.4f}` | `{ar:.4f}` | `{cp:.4f}` | `{cr:.4f}` |"
    print(f"[JournalTracker-sync] Generated Row:\n{new_row}")

    content = journal_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    table_indices = [
        idx
        for idx, line in enumerate(lines)
        if line.strip().startswith("|") and ("Faithfulness" in line or "`" in line or ":---" in line)
    ]
    if not table_indices:
        print("[JournalTracker-sync] Could not locate benchmark summary table in journal.")
        return None

    last_table_idx = max(table_indices)
    lines.insert(last_table_idx + 1, new_row)
    journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[JournalTracker-sync] Successfully appended latest run to {journal_path.name}")

    return {
        "date": date_str,
        "time": time_str,
        "milestone": milestone_name,
        "source_file": latest_csv.name,
        "metrics": {
            "faithfulness": f,
            "answer_relevancy": ar,
            "context_precision": cp,
            "context_recall": cr,
        },
    }


def load_benchmark_history(eval_dir: Path, csv_pattern: str = "2.run_eval_*.csv") -> pd.DataFrame:
    """Parses all historical evaluation CSV files and extracts summary rows."""
    csv_files = sorted(eval_dir.glob(csv_pattern))
    if not csv_files:
        print(f"[JournalTracker-plot] No CSV benchmark files matching '{csv_pattern}' found.")
        return pd.DataFrame()

    records = []
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    for file_path in csv_files:
        match = re.search(r"2\.run_eval_(\d{8})_(\d{6})\.csv", file_path.name)
        if match:
            date_str = f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}"
            time_str = f"{match.group(2)[:2]}:{match.group(2)[2:4]}:{match.group(2)[4:]}"
            label = f"{date_str} {time_str}"
        else:
            label = file_path.stem.replace("2.run_eval_", "")

        try:
            df = pd.read_csv(file_path)
            summary_mask = df["user_input"].astype(str).str.contains("AVERAGE", case=False, na=False)
            summary_row = df[summary_mask].iloc[0] if summary_mask.any() else df[metric_cols].mean()

            entry = {"Run": label, "File": file_path.name}
            for col in metric_cols:
                val = summary_row.get(col, None)
                try:
                    entry[col] = round(float(val), 4)
                except (ValueError, TypeError):
                    entry[col] = None
            records.append(entry)
        except Exception as err:
            print(f"[JournalTracker-plot] Error parsing {file_path.name}: {err}")

    return pd.DataFrame(records)


def plot_benchmark_trends(history_df: pd.DataFrame, output_image_path: Path, targets: dict[str, float] | None = None) -> None:
    """Renders and saves multi-line trajectory charts across historical benchmark runs."""
    if not HAS_MATPLOTLIB:
        print("[JournalTracker-plot] Matplotlib not installed; skipping chart generation.")
        return

    if history_df.empty:
        print("[JournalTracker-plot] No benchmark data available to plot.")
        return

    metric_targets = targets or {
        "faithfulness": 0.80,
        "answer_relevancy": 0.75,
        "context_precision": 0.85,
        "context_recall": 0.80,
    }
    metric_colors = {
        "faithfulness": "#2563eb",
        "answer_relevancy": "#059669",
        "context_precision": "#d97706",
        "context_recall": "#9333ea",
    }

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=100)
    x_labels = history_df["Run"].tolist()
    x_indices = list(range(len(x_labels)))

    for metric, target in metric_targets.items():
        if metric in history_df.columns:
            values = history_df[metric].tolist()
            color = metric_colors.get(metric, "#475569")
            ax.plot(
                x_indices,
                values,
                marker="o",
                linewidth=2.2,
                label=f"{metric.replace('_', ' ').title()} (Target: {target:.2f})",
                color=color,
            )
            ax.axhline(y=target, color=color, linestyle=":", alpha=0.4)

    ax.set_title("RAG Metric Progression Across Benchmark Runs", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Benchmark Run Timestamp", fontsize=11, labelpad=10)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, labelpad=10)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0.2, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()
    print(f"[JournalTracker-plot] Saved benchmark trend chart to '{output_image_path.name}'")


def sync_langsmith() -> None:
    """Executes 2.5.upload_langsmith.py if present."""
    upload_script = CURRENT_DIR / "2.5.upload_langsmith.py"
    if upload_script.exists():
        print(f"[JournalTracker-langsmith] Triggering LangSmith sync via {upload_script.name}...")
        try:
            res = subprocess.run([sys.executable, str(upload_script)], capture_output=True, text=True, check=True)
            print(f"[JournalTracker-langsmith] {res.stdout.strip()}")
        except Exception as e:
            print(f"[JournalTracker-langsmith] Warning: LangSmith sync failed: {e}")


def main() -> None:
    """Execute journal tracking, visualization plotting, and output summary JSON."""
    milestone_title = journal_cfg.get(
        "milestone_title", "[LangGraph Agent] Native Graph Benchmark Run"
    )
    result = sync_latest_eval_to_journal(
        eval_dir=CURRENT_DIR,
        milestone_name=milestone_title,
        journal_path=JOURNAL_FILE,
    )

    history_df = load_benchmark_history(CURRENT_DIR, csv_pattern=viz_cfg.get("csv_pattern", "2.run_eval_*.csv"))
    plot_benchmark_trends(history_df, output_image_path=CHART_FILE, targets=viz_cfg.get("targets"))
    sync_langsmith()

    summary_payload = {
        "status": "success",
        "latest_run": result,
        "historical_runs_count": len(history_df),
        "chart_file": CHART_FILE.name,
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    print(f"[JournalTracker-main] Saved tracking summary to '{SUMMARY_FILE.name}'")


if __name__ == "__main__":
    main()
# endregion
