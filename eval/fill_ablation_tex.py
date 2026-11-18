"""Fill finished ablation rows in main.tex. Skips a variant until every binary claim is present."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/yxfeng/www27")
TEX = ROOT / "main.tex"
RESULTS = Path("/data3/yxfeng/www27/results/ablate")
ORDER = ["averitec", "scifact", "healthver", "climate_fever"]
EXPECTED = {"averitec": 427, "scifact": 188, "healthver": 143, "climate_fever": 907}
ROWS = {
    "no_gate": r"        w/o Judge-1 (retrieve until $C$)",
    "holistic": r"        w/o aspect selector (holistic)",
    "random_gate": "        Random gate",
    "random_aspect": "        Random aspects",
    "doffy_gen": r"        \method{}-Gen (generate all)",
}


def load_variant(name: str) -> dict[str, dict] | None:
    directory = RESULTS / name
    if not directory.is_dir():
        return None
    grouped: dict[str, dict[str, dict]] = {bench: {} for bench in ORDER}
    for path in directory.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            bench = str(row.get("benchmark") or "")
            if bench not in grouped:
                continue
            grouped[bench][str(row.get("id"))] = row
    if any(len(grouped[bench]) != EXPECTED[bench] for bench in ORDER):
        return None
    return grouped


def cell(rows: dict[str, dict]) -> tuple[str, dict]:
    parts = []
    stats = {}
    for bench in ORDER:
        items = list(rows[bench].values())
        scored = [row for row in items if row.get("prediction") is not None and row.get("gold") is not None]
        correct = sum(row["prediction"] == row["gold"] for row in scored)
        calls = [int(row.get("n_calls") or 0) for row in scored]
        acc = 100.0 * correct / len(scored)
        mean_calls = sum(calls) / len(calls)
        parts.append(f"{acc:.2f} & {mean_calls:.2f}")
        lats = [float(row["latency_s"]) for row in scored if row.get("latency_s") is not None]
        fwds = [float(row.get("n_llm_fwd") or 0) for row in scored]
        toks = [float(row.get("n_gen_tokens") or 0) for row in scored]
        stats[bench] = {
            "acc": acc,
            "calls": mean_calls,
            "lat": sum(lats) / len(lats) if lats else 0.0,
            "fwd": sum(fwds) / len(fwds),
            "tok": sum(toks) / len(toks),
            "n": len(scored),
        }
    return " & ".join(parts), stats


def macro(stats: dict) -> dict[str, float]:
    return {
        key: sum(stats[bench][key] for bench in ORDER) / len(ORDER)
        for key in ("fwd", "tok", "lat", "calls")
    }


DOFFY_ACC = {
    "averitec": 85.95,
    "scifact": 81.91,
    "healthver": 77.62,
    "climate_fever": 75.85,
}
PROSE_NAME = {
    "no_gate": "Removing Judge-1",
    "holistic": "Replacing the aspect selector with one holistic query",
    "random_gate": "Replacing Judge-1 with a random draw",
    "random_aspect": "Replacing the aspect selector with a random aspect",
    "doffy_gen": "Generating the gate, the aspect, and the verdict",
}


def measured_paragraph(ready: dict[str, dict]) -> str:
    sentences = []
    for name in ROWS:
        stats = ready.get(name)
        if stats is None:
            continue
        accs = "/".join(f"{stats[bench]['acc']:.2f}" for bench in ORDER)
        calls = "/".join(f"{stats[bench]['calls']:.2f}" for bench in ORDER)
        deltas = "/".join(f"{stats[bench]['acc'] - DOFFY_ACC[bench]:+.2f}" for bench in ORDER)
        sentences.append(
            f"{PROSE_NAME[name]} reaches {accs} on AVeriTeC, SciFact, HealthVer, and Climate-FEVER "
            f"at {calls} calls, a change of {deltas} points from Doffy."
        )
    return " ".join(sentences)


def replace_marked(text: str, paragraph: str) -> str:
    start = "% BEGIN_RQ2_NUMBERS"
    end = "% END_RQ2_NUMBERS"
    before, middle, after = text.partition(start)
    if not middle:
        return text
    _, _, tail = after.partition(end)
    body = f"\n{paragraph}\n" if paragraph else "\n"
    return f"{before}{start}{body}{end}{tail}"


def main() -> None:
    text = TEX.read_text(encoding="utf-8")
    original = text
    ready = {}
    for name, prefix in ROWS.items():
        grouped = load_variant(name)
        if grouped is None:
            continue
        body, stats = cell(grouped)
        ready[name] = stats
        new_line = f"{prefix} & {body} \\\\"
        lines = []
        replaced = False
        for line in text.splitlines():
            if line.startswith(prefix) and not replaced:
                lines.append(new_line)
                replaced = True
            else:
                lines.append(line)
        if not replaced:
            raise SystemExit(f"row not found: {name}")
        text = "\n".join(lines) + "\n"
        if name == "doffy_gen":
            summary = macro(stats)
            eff = (
                f"        \\method{{}}-Gen & {summary['fwd']:.2f} & {summary['tok']:.1f} "
                f"& {summary['lat']:.2f} & {summary['calls']:.2f} \\\\"
            )
            lines = []
            replaced = False
            for line in text.splitlines():
                if line.startswith(r"        \method{}-Gen &") and not replaced:
                    lines.append(eff)
                    replaced = True
                else:
                    lines.append(line)
            if not replaced:
                raise SystemExit("efficiency row not found")
            text = "\n".join(lines) + "\n"
    text = replace_marked(text, measured_paragraph(ready))
    if text != original:
        TEX.write_text(text, encoding="utf-8")
        print("updated", TEX)
    else:
        print("no complete variant yet")
    for name, stats in ready.items():
        print(name, {bench: (round(stats[bench]["acc"], 2), round(stats[bench]["calls"], 2)) for bench in ORDER})


if __name__ == "__main__":
    main()
