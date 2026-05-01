import os
import re
import glob
import pandas as pd
from collections import defaultdict

# ============================================
# Configuration / thresholds
# ============================================
REB_FACTOR = 1.2
LAT_DIFF = 50
THP_DIFF = 0.05
VAR_DIFF = 0.05
DROP_DIFF = 2

METRICS = [
    "current-bitrate", "last-bitrate", "rebuffering", "throughput",
    "bitrate-variation", "buffer-length", "dropped-frames",
    "cpu-pressure", "memory-pressure", "latency"
]

INPUT_GLOB = "*.csv"
OUTPUT_FILE = "tcp_quic_paired_with_best.csv"

IGNORE_PATTERNS = [
    r"tcp_segment_avg\.csv$",
    r"quic_segment_avg\.csv$",
    r"tcp_quic_with_best\.csv$",
    r"tcp_quic_paired_with_best\.csv$",
]

# ============================================
# Helpers
# ============================================
def is_ignored(path: str) -> bool:
    base = os.path.basename(path)
    return any(re.search(pat, base, flags=re.IGNORECASE) for pat in IGNORE_PATTERNS)

def get_protocol(filename: str):
    base = os.path.basename(filename).lower()
    if "quic" in base:
        return "quic"
    if "tcp" in base or re.search(r"\btc\b", base):
        return "tcp"
    return None

def get_category(filename: str):
    base = os.path.basename(filename).lower()
    if "mmwave" in base:
        return "mmwave"
    elif "drive" in base:
        return "midband-drive"
    elif "walk" in base:
        return "midband-walk"
    else:
        return "unknown"

def decide_best(row):
    if row["rebuffering_tcp"] > row["rebuffering_quic"] * REB_FACTOR:
        return 1
    elif row["rebuffering_quic"] > row["rebuffering_tcp"] * REB_FACTOR:
        return 0

    lat_diff = row["latency_tcp"] - row["latency_quic"]
    if abs(lat_diff) > LAT_DIFF:
        return 1 if lat_diff > 0 else 0

    thp_diff = (row["throughput_quic"] - row["throughput_tcp"]) / max(row["throughput_tcp"], 1e-6)
    if abs(thp_diff) > THP_DIFF:
        return 1 if thp_diff > 0 else 0

    var_diff = row["bitrate-variation_tcp"] - row["bitrate-variation_quic"]
    if abs(var_diff) > VAR_DIFF:
        return 1 if var_diff > 0 else 0

    drop_diff = row["dropped-frames_tcp"] - row["dropped-frames_quic"]
    if abs(drop_diff) > DROP_DIFF:
        return 1 if drop_diff > 0 else 0

    return 0

# ============================================
# Main
# ============================================
def main():
    files = [f for f in glob.glob(INPUT_GLOB) if f.lower().endswith(".csv") and not is_ignored(f)]
    if not files:
        print("No CSV files found.")
        return

    tcp_files = [f for f in files if get_protocol(f) == "tcp"]
    quic_files = [f for f in files if get_protocol(f) == "quic"]
    print(f"Found {len(tcp_files)} TCP files and {len(quic_files)} QUIC files.")

    records = []
    for path in sorted(files, key=lambda p: os.path.basename(p).lower()):
        proto = get_protocol(path)
        if proto not in {"tcp", "quic"}:
            continue
        category = get_category(path)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"❌ Failed to read {path}: {e}")
            continue

        if "segment-number" not in df.columns:
            print(f"❌ Skipping {path}: missing 'segment-number' column.")
            continue

        # Ignore segment-number == 0
        df = df[df["segment-number"] != 0]

        cols_present = [c for c in METRICS if c in df.columns]
        df = df.reset_index().rename(columns={"index": "_row_order"})
        df = df.sort_values(["segment-number", "_row_order"]).reset_index(drop=True)
        keep_cols = ["segment-number"] + cols_present + ["_row_order"]
        df = df[keep_cols]
        df["filename"] = os.path.basename(path)
        df["protocol"] = proto
        df["category"] = category

        records.append(df)

    if not records:
        print("No usable data rows found.")
        return

    all_df = pd.concat(records, ignore_index=True)
    all_df = all_df.sort_values(
        ["category", "protocol", "filename", "segment-number", "_row_order"]
    ).reset_index(drop=True)

    grouped = defaultdict(dict)
    for (cat, proto, seg), gdf in all_df.groupby(["category", "protocol", "segment-number"], sort=False):
        key = (cat, seg)
        gdf = gdf.sort_values(["filename", "_row_order"]).reset_index(drop=True)
        grouped[key][proto] = gdf

    pairs = []
    segments_considered = 0
    pairs_total = 0

    print("\n=== Pairing details per (category, segment) ===")
    for (cat, seg), sides in grouped.items():
        tcp_df = sides.get("tcp")
        quic_df = sides.get("quic")
        if tcp_df is None or quic_df is None:
            continue

        k = min(len(tcp_df), len(quic_df))
        if k == 0:
            continue

        segments_considered += 1
        pairs_total += k

        print(f"[{segments_considered}] Category={cat}, Segment={seg}: "
              f"TCP rows={len(tcp_df)}, QUIC rows={len(quic_df)}, "
              f"Pairs formed={k}, Running total={pairs_total}")

        tcp_k = tcp_df.iloc[:k].reset_index(drop=True)
        quic_k = quic_df.iloc[:k].reset_index(drop=True)

        out = pd.DataFrame({
            "category": [cat] * k,
            "segment-number": [seg] * k,
            "filename_quic": quic_k["filename"],
            "filename_tcp": tcp_k["filename"],
        })

        for m in METRICS:
            out[f"{m}_quic"] = quic_k[m] if m in quic_k.columns else pd.NA
            out[f"{m}_tcp"] = tcp_k[m] if m in tcp_k.columns else pd.NA

        out["Best"] = out.apply(decide_best, axis=1)
        pairs.append(out)

    if not pairs:
        print("No TCP↔QUIC pairs could be formed.")
        return

    result = pd.concat(pairs, ignore_index=True)
    ordered_cols = ["category", "segment-number", "filename_quic", "filename_tcp"]
    for m in METRICS:
        q = f"{m}_quic"
        t = f"{m}_tcp"
        if q in result.columns and t in result.columns:
            ordered_cols += [q, t]
    ordered_cols += ["Best"]

    result = result[ordered_cols]
    result.to_csv(OUTPUT_FILE, index=False)

    cats = sorted(result["category"].unique())
    segs = result[["category", "segment-number"]].drop_duplicates().shape[0]

    print(
        f"\n✅ Saved {OUTPUT_FILE} with {len(result)} paired rows.\n"
        f"   • Categories covered: {cats}\n"
        f"   • Unique (category, segment-number) groups paired: {segs}\n"
        f"   • Segments considered for pairing: {segments_considered}\n"
        f"   • Total pairs formed: {pairs_total}"
    )

if __name__ == "__main__":
    main()
