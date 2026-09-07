"""
Convert the annotator's Word document (docs/Pattern_Cutting_Annotation_1.docx)
into a clean CSV of ground-truth collision/near-miss events.

The annotation table uses "M.SS" style timestamps (e.g. "3.09" = 3 min 09 s,
"12.34" = 12 min 34 s) because Word/Excel reformatted "mm:ss" entries with a
period. This script parses the table directly from the docx XML (no
python-docx dependency needed) and writes data/ground_truth.csv with time
in seconds.

Usage:
    python scripts/01_convert_annotations.py
"""

import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCX_PATH = PROJECT_ROOT / "docs" / "Pattern_Cutting_Annotation_1.docx"
OUTPUT_CSV = PROJECT_ROOT / "data" / "ground_truth.csv"

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_table_rows(docx_path: Path) -> list[list[str]]:
    """Return every table row in the docx as a list of cell strings."""
    with zipfile.ZipFile(docx_path) as z:
        xml_bytes = z.read("word/document.xml")
    root = ET.fromstring(xml_bytes)

    rows = []
    for tbl in root.iter(f"{W_NS}tbl"):
        table_rows = []
        for tr in tbl.findall(f"{W_NS}tr"):
            cells = []
            for tc in tr.findall(f"{W_NS}tc"):
                texts = tc.findall(f".//{W_NS}t")
                cells.append("".join(t.text or "" for t in texts).strip())
            table_rows.append(cells)
        rows.append(table_rows)
    return rows


def parse_timestamp(value: str) -> float | None:
    """Parse an 'M.SS' or 'MM:SS' style timestamp into total seconds."""
    value = value.strip()
    if not value:
        return None
    value = value.replace(":", ".")
    match = re.match(r"^(\d+)\.(\d+)$", value)
    if not match:
        return None
    minutes, seconds = match.groups()
    return int(minutes) * 60 + int(seconds)


def parse_int_or_none(value: str) -> int | None:
    value = value.strip()
    return int(value) if value.isdigit() else None


def main() -> None:
    tables = read_table_rows(DOCX_PATH)
    # The annotation table is the one whose header starts with "Start Time"
    # and has more than one column named after it (skips the 1-row-per-field
    # "field guide" table that precedes it in the doc).
    annotation_rows = None
    for table in tables:
        if table and table[0] and table[0][0].strip() == "Start Time" and len(table) > 1:
            annotation_rows = table[1:]  # drop header row
    if annotation_rows is None:
        raise RuntimeError("Could not find the annotation table in the docx")

    records = []
    for row in annotation_rows:
        row = row + [""] * (7 - len(row))  # pad short rows
        start_raw, end_raw, event_type, objects, severity_raw, near_miss_raw, notes = row[:7]

        start_s = parse_timestamp(start_raw)
        end_s = parse_timestamp(end_raw)
        if start_s is None and end_s is None:
            continue  # blank trailing row

        severity = parse_int_or_none(severity_raw)
        is_near_miss = 1 if "near miss" in near_miss_raw.lower() else 0
        is_collision = 1 if severity is not None else 0

        records.append(
            {
                "start_time_s": start_s,
                "end_time_s": end_s if end_s is not None else start_s,
                "event_type": event_type,
                "involved_objects": objects,
                "collision_severity": severity if severity is not None else "",
                "near_miss": is_near_miss,
                "is_collision": is_collision,
                "notes": notes,
            }
        )

    records.sort(key=lambda r: (r["start_time_s"] is None, r["start_time_s"]))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    n_collision = sum(r["is_collision"] for r in records)
    n_near_miss = sum(r["near_miss"] for r in records)
    print(f"Wrote {len(records)} events to {OUTPUT_CSV}")
    print(f"  collisions: {n_collision}  near-misses: {n_near_miss}  "
          f"other/unlabeled: {len(records) - n_collision - n_near_miss}")


if __name__ == "__main__":
    main()
