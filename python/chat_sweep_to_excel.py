"""Convert the 1,000-response Chat Lab sweep into an Excel .xlsx workbook.

Run with: python python/chat_sweep_to_excel.py

No third-party packages are required. The writer emits a minimal, valid
SpreadsheetML package using only the standard library, then also writes a CSV
copy for tools that prefer plain text.
"""

from __future__ import annotations

import csv
import json
import os
import zipfile
from datetime import date
from typing import Any, Dict, List, Sequence
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "chat_sweep_report.jsonl")
XLSX = os.path.join(HERE, "chat_sweep_report.xlsx")
CSV = os.path.join(HERE, "chat_sweep_report.csv")
FALLBACK_SUFFIX = "_v2"

HEADERS = (
    "#",
    "Level",
    "Subject",
    "Question",
    "Expected topic/aliases",
    "Status",
    "Failures",
    "Response",
)

COLUMN_WIDTHS = (7, 15, 20, 62, 42, 10, 24, 110)


def _clean(value: Any) -> str:
    text = str(value if value is not None else "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(char for char in text if char >= " " or char in "\n\t")


def _inline_cell(ref: str, value: Any, style: int = 2) -> str:
    text = escape(_clean(value)).replace("\n", "&#10;").replace("\t", "&#9;")
    return (
        f'<c r="{ref}" s="{style}" t="inlineStr">'
        f'<is><t xml:space="preserve">{text}</t></is></c>'
    )


def _number_cell(ref: str, value: int, style: int = 2) -> str:
    return f'<c r="{ref}" s="{style}"><v>{int(value)}</v></c>'


def _column_letter(index: int) -> str:
    letter = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letter = chr(65 + remainder) + letter
    return letter


def _load_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(SOURCE, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) != 1000:
        raise ValueError(f"Expected 1,000 sweep rows, found {len(rows)}")
    return rows


def _sheet_xml(rows: Sequence[Dict[str, Any]]) -> str:
    cols = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(COLUMN_WIDTHS, start=1)
    )
    header_cells = "".join(
        _inline_cell(f"{_column_letter(index)}1", heading, style=1)
        for index, heading in enumerate(HEADERS, start=1)
    )
    row_xml: List[str] = [
        '<row r="1" ht="30" customHeight="1">' + header_cells + "</row>"
    ]

    for row_index, item in enumerate(rows, start=2):
        failures = "; ".join(item.get("failures") or [])
        values: Sequence[Any] = (
            item.get("number", row_index - 1),
            item.get("level", ""),
            item.get("subject", ""),
            item.get("question", ""),
            ", ".join(item.get("expected") or []),
            "PASS" if item.get("passed") else "FAIL",
            failures,
            item.get("answer", ""),
        )
        cells: List[str] = []
        for column_index, value in enumerate(values, start=1):
            ref = f"{_column_letter(column_index)}{row_index}"
            cells.append(
                _number_cell(ref, value)
                if column_index == 1
                else _inline_cell(ref, value)
            )
        row_xml.append(f'<row r="{row_index}">' + "".join(cells) + "</row>")

    last_row = len(rows) + 1
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '<selection pane="bottomLeft" activeCell="A2" sqref="A2"/>'
        '</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f"<cols>{cols}</cols>"
        f"<sheetData>{''.join(row_xml)}</sheetData>"
        f'<autoFilter ref="A1:H{last_row}"/>'
        "</worksheet>"
    )


def _write_xlsx(rows: Sequence[Dict[str, Any]], path: str = XLSX) -> None:
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Responses" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        "</Relationships>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>'
        "</fonts>"
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>'
        "</fills>"
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="3">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>'
        "</cellXfs>"
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles)
        archive.writestr("xl/worksheets/sheet1.xml", _sheet_xml(rows))


def _write_csv(rows: Sequence[Dict[str, Any]], path: str = CSV) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        for item in rows:
            writer.writerow((
                item.get("number", ""),
                item.get("level", ""),
                item.get("subject", ""),
                item.get("question", ""),
                ", ".join(item.get("expected") or []),
                "PASS" if item.get("passed") else "FAIL",
                "; ".join(item.get("failures") or []),
                item.get("answer", ""),
            ))


def _with_fallback(path: str, writer) -> str:
    try:
        writer(path)
        return path
    except PermissionError:
        root, extension = os.path.splitext(path)
        fallback = root + FALLBACK_SUFFIX + extension
        writer(fallback)
        return fallback


def main() -> int:
    rows = _load_rows()
    xlsx_path = _with_fallback(XLSX, lambda path: _write_xlsx(rows, path))
    csv_path = _with_fallback(CSV, lambda path: _write_csv(rows, path))
    passed = sum(1 for item in rows if item.get("passed"))
    print(f"Excel workbook: {xlsx_path}")
    print(f"CSV copy: {csv_path}")
    print(f"Rows: {len(rows)} (generated {date.today().isoformat()})")
    print(f"Passed: {passed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
