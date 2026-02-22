"""
Utility for reading XLSX sheets without openpyxl (avoids heavy dependencies).
No Streamlit imports here.
"""

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd


def _col_to_idx(col_ref: str) -> int:
    """Convert Excel column letter (e.g., 'A', 'AB') to 0-indexed column number."""
    n = 0
    for ch in col_ref:
        if ch.isalpha():
            n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def read_xlsx_sheet_no_openpyxl(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Read Excel sheet without openpyxl (lightweight, no dependencies).
    
    Args:
        xlsx_path: Path to .xlsx file
        sheet_name: Name of sheet to read
        
    Returns:
        DataFrame with sheet contents
    """
    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    with zipfile.ZipFile(xlsx_path) as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
        }

        sheet_rid = None
        for sh in wb.findall("m:sheets/m:sheet", ns):
            if sh.attrib["name"] == sheet_name:
                sheet_rid = sh.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
                break
        if sheet_rid is None:
            raise KeyError(f"Sheet {sheet_name!r} not found in {xlsx_path.name}")

        target = rel_map[sheet_rid]
        if not target.startswith("xl/"):
            target = f"xl/{target}"

        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sroot.findall("m:si", ns):
                shared.append("".join((t.text or "") for t in si.findall(".//m:t", ns)))

        sroot = ET.fromstring(zf.read(target))
        rows = []
        for row in sroot.findall(".//m:sheetData/m:row", ns):
            rec = {}
            for c in row.findall("m:c", ns):
                ref = c.attrib.get("r", "A1")
                idx = _col_to_idx("".join(ch for ch in ref if ch.isalpha()))
                v = c.find("m:v", ns)
                if v is None or v.text is None:
                    rec[idx] = ""
                    continue
                val = v.text
                if c.attrib.get("t") == "s":
                    val = shared[int(val)]
                rec[idx] = val
            rows.append(rec)

    width = max(max(row.keys(), default=0) for row in rows) + 1
    matrix = [[row.get(i, "") for i in range(width)] for row in rows]
    return pd.DataFrame(matrix[2:], columns=matrix[1])
