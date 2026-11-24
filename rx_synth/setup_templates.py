#!/usr/bin/env python3
"""
Nidaan-Rx Task 1 — Part 2: Jinja2 HTML/CSS Template Builder & Previewer.

Creates all 7 templates inside rx_synth/templates/:
  - T1_emr_bordered.html     (MedSygn-style 6-column bordered EMR table)
  - T2_modern_hospital.html  (Modern e-clinic banner + borderless rows)
  - T3_compact_a5.html       (Compact A5 prescription pad with Rx blocks)
  - T4_dense_derm.html       (Dense 7-9 drug dermatology regimen sheet)
  - T5_minimal_legacy.html   (Low-contrast narrow/monospace legacy printout)
  - HN_lab_report.html       (Hard negative: pathology lab report, 0 drugs)
  - HN_clinic_bill.html      (Hard negative: consultation receipt, 0 drugs)

Usage:
  python rx_synth/setup_templates.py --preview synthetic_rx_dataset/manifests/payloads.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

FONT_STACK = (
    "'Noto Sans', 'Kohinoor Gujarati', 'Noto Sans Gujarati', 'Gujarati Sangam MN', "
    "'Kohinoor Devanagari', 'Noto Sans Devanagari', 'Devanagari Sangam MN', "
    "'Helvetica Neue', Arial, sans-serif"
)

COMMON_RESET = f"""
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 1240px;
    min-height: 1754px;
    background: #ffffff;
    color: #161616;
    font-family: {FONT_STACK};
    -webkit-font-smoothing: antialiased;
  }}
  .page {{
    width: 1240px;
    min-height: 1754px;
    padding: 50px 56px 44px 56px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: #ffffff;
  }}
"""

# ---------------------------------------------------------------------------
# T1: Bordered EMR Table (Primary — mirrors MedSygn dermatology printouts)
# ---------------------------------------------------------------------------
T1_EMR_BORDERED = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
""" + COMMON_RESET + """
  .header-row {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding-bottom: 14px; border-bottom: 2px solid #222;
  }
  .header-left { width: 54%; }
  .logo-box { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
  .logo-mark {
    width: 46px; height: 46px; border: 2.5px solid #2b2b2b; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 20px; color: #222;
  }
  .logo-title { font-size: 23px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
  .doc-name { font-size: 17px; font-weight: 600; margin-bottom: 3px; }
  .doc-meta { font-size: 14.5px; line-height: 1.45; color: #222; }
  .header-right { width: 43%; text-align: right; font-size: 14.5px; line-height: 1.5; color: #222; }
  .clinic-title-r { font-weight: 600; font-size: 15.5px; }
  .patient-bar {
    display: flex; justify-content: space-between;
    padding: 12px 0; border-bottom: 1px solid #bbb; font-size: 15px; line-height: 1.55;
  }
  .patient-right { text-align: right; }
  .clinical-meta { padding: 12px 0 8px 0; font-size: 15px; line-height: 1.75; }
  .rx-heading { font-size: 17px; font-weight: 700; margin: 10px 0 8px 0; }
  table.rx-table { width: 100%; border-collapse: collapse; font-size: 14.5px; margin-bottom: 18px; }
  table.rx-table th, table.rx-table td { border: 1px solid #777; padding: 9px 10px; vertical-align: top; }
  table.rx-table th { background: #f4f4f4; font-weight: 700; text-align: center; }
  table.rx-table th:first-child { text-align: left; padding-left: 14px; }
  .col-drug { width: 41%; }
  .col-dose { width: 11%; text-align: center; }
  .col-freq { width: 15%; text-align: center; }
  .col-instr { width: 16%; text-align: center; }
  .col-dur { width: 11%; text-align: center; }
  .col-qty { width: 6%; text-align: center; }
  .drug-row-flex { display: flex; gap: 8px; align-items: flex-start; }
  .row-idx { font-weight: 600; min-width: 22px; }
  .brand-line { font-weight: 700; font-size: 15px; color: #111; display: inline-block; }
  .salt-line { font-size: 13px; color: #2d2d2d; margin-top: 3px; line-height: 1.38; display: inline-block; }
  .freq-code { font-weight: 600; display: block; }
  .freq-label { display: block; margin-top: 3px; font-size: 13.5px; }
  .freq-note {
    margin-top: 7px; padding-top: 5px; border-top: 1px dashed #ccc;
    font-size: 13px; text-align: left; color: #222; line-height: 1.35;
  }
  .inv-section { margin-top: 14px; font-size: 14px; line-height: 1.5; }
  .inv-heading { font-weight: 700; margin-bottom: 6px; font-size: 15px; }
  .inv-item { margin-bottom: 8px; }
  .inv-title { font-weight: 700; }
  .inv-details { padding-left: 18px; font-size: 13px; color: #222; }
  .followup { margin-top: 16px; font-weight: 700; font-size: 15px; }
  .sig-block { margin-top: 34px; text-align: right; padding-right: 18px; }
  .sig-line {
    display: inline-block; width: 170px; border-bottom: 1.5px solid #333; margin-bottom: 6px;
    font-style: italic; font-family: Georgia, serif; font-size: 20px; text-align: center; color: #222;
  }
  .footer-bar {
    margin-top: 26px; padding-top: 10px; border-top: 1px solid #ccc;
    display: flex; justify-content: space-between; font-size: 13px; color: #333;
  }
</style>
</head>
<body>
<div class="page">
  <div>
    <div class="header-row">
      <div class="header-left">
        <div class="logo-box">
          <div class="logo-mark">{{ doctor.clinic_name[:2].upper() }}</div>
          <div class="logo-title"><span data-ocr-tag="NOISE">{{ doctor.clinic_name.split(' by ')[0] }}</span></div>
        </div>
        <div class="doc-name"><span data-ocr-tag="NOISE">{{ doctor.name }}</span></div>
        <div class="doc-meta">
          <div><span data-ocr-tag="NOISE">Reg. No. {{ doctor.reg_no }}</span></div>
          <div><span data-ocr-tag="NOISE">{{ doctor.qualifications }}</span></div>
        </div>
      </div>
      <div class="header-right">
        <div class="clinic-title-r"><span data-ocr-tag="NOISE">{{ doctor.clinic_name }}</span></div>
        <div><span data-ocr-tag="NOISE">{{ doctor.address_line1 }}</span></div>
        <div><span data-ocr-tag="NOISE">{{ doctor.address_line2 }}</span></div>
        <div><span data-ocr-tag="NOISE">{{ doctor.phone }}</span></div>
        <div><span data-ocr-tag="NOISE">{{ doctor.email }}</span></div>
      </div>
    </div>

    <div class="patient-bar">
      <div>
        <div><strong>Patient Name:</strong> <span data-ocr-tag="NOISE">{{ patient.name }}</span></div>
        <div><strong>Age:</strong> <span data-ocr-tag="NOISE">{{ patient.age }}</span>, <strong>Gender:</strong> <span data-ocr-tag="NOISE">{{ patient.gender }}</span></div>
      </div>
      <div class="patient-right">
        <div><strong>Date:</strong> <span data-ocr-tag="NOISE">{{ patient.visit_date }}</span></div>
        <div><strong>Mobile:</strong> <span data-ocr-tag="NOISE">{{ patient.mobile }}</span>, <strong>UID:</strong> <span data-ocr-tag="NOISE">{{ patient.uid }}</span></div>
      </div>
    </div>

    <div class="clinical-meta">
      <div><strong>Chief Complaints:</strong> <span data-ocr-tag="NOISE">{{ chief_complaints }}</span></div>
      <div><strong>Diagnosis:</strong> <span data-ocr-tag="NOISE">{{ diagnosis }}</span></div>
    </div>

    <div class="rx-heading">Rx</div>
    <table class="rx-table">
      <thead>
        <tr>
          <th class="col-drug">{{ headers.drug }}</th>
          <th class="col-dose">{{ headers.dose }}</th>
          <th class="col-freq">{{ headers.freq }}</th>
          <th class="col-instr">{{ headers.instr }}</th>
          <th class="col-dur">{{ headers.duration }}</th>
          <th class="col-qty">{{ headers.qty }}</th>
        </tr>
      </thead>
      <tbody>
        {% for m in medications %}
        <tr>
          <td class="col-drug">
            <div class="drug-row-flex">
              <span class="row-idx">{{ m.index }}.</span>
              <div>
                <div><span class="brand-line" data-ocr-tag="BRAND" data-row-anchor="{{ m.index }}">{{ m.raw_line }}</span></div>
                <div><span class="salt-line" data-ocr-tag="SALT" data-row-index="{{ m.index }}">{{ m.salt_line }}</span></div>
              </div>
            </div>
          </td>
          <td class="col-dose"><span data-ocr-tag="DOSE">{{ m.posology.dose }}</span></td>
          <td class="col-freq">
            <span class="freq-code" data-ocr-tag="FREQ">{{ m.posology.freq_code }}</span>
            {% if m.posology.freq_label %}<span class="freq-label" data-ocr-tag="FREQ">{{ m.posology.freq_label }}</span>{% endif %}
            {% if m.posology.freq_note %}<div class="freq-note"><strong>Freq. Note:</strong> <span data-ocr-tag="FREQ_NOTE">{{ m.posology.freq_note }}</span></div>{% endif %}
          </td>
          <td class="col-instr"><span data-ocr-tag="INSTR">{{ m.posology.instruction }}</span></td>
          <td class="col-dur"><span data-ocr-tag="DURATION">{{ m.posology.duration }}</span></td>
          <td class="col-qty"><span data-ocr-tag="QTY">{{ m.posology.qty }}</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>

    {% if investigations %}
    <div class="inv-section">
      <div class="inv-heading">Investigation Advised:</div>
      {% for inv in investigations %}
      <div class="inv-item">
        <div class="inv-title"><span data-ocr-tag="NOISE">{{ loop.index }}. {{ inv.title }}</span></div>
        <div class="inv-details"><span data-ocr-tag="NOISE">{{ inv.details }}</span></div>
      </div>
      {% endfor %}
    </div>
    {% endif %}

    {% if followup_days %}
    <div class="followup"><span data-ocr-tag="NOISE">Next followup after {{ followup_days }} Day(s)</span></div>
    {% endif %}

    <div class="sig-block">
      <div class="sig-line">{{ doctor.name.replace('Dr. ', '') }}</div>
      <div style="font-size: 14.5px; font-weight: 600;"><span data-ocr-tag="NOISE">{{ doctor.name }}</span></div>
    </div>
  </div>
  <div class="footer-bar">
    <div><span data-ocr-tag="NOISE">Generated On: {{ patient.generated_on }}</span></div>
    <div>Page No. 1</div>
    <div><span data-ocr-tag="NOISE">{{ doctor.emr_footer }}</span></div>
  </div>
</div>
</body>
</html>"""

# ---------------------------------------------------------------------------
# T2: Modern Hospital / E-Clinic (Accent banner + clean borderless dividers)
# ---------------------------------------------------------------------------
T2_MODERN_HOSPITAL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
""" + COMMON_RESET + """
  .page { padding: 0; }
  .banner {
    background: #1b3b4b; color: #ffffff; padding: 34px 56px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .banner-title { font-size: 26px; font-weight: 700; letter-spacing: 0.8px; }
  .banner-sub { font-size: 14px; color: #dbe7ee; margin-top: 5px; line-height: 1.45; }
  .banner-right { text-align: right; font-size: 14px; line-height: 1.5; color: #eef5f9; }
  .body-wrap { padding: 30px 56px 40px 56px; flex: 1; display: flex; flex-direction: column; justify-content: space-between; }
  .patient-card {
    background: #f4f7f9; border-left: 4px solid #1b3b4b; padding: 14px 18px;
    display: flex; justify-content: space-between; font-size: 14.5px; line-height: 1.6; margin-bottom: 18px;
  }
  .diag-strip {
    display: flex; gap: 30px; padding: 8px 4px 14px 4px; border-bottom: 1px solid #d4dde3;
    font-size: 14.5px; margin-bottom: 16px;
  }
  .rx-badge {
    display: inline-block; background: #1b3b4b; color: #fff; font-weight: 700;
    padding: 4px 12px; border-radius: 4px; font-size: 15px; margin-bottom: 10px;
  }
  table.modern-table { width: 100%; border-collapse: collapse; font-size: 14.5px; }
  table.modern-table th {
    text-align: left; padding: 10px 8px; border-bottom: 2px solid #1b3b4b;
    color: #1b3b4b; font-size: 14px; text-transform: uppercase;
  }
  table.modern-table td { padding: 12px 8px; border-bottom: 1px solid #dde3e8; vertical-align: top; }
  .brand-line { font-weight: 700; font-size: 15px; color: #0f172a; display: inline-block; }
  .salt-line { font-size: 13px; color: #475569; margin-top: 3px; display: inline-block; }
  .note-pill { font-size: 12.5px; color: #334155; margin-top: 5px; font-style: italic; }
  .inv-box { margin-top: 22px; padding: 14px 16px; background: #fafafa; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 13.5px; }
  .sig-row { display: flex; justify-content: space-between; align-items: flex-end; margin-top: 34px; }
  .qr-stamp {
    border: 1.5px dashed #94a3b8; padding: 8px 14px; font-size: 12.5px; color: #475569;
  }
  .footer-bar {
    margin-top: 24px; padding-top: 10px; border-top: 1px solid #cbd5e1;
    display: flex; justify-content: space-between; font-size: 12.5px; color: #475569;
  }
</style>
</head>
<body>
<div class="page">
  <div class="banner">
    <div>
      <div class="banner-title"><span data-ocr-tag="NOISE">{{ doctor.clinic_name.split(' by ')[0] }}</span></div>
      <div class="banner-sub">
        <div><span data-ocr-tag="NOISE">{{ doctor.name }} | Reg: {{ doctor.reg_no }}</span></div>
        <div><span data-ocr-tag="NOISE">{{ doctor.qualifications }}</span></div>
      </div>
    </div>
    <div class="banner-right">
      <div><span data-ocr-tag="NOISE">{{ doctor.address_line1 }}</span></div>
      <div><span data-ocr-tag="NOISE">{{ doctor.address_line2 }}</span></div>
      <div><span data-ocr-tag="NOISE">Ph: {{ doctor.phone }} | {{ doctor.email }}</span></div>
    </div>
  </div>
  <div class="body-wrap">
    <div>
      <div class="patient-card">
        <div>
          <div><strong>Patient:</strong> <span data-ocr-tag="NOISE">{{ patient.name }}</span> (<span data-ocr-tag="NOISE">{{ patient.age }}</span> / <span data-ocr-tag="NOISE">{{ patient.gender }}</span>)</div>
          <div><strong>Patient UID:</strong> <span data-ocr-tag="NOISE">{{ patient.uid }}</span></div>
        </div>
        <div style="text-align:right;">
          <div><strong>Visit Date:</strong> <span data-ocr-tag="NOISE">{{ patient.visit_date }}</span></div>
          <div><strong>Contact:</strong> <span data-ocr-tag="NOISE">{{ patient.mobile }}</span></div>
        </div>
      </div>

      <div class="diag-strip">
        <div><strong>Complaints:</strong> <span data-ocr-tag="NOISE">{{ chief_complaints }}</span></div>
        <div><strong>Diagnosis:</strong> <span data-ocr-tag="NOISE">{{ diagnosis }}</span></div>
      </div>

      <div class="rx-badge">Rx — PRESCRIPTION</div>
      <table class="modern-table">
        <thead>
          <tr>
            <th style="width:42%;">{{ headers.drug }}</th>
            <th style="width:12%;">{{ headers.dose }}</th>
            <th style="width:16%;">{{ headers.freq }}</th>
            <th style="width:16%;">{{ headers.instr }}</th>
            <th style="width:14%;">{{ headers.duration }}</th>
          </tr>
        </thead>
        <tbody>
          {% for m in medications %}
          <tr>
            <td>
              <div><strong>{{ m.index }}.</strong> <span class="brand-line" data-ocr-tag="BRAND" data-row-anchor="{{ m.index }}">{{ m.raw_line }}</span></div>
              <div style="padding-left:18px;"><span class="salt-line" data-ocr-tag="SALT" data-row-index="{{ m.index }}">{{ m.salt_line }}</span></div>
              {% if m.posology.freq_note %}<div style="padding-left:18px;" class="note-pill">Note: <span data-ocr-tag="FREQ_NOTE">{{ m.posology.freq_note }}</span></div>{% endif %}
            </td>
            <td><span data-ocr-tag="DOSE">{{ m.posology.dose }}</span></td>
            <td>
              <div><span data-ocr-tag="FREQ">{{ m.posology.freq_code }}</span></div>
              {% if m.posology.freq_label %}<div style="font-size:13px;"><span data-ocr-tag="FREQ">{{ m.posology.freq_label }}</span></div>{% endif %}
            </td>
            <td><span data-ocr-tag="INSTR">{{ m.posology.instruction }}</span></td>
            <td><span data-ocr-tag="DURATION">{{ m.posology.duration }}</span> {% if m.posology.qty != '-' %}(<span data-ocr-tag="QTY">{{ m.posology.qty }}</span>){% endif %}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>

      {% if investigations %}
      <div class="inv-box">
        <div style="font-weight:700; margin-bottom:6px; color:#1b3b4b;">Recommended Investigations:</div>
        {% for inv in investigations %}
        <div style="margin-bottom:5px;">
          <strong><span data-ocr-tag="NOISE">{{ loop.index }}. {{ inv.title }}</span></strong> — <span data-ocr-tag="NOISE">{{ inv.details }}</span>
        </div>
        {% endfor %}
      </div>
      {% endif %}

      <div class="sig-row">
        <div class="qr-stamp">
          <div style="font-weight:700;">VERIFIED E-RX</div>
          <div><span data-ocr-tag="NOISE">Follow-up: {{ followup_days }} Days</span></div>
        </div>
        <div style="text-align:right;">
          <div style="font-family:Georgia,serif; font-style:italic; font-size:20px; border-bottom:1px solid #333; padding:0 24px 4px 24px;">{{ doctor.name.replace('Dr. ', '') }}</div>
          <div style="font-size:14px; font-weight:600; margin-top:4px;"><span data-ocr-tag="NOISE">{{ doctor.name }}</span></div>
        </div>
      </div>
    </div>

    <div class="footer-bar">
      <div><span data-ocr-tag="NOISE">Generated On: {{ patient.generated_on }}</span></div>
      <div>Page 1 of 1</div>
      <div><span data-ocr-tag="NOISE">{{ doctor.emr_footer }}</span></div>
    </div>
  </div>
</div>
</body>
</html>"""

# ---------------------------------------------------------------------------
# T3: Compact A5 Clinic Pad (No table grid, stacked Rx blocks, large ℞)
# ---------------------------------------------------------------------------
T3_COMPACT_A5 = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
""" + COMMON_RESET + """
  html, body, .page { width: 1120px; min-height: 1580px; }
  .page { padding: 48px 54px; border: 1px solid #eee; }
  .top-head { text-align: center; border-bottom: 2.5px double #222; padding-bottom: 14px; margin-bottom: 16px; }
  .top-head h1 { font-size: 25px; letter-spacing: 1.5px; text-transform: uppercase; }
  .top-head .sub { font-size: 14.5px; margin-top: 4px; line-height: 1.45; }
  .pt-grid {
    display: flex; justify-content: space-between; font-size: 15px; line-height: 1.65;
    padding-bottom: 12px; border-bottom: 1px solid #999; margin-bottom: 14px;
  }
  .rx-sym { font-size: 38px; font-weight: 700; font-family: Georgia, serif; margin: 6px 0 12px 0; }
  .med-card {
    padding: 12px 14px; margin-bottom: 14px; border-left: 3px solid #333; background: #fbfbfb;
  }
  .brand-line { font-weight: 700; font-size: 16px; color: #111; display: inline-block; }
  .salt-line { font-size: 13.5px; color: #333; margin-top: 3px; display: inline-block; }
  .poso-line { margin-top: 7px; font-size: 14px; color: #222; }
  .note-line { margin-top: 4px; font-size: 13px; color: #333; font-style: italic; }
  .footer-bar {
    margin-top: 30px; padding-top: 10px; border-top: 1px solid #bbb;
    display: flex; justify-content: space-between; font-size: 13px;
  }
</style>
</head>
<body>
<div class="page">
  <div>
    <div class="top-head">
      <h1><span data-ocr-tag="NOISE">{{ doctor.clinic_name.split(' by ')[0] }}</span></h1>
      <div class="sub">
        <div><strong><span data-ocr-tag="NOISE">{{ doctor.name }}</span></strong> — <span data-ocr-tag="NOISE">{{ doctor.qualifications }}</span></div>
        <div><span data-ocr-tag="NOISE">Reg: {{ doctor.reg_no }} | {{ doctor.address_line1 }}, {{ doctor.address_line2 }} | Ph: {{ doctor.phone }}</span></div>
      </div>
    </div>

    <div class="pt-grid">
      <div>
        <div><strong>Name:</strong> <span data-ocr-tag="NOISE">{{ patient.name }}</span> (<span data-ocr-tag="NOISE">{{ patient.age }}</span>, <span data-ocr-tag="NOISE">{{ patient.gender }}</span>)</div>
        <div><strong>Dx:</strong> <span data-ocr-tag="NOISE">{{ diagnosis }}</span> | <strong>C/O:</strong> <span data-ocr-tag="NOISE">{{ chief_complaints }}</span></div>
      </div>
      <div style="text-align:right;">
        <div><strong>Date:</strong> <span data-ocr-tag="NOISE">{{ patient.visit_date }}</span></div>
        <div><strong>UID:</strong> <span data-ocr-tag="NOISE">{{ patient.uid }}</span></div>
      </div>
    </div>

    <div class="rx-sym">&#8478;</div>

    {% for m in medications %}
    <div class="med-card">
      <div><strong>{{ m.index }})</strong> <span class="brand-line" data-ocr-tag="BRAND" data-row-anchor="{{ m.index }}">{{ m.raw_line }}</span></div>
      <div style="padding-left:22px;"><span class="salt-line" data-ocr-tag="SALT" data-row-index="{{ m.index }}">{{ m.salt_line }}</span></div>
      <div style="padding-left:22px;" class="poso-line">
        <span data-ocr-tag="DOSE">{{ m.posology.dose }}</span> &nbsp;|&nbsp;
        <span data-ocr-tag="FREQ">{{ m.posology.freq_code }} {{ m.posology.freq_label }}</span> &nbsp;|&nbsp;
        <span data-ocr-tag="INSTR">{{ m.posology.instruction }}</span> &nbsp;|&nbsp;
        <span data-ocr-tag="DURATION">{{ m.posology.duration }}</span>
        {% if m.posology.qty != '-' %} &nbsp;(Qty: <span data-ocr-tag="QTY">{{ m.posology.qty }}</span>){% endif %}
      </div>
      {% if m.posology.freq_note %}
      <div style="padding-left:22px;" class="note-line">Direction: <span data-ocr-tag="FREQ_NOTE">{{ m.posology.freq_note }}</span></div>
      {% endif %}
    </div>
    {% endfor %}

    {% if followup_days %}
    <div style="margin-top:20px; font-weight:700; font-size:15px;"><span data-ocr-tag="NOISE">Review after {{ followup_days }} Day(s)</span></div>
    {% endif %}

    <div style="margin-top:46px; text-align:right;">
      <div style="display:inline-block; border-top:1px solid #333; padding-top:5px; min-width:180px; text-align:center; font-weight:600;">
        <span data-ocr-tag="NOISE">{{ doctor.name }}</span>
      </div>
    </div>
  </div>
  <div class="footer-bar">
    <div><span data-ocr-tag="NOISE">{{ patient.generated_on }}</span></div>
    <div><span data-ocr-tag="NOISE">{{ doctor.emr_footer }}</span></div>
  </div>
</div>
</body>
</html>"""

# ---------------------------------------------------------------------------
# T4: Dense Multi-Drug Dermatology Sheet (7-9 drugs, compact tight grid)
# ---------------------------------------------------------------------------
T4_DENSE_DERM = T1_EMR_BORDERED.replace(
    "padding: 50px 56px 44px 56px;", "padding: 36px 44px 32px 44px;"
).replace(
    "table.rx-table { width: 100%; border-collapse: collapse; font-size: 14.5px; margin-bottom: 18px; }",
    "table.rx-table { width: 100%; border-collapse: collapse; font-size: 13.5px; margin-bottom: 12px; }"
).replace(
    "table.rx-table th, table.rx-table td { border: 1px solid #777; padding: 9px 10px; vertical-align: top; }",
    "table.rx-table th, table.rx-table td { border: 1px solid #666; padding: 6px 8px; vertical-align: top; }"
).replace(
    ".brand-line { font-weight: 700; font-size: 15px; color: #111; display: inline-block; }",
    ".brand-line { font-weight: 700; font-size: 14px; color: #111; display: inline-block; }"
).replace(
    ".salt-line { font-size: 13px; color: #2d2d2d; margin-top: 3px; line-height: 1.38; display: inline-block; }",
    ".salt-line { font-size: 12px; color: #2d2d2d; margin-top: 2px; line-height: 1.3; display: inline-block; }"
)

# ---------------------------------------------------------------------------
# T5: Minimal / Legacy Low-Contrast Printout (Monospace-leaning OCR stress test)
# ---------------------------------------------------------------------------
T5_MINIMAL_LEGACY = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 1240px; min-height: 1754px; background: #fcfcfc; color: #2f2f2f;
    font-family: 'Courier New', 'Kohinoor Gujarati', 'Kohinoor Devanagari', monospace;
  }
  .page {
    width: 1240px; min-height: 1754px; padding: 54px 60px;
    display: flex; flex-direction: column; justify-content: space-between;
  }
  .divider { border-bottom: 1.5px dashed #777; margin: 12px 0; }
  .head-flex { display: flex; justify-content: space-between; font-size: 14.5px; line-height: 1.5; }
  table.leg-table { width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 10px; }
  table.leg-table th { border-top: 1px dashed #666; border-bottom: 1px dashed #666; padding: 7px 6px; text-align: left; }
  table.leg-table td { padding: 8px 6px; border-bottom: 1px dotted #aaa; vertical-align: top; }
  .brand-line { font-weight: 700; font-size: 14.5px; display: inline-block; }
  .salt-line { font-size: 13px; color: #444; display: inline-block; margin-top: 2px; }
  .footer-bar { display: flex; justify-content: space-between; font-size: 12.5px; border-top: 1px dashed #888; padding-top: 10px; }
</style>
</head>
<body>
<div class="page">
  <div>
    <div class="head-flex">
      <div>
        <div style="font-size:19px; font-weight:700;"><span data-ocr-tag="NOISE">{{ doctor.clinic_name.split(' by ')[0].upper() }}</span></div>
        <div><span data-ocr-tag="NOISE">{{ doctor.name }} ({{ doctor.reg_no }})</span></div>
        <div><span data-ocr-tag="NOISE">{{ doctor.qualifications }}</span></div>
      </div>
      <div style="text-align:right;">
        <div><span data-ocr-tag="NOISE">{{ doctor.address_line1 }}</span></div>
        <div><span data-ocr-tag="NOISE">{{ doctor.address_line2 }}</span></div>
        <div><span data-ocr-tag="NOISE">TEL: {{ doctor.phone }}</span></div>
      </div>
    </div>
    <div class="divider"></div>
    <div class="head-flex">
      <div>
        <div>PT: <span data-ocr-tag="NOISE">{{ patient.name }}</span> | <span data-ocr-tag="NOISE">{{ patient.age }}/{{ patient.gender }}</span></div>
        <div>DX: <span data-ocr-tag="NOISE">{{ diagnosis }}</span></div>
      </div>
      <div style="text-align:right;">
        <div>DT: <span data-ocr-tag="NOISE">{{ patient.visit_date }}</span></div>
        <div>UID: <span data-ocr-tag="NOISE">{{ patient.uid }}</span></div>
      </div>
    </div>
    <div class="divider"></div>
    <div style="font-weight:700; font-size:16px; margin-bottom:6px;">RX:</div>
    <table class="leg-table">
      <thead>
        <tr>
          <th style="width:44%;">{{ headers.drug }}</th>
          <th style="width:12%;">{{ headers.dose }}</th>
          <th style="width:16%;">{{ headers.freq }}</th>
          <th style="width:16%;">{{ headers.instr }}</th>
          <th style="width:12%;">{{ headers.duration }}</th>
        </tr>
      </thead>
      <tbody>
        {% for m in medications %}
        <tr>
          <td>
            <div>{{ m.index }}. <span class="brand-line" data-ocr-tag="BRAND" data-row-anchor="{{ m.index }}">{{ m.raw_line }}</span></div>
            <div style="padding-left:18px;"><span class="salt-line" data-ocr-tag="SALT" data-row-index="{{ m.index }}">{{ m.salt_line }}</span></div>
            {% if m.posology.freq_note %}<div style="padding-left:18px; font-size:12.5px;">* <span data-ocr-tag="FREQ_NOTE">{{ m.posology.freq_note }}</span></div>{% endif %}
          </td>
          <td><span data-ocr-tag="DOSE">{{ m.posology.dose }}</span></td>
          <td><span data-ocr-tag="FREQ">{{ m.posology.freq_code }} {{ m.posology.freq_label }}</span></td>
          <td><span data-ocr-tag="INSTR">{{ m.posology.instruction }}</span></td>
          <td><span data-ocr-tag="DURATION">{{ m.posology.duration }}</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>

    {% if investigations %}
    <div style="margin-top:18px; font-size:13px; line-height:1.5;">
      <div style="font-weight:700;">LABS ADVISED:</div>
      {% for inv in investigations %}
      <div>- <span data-ocr-tag="NOISE">{{ inv.title }}: {{ inv.details }}</span></div>
      {% endfor %}
    </div>
    {% endif %}
    <div style="margin-top:18px; font-weight:700;"><span data-ocr-tag="NOISE">NEXT VISIT: {{ followup_days }} DAYS</span></div>
  </div>
  <div class="footer-bar">
    <div><span data-ocr-tag="NOISE">{{ patient.generated_on }}</span></div>
    <div>PAGE 1</div>
    <div><span data-ocr-tag="NOISE">{{ doctor.emr_footer }}</span></div>
  </div>
</div>
</body>
</html>"""

# ---------------------------------------------------------------------------
# HN 1: Diagnostic Pathology Lab Report (Hard Negative — ZERO Medications)
# ---------------------------------------------------------------------------
HN_LAB_REPORT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
""" + COMMON_RESET + """
  .lab-head { border-bottom: 3px solid #1e293b; padding-bottom: 14px; display: flex; justify-content: space-between; }
  .lab-title { font-size: 24px; font-weight: 800; color: #1e293b; }
  .pt-box { margin: 18px 0; padding: 14px 16px; border: 1px solid #94a3b8; display: flex; justify-content: space-between; font-size: 14.5px; line-height: 1.6; }
  .report-banner { text-align: center; font-size: 17px; font-weight: 700; padding: 8px; background: #eef2f6; margin-bottom: 14px; letter-spacing: 1px; }
  table.lab-table { width: 100%; border-collapse: collapse; font-size: 14.5px; }
  table.lab-table th, table.lab-table td { border: 1px solid #94a3b8; padding: 10px 12px; text-align: left; }
  table.lab-table th { background: #f1f5f9; }
  .footer-bar { border-top: 1px solid #cbd5e1; padding-top: 10px; display: flex; justify-content: space-between; font-size: 13px; color: #475569; }
</style>
</head>
<body>
<div class="page">
  <div>
    <div class="lab-head">
      <div>
        <div class="lab-title"><span data-ocr-tag="NOISE">APEX DIAGNOSTIC & PATHOLOGY LAB</span></div>
        <div style="font-size:14px; margin-top:4px;"><span data-ocr-tag="NOISE">NABL Accredited Laboratory | Reg: {{ doctor.reg_no }}</span></div>
      </div>
      <div style="text-align:right; font-size:14px; line-height:1.45;">
        <div><span data-ocr-tag="NOISE">{{ doctor.address_line1 }}</span></div>
        <div><span data-ocr-tag="NOISE">{{ doctor.address_line2 }}</span></div>
        <div><span data-ocr-tag="NOISE">Tel: {{ doctor.phone }}</span></div>
      </div>
    </div>

    <div class="pt-box">
      <div>
        <div><strong>Patient Name:</strong> <span data-ocr-tag="NOISE">{{ patient.name }}</span></div>
        <div><strong>Age / Gender:</strong> <span data-ocr-tag="NOISE">{{ patient.age }} / {{ patient.gender }}</span></div>
        <div><strong>Ref. By:</strong> <span data-ocr-tag="NOISE">{{ doctor.name }}</span></div>
      </div>
      <div style="text-align:right;">
        <div><strong>Lab ID:</strong> <span data-ocr-tag="NOISE">{{ patient.uid }}</span></div>
        <div><strong>Sample Date:</strong> <span data-ocr-tag="NOISE">{{ patient.visit_date }}</span></div>
        <div><strong>Contact:</strong> <span data-ocr-tag="NOISE">{{ patient.mobile }}</span></div>
      </div>
    </div>

    <div class="report-banner">DEPARTMENT OF HEMATOLOGY & BIOCHEMISTRY — TEST REPORT</div>
    <table class="lab-table">
      <thead>
        <tr>
          <th style="width:42%;">Test Description</th>
          <th style="width:18%;">Observed Value</th>
          <th style="width:16%;">Unit</th>
          <th style="width:24%;">Biological Reference Interval</th>
        </tr>
      </thead>
      <tbody>
        {% for row in hard_negative_content.lab_rows %}
        <tr>
          <td><span data-ocr-tag="NOISE">{{ row.test }}</span></td>
          <td><strong><span data-ocr-tag="NOISE">{{ row.value }}</span></strong></td>
          <td><span data-ocr-tag="NOISE">{{ row.unit }}</span></td>
          <td><span data-ocr-tag="NOISE">{{ row.ref }}</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    <div style="margin-top:24px; font-size:13.5px; color:#334155;">
      <span data-ocr-tag="NOISE">Note: Clinical correlation is strictly advised. This is a diagnostic laboratory report and not a medical prescription.</span>
    </div>
  </div>
  <div class="footer-bar">
    <div><span data-ocr-tag="NOISE">Reported On: {{ patient.generated_on }}</span></div>
    <div>End of Report — Page 1</div>
    <div><span data-ocr-tag="NOISE">Pathologist Verified</span></div>
  </div>
</div>
</body>
</html>"""

# ---------------------------------------------------------------------------
# HN 2: Clinic Consultation Bill / Receipt (Hard Negative — ZERO Medications)
# ---------------------------------------------------------------------------
HN_CLINIC_BILL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
""" + COMMON_RESET + """
  .bill-head { display: flex; justify-content: space-between; border-bottom: 2px solid #222; padding-bottom: 14px; }
  .inv-badge { font-size: 22px; font-weight: 800; letter-spacing: 1px; margin: 22px 0 14px 0; text-align: center; }
  table.bill-table { width: 100%; border-collapse: collapse; font-size: 15px; margin-top: 14px; }
  table.bill-table th, table.bill-table td { border: 1px solid #777; padding: 11px 14px; }
  table.bill-table th { background: #f3f4f6; text-align: left; }
  .total-row td { font-weight: 700; font-size: 16px; background: #fafafa; }
  .footer-bar { border-top: 1px solid #bbb; padding-top: 10px; display: flex; justify-content: space-between; font-size: 13px; }
</style>
</head>
<body>
<div class="page">
  <div>
    <div class="bill-head">
      <div>
        <div style="font-size:22px; font-weight:700;"><span data-ocr-tag="NOISE">{{ doctor.clinic_name }}</span></div>
        <div style="font-size:14px; margin-top:4px;"><span data-ocr-tag="NOISE">{{ doctor.name }} | Reg: {{ doctor.reg_no }}</span></div>
      </div>
      <div style="text-align:right; font-size:14px; line-height:1.5;">
        <div><span data-ocr-tag="NOISE">{{ doctor.address_line1 }}</span></div>
        <div><span data-ocr-tag="NOISE">{{ doctor.address_line2 }}</span></div>
        <div><span data-ocr-tag="NOISE">Ph: {{ doctor.phone }}</span></div>
      </div>
    </div>

    <div class="inv-badge">CONSULTATION RECEIPT / TAX INVOICE</div>
    <div style="display:flex; justify-content:space-between; font-size:15px; line-height:1.65; margin-bottom:14px;">
      <div>
        <div><strong>Received From:</strong> <span data-ocr-tag="NOISE">{{ patient.name }}</span></div>
        <div><strong>Patient UID:</strong> <span data-ocr-tag="NOISE">{{ patient.uid }}</span></div>
      </div>
      <div style="text-align:right;">
        <div><strong>Receipt Date:</strong> <span data-ocr-tag="NOISE">{{ patient.visit_date }}</span></div>
        <div><strong>Payment Mode:</strong> <span data-ocr-tag="NOISE">UPI / Cash</span></div>
      </div>
    </div>

    <table class="bill-table">
      <thead>
        <tr>
          <th style="width:10%;">Sr.</th>
          <th style="width:68%;">Description of Clinic Services</th>
          <th style="width:22%; text-align:right;">Amount (INR)</th>
        </tr>
      </thead>
      <tbody>
        {% for b in hard_negative_content.bill_items %}
        <tr>
          <td>{{ loop.index }}</td>
          <td><span data-ocr-tag="NOISE">{{ b.item }}</span></td>
          <td style="text-align:right;"><span data-ocr-tag="NOISE">Rs. {{ b.amount }}.00</span></td>
        </tr>
        {% endfor %}
        <tr class="total-row">
          <td colspan="2" style="text-align:right;">Total Paid</td>
          <td style="text-align:right;"><span data-ocr-tag="NOISE">Rs. {{ hard_negative_content.total_inr }}.00</span></td>
        </tr>
      </tbody>
    </table>

    <div style="margin-top:40px; text-align:right; font-size:14.5px;">
      <div><strong>Authorized Signatory</strong></div>
      <div style="margin-top:6px;"><span data-ocr-tag="NOISE">{{ doctor.clinic_name }}</span></div>
    </div>
  </div>
  <div class="footer-bar">
    <div><span data-ocr-tag="NOISE">Printed: {{ patient.generated_on }}</span></div>
    <div>Computer Generated Receipt</div>
    <div><span data-ocr-tag="NOISE">{{ doctor.emr_footer }}</span></div>
  </div>
</div>
</body>
</html>"""

TEMPLATES = {
    "T1_emr_bordered.html": T1_EMR_BORDERED,
    "T2_modern_hospital.html": T2_MODERN_HOSPITAL,
    "T3_compact_a5.html": T3_COMPACT_A5,
    "T4_dense_derm.html": T4_DENSE_DERM,
    "T5_minimal_legacy.html": T5_MINIMAL_LEGACY,
    "HN_lab_report.html": HN_LAB_REPORT,
    "HN_clinic_bill.html": HN_CLINIC_BILL,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create all 7 Jinja2 templates and optional HTML previews")
    parser.add_argument("--templates-dir", type=Path, default=Path("rx_synth/templates"))
    parser.add_argument("--preview", type=Path, default=None, help="Path to payloads.jsonl to render 1 HTML preview per template")
    parser.add_argument("--preview-dir", type=Path, default=Path("rx_synth/previews"))
    args = parser.parse_args()

    args.templates_dir.mkdir(parents=True, exist_ok=True)
    for fname, content in TEMPLATES.items():
        (args.templates_dir / fname).write_text(content, encoding="utf-8")
    print(f"[DONE] Wrote {len(TEMPLATES)} Jinja2 templates to {args.templates_dir}")

    if args.preview:
        env = Environment(loader=FileSystemLoader(str(args.templates_dir)), autoescape=True)
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        with args.preview.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                tid = payload["template_id"]
                if tid in seen:
                    continue
                tpl = env.get_template(f"{tid}.html")
                html = tpl.render(**payload)
                out_html = args.preview_dir / f"preview_{tid}_{payload['language']}.html"
                out_html.write_text(html, encoding="utf-8")
                seen.add(tid)
                if len(seen) == len(TEMPLATES):
                    break
        print(f"[DONE] Rendered {len(seen)} HTML previews into {args.preview_dir}")
        print(f"       Open them on macOS with: open {args.preview_dir}/*.html")


if __name__ == "__main__":
    main()
