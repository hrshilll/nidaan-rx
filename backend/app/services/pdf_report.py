"""
Nidaan-Rx v2.0 — Multi-Page Clinical Triage Summary PDF Generator.
Produces a structured medical summary containing ALL patient inputs, symptoms,
triggers, Stage-1 vs Stage-2 differentials, precautions, treatments + PMBJP
Jan Aushadhi equivalents, lifestyle plan, and the complete 12-Gate Safety Audit.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from fpdf import FPDF


def _clean(val: Any) -> str:
    s = str(val if val is not None else "")
    replacements = {
        "\u20b9": "Rs. ",
        "\u2014": "-",
        "\u2013": "-",
        "\u2022": "*",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2192": "->",
        "\u00b0": " deg",
        "\u00b7": "|",
        "\u2265": ">=",
        "\u2264": "<=",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.encode("latin-1", "replace").decode("latin-1").strip()


def _fmt_list(items: list[Any] | None, empty: str = "None reported") -> str:
    if not items:
        return empty
    cleaned = [_clean(str(x).replace("_", " ")) for x in items if str(x).strip()]
    return ", ".join(cleaned) if cleaned else empty


class _ClinicalPDF(FPDF):
    def __init__(self, case_id: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.case_id = _clean(case_id)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(15, 16, 15)

    def header(self) -> None:
        if self.page_no() > 1:
            self.set_font("Helvetica", "B", 8.5)
            self.set_text_color(8, 145, 178)
            self.cell(0, 6, f"Nidaan-Rx Clinical Summary  |  Case {self.case_id}", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(203, 235, 242)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(3)

    def footer(self) -> None:
        self.set_y(-13)
        self.set_draw_color(220, 236, 241)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(1.5)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(100, 116, 139)
        self.cell(
            self.epw - 25,
            5,
            "Nidaan-Rx v2.0 | Educational clinical triage summary - not a substitute for an in-person dermatologist.",
        )
        self.cell(25, 5, f"Page {self.page_no()} / {{nb}}", align="R")

    def section_title(self, title: str) -> None:
        self.ln(3)
        if self.get_y() > self.h - 36:
            self.add_page()
        self.set_fill_color(236, 254, 255)
        self.set_text_color(22, 78, 99)
        self.set_font("Helvetica", "B", 11)
        self.set_x(self.l_margin)
        self.cell(self.epw, 8, f"  {_clean(title)}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2.5)

    def kv_row(self, label: str, value: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(71, 85, 105)
        self.cell(48, 5.5, _clean(label) + ":")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(22, 78, 99)
        self.multi_cell(self.epw - 48, 5.5, _clean(value), new_x="LMARGIN", new_y="NEXT")

    def bullet(self, text: str, color: tuple[int, int, int] = (51, 65, 85)) -> None:
        self.set_x(self.l_margin + 3)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*color)
        self.multi_cell(self.epw - 3, 5.2, f"- {_clean(text)}", new_x="LMARGIN", new_y="NEXT")
        self.ln(0.8)


def build_clinical_summary_pdf(record: dict[str, Any], user: dict[str, Any] | None = None) -> bytes:
    case_id = str(record.get("case_id") or "N/A")
    created_raw = str(record.get("created_at") or "")
    try:
        dt_str = datetime.fromisoformat(created_raw.replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        dt_str = created_raw or "N/A"

    profile: dict[str, Any] = record.get("profile") or {}
    ctx: dict[str, Any] = record.get("case_context") or {}
    inf: dict[str, Any] = record.get("inference") or {}
    asmt: dict[str, Any] = record.get("assessment") or {}

    pdf = _ClinicalPDF(case_id=case_id)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title Banner
    pdf.set_fill_color(8, 145, 178)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(pdf.epw, 11, "  Nidaan-Rx v2.0 - Clinical Triage Summary", fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_fill_color(243, 250, 252)
    pdf.set_text_color(22, 78, 99)
    pdf.set_font("Helvetica", "", 9)
    meta_parts = [
        f"Case ID: {case_id}",
        f"Date: {dt_str}",
        f"Model: {record.get('backbone', 'panderm_base')}",
        f"Views: {record.get('view_count', 1)}",
        f"OOD: {'Flagged' if inf.get('is_ood') else 'In-Distribution'} ({float(inf.get('ood_score', 0.0)):.2f})",
    ]
    if user:
        meta_parts.insert(0, f"Patient: {user.get('full_name', '')} ({user.get('email', '')})")
    pdf.multi_cell(pdf.epw, 5.5, "  " + _clean("  |  ".join(meta_parts)), fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Hard-stop Escalation Alert
    if asmt.get("escalation_hard_stop"):
        pdf.set_fill_color(254, 242, 242)
        pdf.set_draw_color(220, 38, 38)
        pdf.set_text_color(153, 27, 27)
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.cell(pdf.epw, 7, "  URGENT ESCALATION: IN-PERSON MEDICAL EVALUATION ADVISED", border="TLR", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        reason = asmt.get("escalation_reason") or "Red-flag signs or high-risk features detected. Self-treatment suppressed."
        pdf.multi_cell(pdf.epw, 5.5, f"  {_clean(reason)}", border="BLR", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
    elif asmt.get("escalation_reason"):
        pdf.set_fill_color(255, 251, 235)
        pdf.set_draw_color(217, 119, 6)
        pdf.set_text_color(146, 64, 14)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(pdf.epw, 7, "  LONGITUDINAL ESCALATION NOTICE (GATE 8)", border="TLR", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(pdf.epw, 5.5, f"  {_clean(asmt['escalation_reason'])}", border="BLR", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # SECTION 1: PATIENT INPUTS & SYMPTOMS
    pdf.section_title("1. Patient Inputs & Clinical Context")
    fitz_roman = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}.get(int(profile.get("fitzpatrick_tone", 4)), str(profile.get("fitzpatrick_tone", "")))
    pdf.kv_row("Demographics", f"Age {profile.get('age', '-')} yrs | Sex: {str(profile.get('biological_sex', '-')).title()} | Weight: {profile.get('weight_kg', '-')} kg")
    pdf.kv_row("Skin Tone", f"Monk Skin Tone {profile.get('monk_skin_tone', '-')} / 10  |  Fitzpatrick Type {fitz_roman}")
    if str(profile.get("biological_sex", "")).lower() == "female":
        pdf.kv_row("Pregnancy / Lactation", f"Pregnant: {'Yes' if profile.get('is_pregnant') else 'No'}  |  Lactating: {'Yes' if profile.get('is_lactating') else 'No'}")
    pdf.kv_row("Sleep & Stress", f"{profile.get('sleep_hours', '-')} hrs/night ({str(profile.get('sleep_pattern', '-')).replace('_', ' ')})  |  Stress: {str(profile.get('stress_level', '-')).title()}")
    pdf.kv_row("Local Climate Tags", _fmt_list(profile.get("climate_tags")))
    pdf.kv_row("Drug Allergies", _fmt_list(profile.get("drug_allergies")))
    pdf.kv_row("Ongoing Oral Meds", _fmt_list(profile.get("ongoing_oral_meds")))
    pdf.kv_row("Long-Term Conditions", _fmt_list(profile.get("systemic_conditions")))
    pdf.kv_row("Current Skincare Actives", _fmt_list(profile.get("current_routine_actives")))
    pdf.ln(1.5)

    site = str(ctx.get("anatomical_site", "-")).replace("_", " ").title()
    _d = ctx.get('duration_days', '-')
    pdf.kv_row("Affected Site & Duration", f"{site}  |  Duration: {_d} {'day' if _d == 1 else 'days'}")
    pdf.kv_row("Symptom Scores", f"Itch: {ctx.get('itch_score', 0)}/10  |  Pain/Burning: {ctx.get('pain_burning_score', 0)}/10")
    pdf.kv_row("Surface Findings", f"Visible scaling: {'Yes' if ctx.get('has_scaling') else 'No'}  |  Raw/cracked barrier: {'Yes' if ctx.get('compromised_barrier') else 'No'}")

    rf_dict = ctx.get("red_flags") or {}
    active_rf = [k.replace("_", " ") for k, v in rf_dict.items() if v]
    pdf.kv_row("Red-Flag Symptoms", _fmt_list(active_rf, empty="None active"))
    pdf.kv_row("Selected Triggers", _fmt_list(asmt.get("identified_triggers") or ctx.get("active_triggers"), empty="None selected"))
    if ctx.get("other_trigger_note"):
        pdf.kv_row("Patient Notes", str(ctx.get("other_trigger_note")))

    # SECTION 2: DIFFERENTIAL ASSESSMENT
    pdf.section_title("2. Differential Assessment (Stage-1 Vision vs. Stage-2 Refined)")
    matches = asmt.get("top_matches") or []
    if not matches:
        pdf.bullet("No differential matches available.")
    else:
        for idx, m in enumerate(matches, 1):
            prior = float(m.get("prior_prob", 0.0)) * 100.0
            post = float(m.get("posterior_prob", 0.0)) * 100.0
            delta = post - prior
            fam = str(m.get("tier1_family", "")).replace("_", " ").title()
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(22, 78, 99)
            pdf.multi_cell(
                pdf.epw,
                5.5,
                _clean(f"{idx}. {m.get('display_name', m.get('condition_id'))}  -  {post:.1f}%  (Stage-1: {prior:.1f}%, Shift: {delta:+.1f}%)  [{fam}]"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            contribs = m.get("contributing_triggers") or []
            labels = [str(c.get("label") or c.get("trigger_id", "")) for c in contribs if isinstance(c, dict)]
            if labels:
                pdf.set_x(pdf.l_margin + 4)
                pdf.set_font("Helvetica", "", 8.5)
                pdf.set_text_color(71, 85, 105)
                pdf.multi_cell(pdf.epw - 4, 4.8, _clean("Contributing factors: " + "; ".join(labels)), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)

    # SECTION 3: DO-NOT-APPLY PRECAUTIONS
    dna = asmt.get("do_not_apply_warnings") or []
    if dna:
        pdf.section_title("3. Do-Not-Apply Precautions")
        for w in dna:
            pdf.bullet(w, color=(153, 27, 27))

    # SECTION 4: TREATMENTS & PMBJP JAN AUSHADHI EQUIVALENTS
    def _render_cards(cards: list[dict[str, Any]], heading: str) -> None:
        pdf.section_title(heading)
        if not cards:
            pdf.bullet("No options listed (withheld or in-person dermatologist evaluation required).")
            return
        for card in cards:
            opt = card.get("option") or {}
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(22, 78, 99)
            pdf.multi_cell(
                pdf.epw,
                5.5,
                _clean(f"{opt.get('display_name', '')}  [{opt.get('vehicle', '')} | {card.get('regulatory_label', '')}]"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            if opt.get("posology"):
                pdf.bullet(f"Usage: {opt['posology']}")
            if opt.get("notes"):
                pdf.bullet(f"Notes: {opt['notes']}")
            if card.get("dosage_note"):
                pdf.bullet(f"Dosage Safety Note: {card['dosage_note']}", color=(146, 64, 14))
            badges = card.get("inline_badges") or []
            if badges:
                pdf.bullet("Safety Notes: " + " | ".join(str(b) for b in badges))
            matches_pmbjp = card.get("pmbjp_matches") or []
            if matches_pmbjp:
                pdf.set_x(pdf.l_margin + 3)
                pdf.set_font("Helvetica", "B", 8.5)
                pdf.set_text_color(4, 120, 87)
                pdf.cell(pdf.epw - 3, 5, "Verified Jan Aushadhi (PMBJP) Generics:", new_x="LMARGIN", new_y="NEXT")
                for gm in matches_pmbjp:
                    code = f"#{gm['generic_code']}" if gm.get("generic_code") else ""
                    mrp = f"Rs. {float(gm['mrp_inr']):.0f}" if gm.get("mrp_inr") is not None else "MRP N/A"
                    comm = f" (Commercial avg: Rs. {float(gm['commercial_avg_inr']):.0f})" if gm.get("commercial_avg_inr") is not None else ""
                    sav = f" - Save {round(float(gm['estimated_savings_pct']))}%" if gm.get("estimated_savings_pct") else ""
                    pdf.bullet(f"{gm.get('product_name', '')} {code} [{gm.get('vehicle', '')}] - {mrp}{comm}{sav}", color=(4, 120, 87))
            pdf.ln(2)

    _render_cards(asmt.get("tier1_otc_care") or [], "4A. Tier-1 OTC & Supportive Care + Jan Aushadhi Generics")
    _render_cards(asmt.get("tier2_dermatologist_prep") or [], "4B. Tier-2 Dermatologist Discussion Options (Not Self-Prescription)")

    # SECTION 5: LIFESTYLE PLAN
    lifestyle = asmt.get("lifestyle_action_plan") or []
    if lifestyle:
        pdf.section_title("5. Daily Skin & Environment Action Plan")
        for item in lifestyle:
            pdf.bullet(item)

    # SECTION 5B: LONGITUDINAL FOLLOW-UP TIMELINE (GATE 8)
    followups = record.get("followups") or []
    if followups:
        pdf.section_title("5B. Longitudinal Follow-Up Timeline (Gate 8)")
        for fu in followups:
            tag = "[GATE 8 ESCALATED]" if fu.get("escalated") else "[ON TRACK]"
            col = (180, 83, 9) if fu.get("escalated") else (4, 120, 87)
            scores = []
            if fu.get("itch_score") is not None:
                scores.append(f"Itch {fu['itch_score']}/10")
            if fu.get("pain_burning_score") is not None:
                scores.append(f"Pain {fu['pain_burning_score']}/10")
            score_str = f" ({', '.join(scores)})" if scores else ""
            note_str = f" - Note: {fu['notes']}" if fu.get("notes") else ""
            pdf.bullet(
                f"Day {fu.get('followup_day')} | Trend: {str(fu.get('followup_trend', '')).title()}{score_str} {tag}{note_str}",
                color=col,
            )

    # SECTION 6: 12-GATE CLINICAL SAFETY AUDIT LOG
    audit_log = asmt.get("safety_audit_log") or []
    if audit_log:
        pdf.section_title("6. Complete 12-Gate Clinical Safety Audit Log")
        for g in audit_log:
            action = str(g.get("action", "PASS"))
            col = (185, 28, 28) if action in {"BLOCK", "HARD_STOP"} else (146, 64, 14) if action in {"WARN", "MODIFY"} else (4, 120, 87)
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_text_color(*col)
            pdf.multi_cell(
                pdf.epw,
                5,
                _clean(f"[{action}] {g.get('gate_id', '')} - {g.get('gate_name', '')}: {g.get('message', '')}"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.ln(0.5)

    # Mandatory Disclaimer
    if asmt.get("disclaimer_banner"):
        pdf.ln(3)
        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(71, 85, 105)
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(pdf.epw, 4.6, _clean(asmt["disclaimer_banner"]), fill=True, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())
