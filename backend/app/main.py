"""
Nidaan-Rx v2.0 — FastAPI Backend with Strict Auth, Logout Revocation, Saved Profiles & Multi-Page PDF Reports.
"""

from __future__ import annotations

import io
import json
import os
import re
import sqlite3
from typing import Literal
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

from app.auth import (
    create_access_token,
    hash_password,
    require_user,
    verify_password,
)
from app.core.schemas import (
    CaseContext,
    FollowUpTrend,
    ClinicalAssessmentResponse,
    ModelInferenceOutput,
    PatientProfile,
)
from app.db import CaseRepository
from app.services.inference import ARTIFACTS_DIR, NidaanInferenceService
from app.services.pdf_report import build_clinical_summary_pdf
from app.services.rx_scanner import PrescriptionScannerService

app = FastAPI(
    title="Nidaan-Rx v2.0 API",
    description="Multimodal AI Dermatology Triage with Strict User Auth, 12-Gate Safety, PMBJP Generics & Clinical PDF Reports",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        os.getenv("NIDAAN_FRONTEND_ORIGIN", "http://localhost:3000"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_inference_service() -> NidaanInferenceService:
    return NidaanInferenceService()


@lru_cache(maxsize=1)
def get_case_repo() -> CaseRepository:
    db_path = os.getenv("NIDAAN_DB_PATH", "data/nidaan_v2_cases.db")
    return CaseRepository(db_path=db_path)


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,30}$")


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=160)
    username: str | None = Field(default=None, min_length=3, max_length=30)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str = Field(..., min_length=2, max_length=100)


class LoginRequest(BaseModel):
    identifier: str | None = Field(default=None, min_length=3, max_length=160)
    email: str | None = Field(default=None, min_length=3, max_length=160)
    username: str | None = Field(default=None, min_length=3, max_length=30)
    login_by: Literal["auto", "email", "username"] = "auto"
    password: str = Field(..., min_length=1, max_length=128)


class ReassessRequest(BaseModel):
    profile: PatientProfile
    case_context: CaseContext
    stage1_output: ModelInferenceOutput
    save_case: bool = Field(default=True)
    view_count: int = Field(default=1, ge=1, le=3)


def _default_profile_dict() -> dict[str, Any]:
    return PatientProfile(
        age=25,
        biological_sex="male",
        weight_kg=60.0,
        fitzpatrick_tone=4,
        monk_skin_tone=6,
    ).model_dump(mode="json")


# ==================== AUTH & PROFILE ====================

@app.post("/api/v2/auth/register")
def register_account(payload: RegisterRequest) -> dict[str, Any]:
    email = payload.email.strip().lower()
    full_name = payload.full_name.strip()
    username = payload.username.strip().lower() if payload.username else None

    if len(full_name) < 2:
        raise HTTPException(status_code=422, detail="Full name must be at least 2 characters.")
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="Please enter a valid email address.")
    if username is not None and not USERNAME_RE.match(username):
        raise HTTPException(
            status_code=422,
            detail="Username must be 3–30 characters using only letters, numbers, '.', '_' or '-'.",
        )
    if not payload.password.strip():
        raise HTTPException(status_code=422, detail="Password cannot be blank.")

    repo = get_case_repo()
    if repo.get_user_by_email(email) is not None:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    if username is not None and repo.get_user_by_username(username) is not None:
        raise HTTPException(status_code=409, detail="This username is already taken.")

    try:
        user = repo.create_user(
            email=email,
            username=username,
            full_name=full_name,
            password_hash=hash_password(payload.password),
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="An account with this email or username already exists.") from exc

    saved_profile = repo.upsert_user_profile(
        user["user_id"],
        _default_profile_dict(),
        force_weight_timestamp=True,
        mark_completed=False,
    )
    token = create_access_token(
        user_id=user["user_id"],
        email=user["email"],
        full_name=user["full_name"],
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user,
        "profile": saved_profile["profile"],
        "weight_updated_at": saved_profile["weight_updated_at"],
        "profile_completed": saved_profile["profile_completed"],
    }


@app.post("/api/v2/auth/login")
def login_account(payload: LoginRequest) -> dict[str, Any]:
    repo = get_case_repo()
    raw_id = (payload.identifier or payload.email or payload.username or "").strip().lower()
    if not raw_id:
        raise HTTPException(status_code=422, detail="Please enter your email or username.")

    row = None
    if payload.login_by == "email":
        if not EMAIL_RE.match(raw_id):
            raise HTTPException(status_code=422, detail="Please enter a valid email address.")
        row = repo.get_user_by_email(raw_id)
    elif payload.login_by == "username":
        if not USERNAME_RE.match(raw_id):
            raise HTTPException(status_code=422, detail="Please enter a valid username (3–30 characters).")
        row = repo.get_user_by_username(raw_id)
    else:
        row = repo.get_user_by_email(raw_id) if "@" in raw_id else repo.get_user_by_username(raw_id)

    if row is None or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials. Check your email/username and password.")

    user = {
        "user_id": row["user_id"],
        "email": row["email"],
        "username": row.get("username"),
        "full_name": row["full_name"],
        "created_at": row["created_at"],
    }
    prof = repo.get_user_profile(user["user_id"])
    if prof is None:
        prof = repo.upsert_user_profile(
            user["user_id"],
            _default_profile_dict(),
            force_weight_timestamp=True,
        )

    token = create_access_token(
        user_id=user["user_id"],
        email=user["email"],
        full_name=user["full_name"],
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user,
        "profile": prof["profile"],
        "weight_updated_at": prof["weight_updated_at"],
        "profile_completed": prof["profile_completed"],
    }


@app.post("/api/v2/auth/logout")
def logout_account(claims: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
    jti = claims.get("jti")
    exp = int(claims.get("exp", 0))
    if jti:
        get_case_repo().revoke_token(str(jti), exp)
    return {"logged_out": True}


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=6, max_length=128)


@app.post("/api/v2/auth/change-password")
def change_password(
    payload: ChangePasswordRequest,
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    repo = get_case_repo()
    row = repo.get_user_with_password_by_id(claims["sub"])
    if row is None:
        raise HTTPException(status_code=404, detail="Account not found.")
    if not verify_password(payload.current_password, row["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    if payload.new_password == payload.current_password:
        raise HTTPException(
            status_code=422,
            detail="New password must be different from your current password.",
        )

    repo.update_user_password(row["user_id"], hash_password(payload.new_password))

    # Revoke the previous session token and issue a fresh rotated token
    old_jti = claims.get("jti")
    old_exp = int(claims.get("exp") or 0)
    if old_jti:
        repo.revoke_token(str(old_jti), old_exp)

    new_token = create_access_token(
        user_id=row["user_id"],
        email=row["email"],
        full_name=row["full_name"],
    )
    return {
        "updated": True,
        "access_token": new_token,
        "token_type": "bearer",
    }


@app.get("/api/v2/auth/me")
def get_me(claims: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
    repo = get_case_repo()
    user = repo.get_user_by_id(claims["sub"])
    if user is None:
        raise HTTPException(status_code=401, detail="Account no longer exists.")
    prof = repo.get_user_profile(user["user_id"])
    if prof is None:
        prof = repo.upsert_user_profile(
            user["user_id"],
            _default_profile_dict(),
            force_weight_timestamp=True,
        )
    return {
        "user": user,
        "profile": prof["profile"],
        "weight_updated_at": prof["weight_updated_at"],
        "updated_at": prof["updated_at"],
        "profile_completed": prof["profile_completed"],
    }


@app.get("/api/v2/profile")
def get_saved_profile(claims: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
    repo = get_case_repo()
    prof = repo.get_user_profile(claims["sub"])
    if prof is None:
        prof = repo.upsert_user_profile(
            claims["sub"],
            _default_profile_dict(),
            force_weight_timestamp=True,
        )
    return prof


@app.put("/api/v2/profile")
def update_saved_profile(
    profile: PatientProfile,
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    return get_case_repo().upsert_user_profile(claims["sub"], profile.model_dump(mode="json"), mark_completed=True)


# ==================== PUBLIC HEALTH & BENCHMARKS ====================

@app.get("/api/v2/health")
def health_check() -> dict[str, Any]:
    svc = get_inference_service()
    return {
        "status": "ok",
        "version": "2.2.0",
        "active_backbone": svc.backbone,
        "device": str(svc.device),
        "canonical_classes": len(svc.canonical_classes),
        "tier1_families": len(svc.tier1_classes),
        "vision_encoder_loaded": svc._extractor is not None,
    }


@app.get("/api/v2/metrics")
def get_ablation_metrics() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for bb in ("panderm_base", "dinov2_small"):
        p = Path(ARTIFACTS_DIR) / f"ablation_report_{bb}.json"
        if p.exists():
            reports[bb] = json.loads(p.read_text(encoding="utf-8"))
    return {"ablation_reports": reports}


# ==================== PROTECTED TRIAGE, CASES & PDF ====================

def _parse_json_payload(raw_json: str, model_cls: type[BaseModel], field_name: str) -> Any:
    try:
        data = json.loads(raw_json)
        return model_cls.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {field_name}: {exc}") from exc


@app.post("/api/v2/preview-image")
async def preview_image(
    image: UploadFile = File(...),
    claims: dict[str, Any] = Depends(require_user),
) -> Response:
    """Convert any image format (JPG, PNG, WebP, HEIC, HEIF, BMP, TIFF, AVIF) into a fast JPEG preview."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass

    raw = await image.read()
    with Image.open(io.BytesIO(raw)) as src:
        src = ImageOps.exif_transpose(src)
        if src.mode != "RGB":
            src = src.convert("RGB")
        src.thumbnail((960, 960), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        src.save(out, format="JPEG", quality=86, optimize=True)
    return Response(content=out.getvalue(), media_type="image/jpeg")


@app.post("/api/v2/triage")
async def stage1_triage(
    images: list[UploadFile] = File(..., description="1 to 3 multi-angle skin photos"),
    profile_json: str = Form(..., description="Serialized PatientProfile JSON"),
    case_json: str = Form(..., description="Serialized CaseContext JSON"),
    enforce_quality_gate: bool = Form(True),
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    if not 1 <= len(images) <= 3:
        raise HTTPException(status_code=400, detail="Upload between 1 and 3 images.")

    profile = _parse_json_payload(profile_json, PatientProfile, "profile_json")
    case = _parse_json_payload(case_json, CaseContext, "case_json")
    raw_bytes = [await f.read() for f in images]

    svc = get_inference_service()
    try:
        vis = svc.run_vision_stage(
            raw_images=raw_bytes,
            profile=profile,
            case=case,
            enforce_quality_gate=enforce_quality_gate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    triggers = svc.get_dynamic_triggers(vis.inference_output, profile=profile)
    return {
        "backbone": vis.backbone,
        "latency_ms": vis.latency_ms,
        "all_views_passed_quality": vis.all_views_passed_quality,
        "quality_reports": vis.quality_reports,
        "view_attention_weights": vis.view_attention_weights,
        "stage1_output": vis.inference_output.model_dump(mode="json"),
        "dynamic_trigger_buckets": [tb.model_dump(mode="json") for tb in triggers],
    }


@app.post("/api/v2/assess")
async def full_multimodal_assess(
    images: list[UploadFile] = File(..., description="1 to 3 multi-angle skin photos"),
    profile_json: str = Form(..., description="Serialized PatientProfile JSON"),
    case_json: str = Form(..., description="Serialized CaseContext JSON"),
    enforce_quality_gate: bool = Form(True),
    save_case: bool = Form(True),
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    if not 1 <= len(images) <= 3:
        raise HTTPException(status_code=400, detail="Upload between 1 and 3 images.")

    profile = _parse_json_payload(profile_json, PatientProfile, "profile_json")
    case = _parse_json_payload(case_json, CaseContext, "case_json")
    raw_bytes = [await f.read() for f in images]

    svc = get_inference_service()
    try:
        vis = svc.run_vision_stage(
            raw_images=raw_bytes,
            profile=profile,
            case=case,
            enforce_quality_gate=enforce_quality_gate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    assessment: ClinicalAssessmentResponse = svc.run_clinical_assessment(
        profile=profile,
        case=case,
        stage1_output=vis.inference_output,
    )
    triggers = svc.get_dynamic_triggers(vis.inference_output, profile=profile)

    assessment_dict = assessment.model_dump(mode="json")
    inference_dict = vis.inference_output.model_dump(mode="json")

    case_id = None
    if save_case:
        case_id = get_case_repo().save_assessment(
            backbone=vis.backbone,
            view_count=len(raw_bytes),
            profile_dict=profile.model_dump(mode="json"),
            case_dict=case.model_dump(mode="json"),
            inference_dict=inference_dict,
            assessment_dict=assessment_dict,
            user_id=claims["sub"],
        )

    return {
        "case_id": case_id,
        "backbone": vis.backbone,
        "latency_ms": vis.latency_ms,
        "quality_reports": vis.quality_reports,
        "view_attention_weights": vis.view_attention_weights,
        "stage1_output": inference_dict,
        "dynamic_trigger_buckets": [tb.model_dump(mode="json") for tb in triggers],
        "assessment": assessment_dict,
    }


@app.post("/api/v2/assess-from-triage")
def assess_from_triage(
    payload: ReassessRequest,
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    svc = get_inference_service()
    assessment = svc.run_clinical_assessment(
        profile=payload.profile,
        case=payload.case_context,
        stage1_output=payload.stage1_output,
    )
    assessment_dict = assessment.model_dump(mode="json")
    inference_dict = payload.stage1_output.model_dump(mode="json")

    case_id = None
    if payload.save_case:
        case_id = get_case_repo().save_assessment(
            backbone=svc.backbone,
            view_count=payload.view_count,
            profile_dict=payload.profile.model_dump(mode="json"),
            case_dict=payload.case_context.model_dump(mode="json"),
            inference_dict=inference_dict,
            assessment_dict=assessment_dict,
            user_id=claims["sub"],
        )

    return {
        "case_id": case_id,
        "backbone": svc.backbone,
        "assessment": assessment_dict,
    }


@app.get("/api/v2/cases")
def list_cases(
    limit: int = 20,
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    return {"cases": get_case_repo().list_recent(limit=min(max(limit, 1), 100), user_id=claims["sub"])}


@app.get("/api/v2/cases/{case_id}")
def get_case_detail(
    case_id: str,
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    record = get_case_repo().get_case(case_id, user_id=claims["sub"])
    if record is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    return record


@app.get("/api/v2/cases/{case_id}/report.pdf")
def download_case_pdf(
    case_id: str,
    claims: dict[str, Any] = Depends(require_user),
) -> Response:
    repo = get_case_repo()
    record = repo.get_case(case_id, user_id=claims["sub"])
    if record is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    user = repo.get_user_by_id(claims["sub"])
    pdf_bytes = build_clinical_summary_pdf(record, user=user)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="Nidaan-Rx-{case_id}.pdf"',
            "Cache-Control": "no-store",
        },
    )




class FollowUpCreateRequest(BaseModel):
    followup_day: int = Field(..., ge=1, le=365)
    followup_trend: str = Field(..., min_length=3, max_length=32)
    itch_score: int | None = Field(default=None, ge=0, le=10)
    pain_burning_score: int | None = Field(default=None, ge=0, le=10)
    notes: str = Field(default="", max_length=500)


def _resolve_followup_trend(raw: str) -> FollowUpTrend:
    key = raw.strip().upper()
    if key in FollowUpTrend.__members__:
        return FollowUpTrend[key]
    low = raw.strip().lower()
    for member in FollowUpTrend:
        if str(member.value).lower() == low:
            return member
    raise HTTPException(
        status_code=422,
        detail="Invalid followup_trend. Expected 'improving', 'unchanged', or 'worsening'.",
    )


@app.post("/api/v2/cases/{case_id}/followups")
def log_case_followup(
    case_id: str,
    payload: FollowUpCreateRequest,
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    repo = get_case_repo()
    record = repo.get_case(case_id, user_id=claims["sub"])
    if record is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    max_day = repo.get_max_followup_day(case_id, claims["sub"])
    if payload.followup_day <= max_day:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Follow-up day must be greater than your latest logged check-in (Day {max_day}). "
                f"Enter Day {max_day + 1} or later, or delete a previous entry to revise it."
            ),
        )

    trend_enum = _resolve_followup_trend(payload.followup_trend)
    profile = PatientProfile.model_validate(record["profile"])

    # Keep Day 0 baseline itch/pain immutable in stored case_context_json;
    # evaluate Gate 8 with the check-in's current symptom scores.
    stored_case_dict = dict(record["case_context"])
    stored_case_dict["followup_day"] = int(payload.followup_day)
    stored_case_dict["followup_trend"] = trend_enum.value

    eval_case_dict = dict(stored_case_dict)
    if payload.itch_score is not None:
        eval_case_dict["itch_score"] = int(payload.itch_score)
    if payload.pain_burning_score is not None:
        eval_case_dict["pain_burning_score"] = int(payload.pain_burning_score)

    eval_case = CaseContext.model_validate(eval_case_dict)
    stored_case = CaseContext.model_validate(stored_case_dict)
    stage1_output = ModelInferenceOutput.model_validate(record["inference"])

    svc = get_inference_service()
    assessment = svc.run_clinical_assessment(
        profile=profile,
        case=eval_case,
        stage1_output=stage1_output,
    )
    assessment_dict = assessment.model_dump(mode="json")

    g8_triggered = any(
        g.get("gate_id") == "G8_LONGITUDINAL_ESCALATION" and g.get("triggered")
        for g in (assessment_dict.get("safety_audit_log") or [])
    )
    escalated = bool(g8_triggered or assessment_dict.get("escalation_hard_stop"))
    escalation_reason = assessment_dict.get("escalation_reason")

    entry = repo.add_followup(
        case_id=case_id,
        user_id=claims["sub"],
        followup_day=payload.followup_day,
        followup_trend=trend_enum.value,
        itch_score=payload.itch_score,
        pain_burning_score=payload.pain_burning_score,
        notes=payload.notes,
        escalated=escalated,
        escalation_reason=escalation_reason,
        updated_case_dict=stored_case.model_dump(mode="json"),
        updated_assessment_dict=assessment_dict,
    )
    followups = repo.list_followups(case_id, user_id=claims["sub"])
    return {
        "followup": entry,
        "assessment": assessment_dict,
        "case_context": stored_case.model_dump(mode="json"),
        "followups": followups,
    }


@app.delete("/api/v2/cases/{case_id}/followups/{followup_id}")
def delete_case_followup(
    case_id: str,
    followup_id: str,
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    repo = get_case_repo()
    record = repo.get_case(case_id, user_id=claims["sub"])
    if record is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    removed = repo.delete_followup(case_id, followup_id, claims["sub"])
    if not removed:
        raise HTTPException(status_code=404, detail=f"Follow-up '{followup_id}' not found.")

    latest = repo.get_latest_followup_with_assessment(case_id, claims["sub"])
    stored_case_dict = dict(record["case_context"])

    if latest is not None:
        stored_case_dict["followup_day"] = int(latest["followup_day"])
        stored_case_dict["followup_trend"] = str(latest["followup_trend"])
        restored_assessment = latest["assessment"]
    else:
        stored_case_dict["followup_day"] = 0
        stored_case_dict["followup_trend"] = FollowUpTrend.NONE.value
        profile = PatientProfile.model_validate(record["profile"])
        baseline_case = CaseContext.model_validate(stored_case_dict)
        stage1_output = ModelInferenceOutput.model_validate(record["inference"])
        restored_assessment = get_inference_service().run_clinical_assessment(
            profile=profile,
            case=baseline_case,
            stage1_output=stage1_output,
        ).model_dump(mode="json")

    repo.update_case_state(
        case_id=case_id,
        user_id=claims["sub"],
        case_context_dict=stored_case_dict,
        assessment_dict=restored_assessment,
    )
    return {
        "deleted": True,
        "followup_id": followup_id,
        "assessment": restored_assessment,
        "case_context": stored_case_dict,
        "followups": repo.list_followups(case_id, user_id=claims["sub"]),
    }


@app.delete("/api/v2/cases/{case_id}")
def delete_case_record(
    case_id: str,
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    deleted = get_case_repo().delete_case(case_id, user_id=claims["sub"])
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    return {"deleted": True, "case_id": case_id}




@lru_cache(maxsize=1)
def get_rx_scanner() -> PrescriptionScannerService:
    return PrescriptionScannerService(catalog=get_inference_service()._catalog)


class RxSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=600)


def _load_user_profile_model(user_id: str) -> PatientProfile | None:
    prof_rec = get_case_repo().get_user_profile(user_id)
    if not prof_rec:
        return None
    try:
        return PatientProfile.model_validate(prof_rec["profile"])
    except Exception:
        return None


@app.post("/api/v2/rx/scan")
async def scan_prescription(
    image: UploadFile = File(..., description="Prescription photo (JPG, PNG, WebP, HEIC)"),
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    profile = _load_user_profile_model(claims["sub"])
    return get_rx_scanner().scan_prescription_image(raw_bytes, profile=profile)


@app.post("/api/v2/rx/search")
def search_prescription_text(
    payload: RxSearchRequest,
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    profile = _load_user_profile_model(claims["sub"])
    return get_rx_scanner().search_medicine_text(payload.query, profile=profile)


@app.exception_handler(UnidentifiedImageError)
async def _bad_image_handler(request: Request, exc: UnidentifiedImageError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": "Unsupported or unreadable image file. Please upload a valid JPG, PNG, WebP, or HEIC photo."},
        headers={"Access-Control-Allow-Origin": request.headers.get("origin", "http://localhost:3000")},
    )


@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
        headers={"Access-Control-Allow-Origin": request.headers.get("origin", "http://localhost:3000")},
    )
