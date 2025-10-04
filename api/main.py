import os
import uuid
import json
from functools import lru_cache
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy import Column, Float, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from transformers import ClapModel, ClapProcessor

SYNONYMS_PATH = os.getenv("SYNONYMS_PATH", "./config/synonyms.json")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "sfx")
DB_PATH = os.getenv("DB_PATH", "./data/meta.sqlite3")
LIBRARY_DIR = os.getenv("LIBRARY_DIR", "./library")
CLAP_MODEL_NAME = os.getenv("CLAP_MODEL", "laion/clap-htsat-unfused")

AUTO_TAGS = os.getenv("AUTO_TAGS", "1") == "1"
AUTO_TAGS_TOPK = int(os.getenv("AUTO_TAGS_TOPK", "5"))
AUTO_TAGS_OVERWRITE = os.getenv("AUTO_TAGS_OVERWRITE", "0") == "1"
CAPTION_MODE = os.getenv("CAPTION_MODE", "basic")
SIM_WEIGHT = float(os.getenv("SIM_WEIGHT", "0.9"))
KW_WEIGHT = float(os.getenv("KW_WEIGHT", "0.1"))

AUTO_TAGS_LABELS_PATH = os.getenv("AUTO_TAGS_LABELS_PATH", "./config/auto_tags_labels.json")
AUTO_TAGS_MIN_SIM = float(os.getenv("AUTO_TAGS_MIN_SIM", "0.28"))
AUTOTAGS_MODE = os.getenv("AUTOTAGS_MODE", "merge").lower()

os.makedirs(LIBRARY_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)


def load_synonyms():
    try:
        with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


SYNONYMS = load_synonyms()


def expand_terms(q: str):
    terms = set()
    for t in q.lower().split():
        terms.add(t)
        if t in SYNONYMS:
            terms.update(SYNONYMS[t])
    return terms


def keyword_bonus(terms, text):
    text = (text or "").lower()
    score = 0
    for t in terms:
        if t in text:
            score += 1
    return score


class Track(Base):
    __tablename__ = "tracks"
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False, unique=True)
    duration = Column(Float, nullable=True)
    tags = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    source_mtime = Column(Float, nullable=True)
    source_size = Column(Float, nullable=True)


Base.metadata.create_all(bind=engine)

qd = QdrantClient(url=QDRANT_URL)


def ensure_collection(size=512):
    collections = qd.get_collections().collections
    if COLLECTION not in {c.name for c in collections}:
        qd.create_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(size=size, distance=qmodels.Distance.COSINE),
            hnsw_config=qmodels.HnswConfigDiff(m=32, ef_construct=256),
        )


_clap_model = None
_clap_proc = None
_label_texts: List[str] = []
_label_embs: Optional[np.ndarray] = None


def get_clap():
    global _clap_model, _clap_proc
    if _clap_model is None or _clap_proc is None:
        _clap_proc = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)
        _clap_model = ClapModel.from_pretrained(CLAP_MODEL_NAME)
        _clap_model.eval()
    return _clap_model, _clap_proc


def embed_text_list(texts: List[str]) -> np.ndarray:
    model, proc = get_clap()
    with torch.no_grad():
        inputs = proc(text=texts, return_tensors="pt", padding=True, truncation=True)
        vecs = model.get_text_features(**inputs).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms


def embed_text(query: str) -> np.ndarray:
    return embed_text_list([query])[0]


@lru_cache(maxsize=10000)
def embed_text_cached(q: str):
    return embed_text(q)


def load_audio(path: str, sr_target=48000):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != sr_target:
        data = librosa.resample(data, orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    max_len = sr_target * 10
    if data.shape[0] > max_len:
        data = data[:max_len]
    return data, sr


def embed_audio(path: str) -> np.ndarray:
    model, proc = get_clap()
    audio, sr = load_audio(path)
    with torch.no_grad():
        inputs = proc(audios=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
        vec = model.get_audio_features(**inputs)[0].cpu().numpy().astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)


def file_fingerprint(path: str) -> Tuple[float, float]:
    st = os.stat(path)
    return float(st.st_mtime), float(st.st_size)


def get_track_by_path(db, path: str):
    return db.query(Track).filter(Track.path == path).first()


def _default_label_texts():
    return [
        "applause", "audience clapping", "crowd cheering",
        "ui click", "ui tap", "ui confirm", "ui error", "ui success", "button press",
        "beep", "chime", "ding", "buzzer", "alarm", "bell",
        "win jingle", "lose jingle", "game over", "level up", "coin", "power up",
        "whoosh", "swoosh", "swipe", "pop", "click", "tick", "boing",
        "ambient room tone", "crowd ambience", "wind", "rain", "water", "fire crackle",
        "footsteps", "door open", "door close", "knock", "glass break", "explosion",
        "animal dog bark", "cat meow", "bird chirp", "car horn", "engine", "airplane",
        "keyboard typing", "camera shutter", "phone vibration"
    ]


def _load_label_space():
    global _label_texts, _label_embs
    texts = None
    try:
        if os.path.isfile(AUTO_TAGS_LABELS_PATH):
            with open(AUTO_TAGS_LABELS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = [str(x) for x in data if str(x).strip()]
                elif isinstance(data, dict) and "labels" in data and isinstance(data["labels"], list):
                    texts = [str(x) for x in data["labels"] if str(x).strip()]
    except Exception:
        texts = None
    if not texts:
        texts = _default_label_texts()
    _label_texts = texts
    _label_embs = embed_text_list(_label_texts)


def _ensure_labels_ready():
    if not _label_texts or _label_embs is None:
        _load_label_space()


def _split_tags(s: str):
    if not s:
        return []
    return [t.strip() for t in s.replace(";", ",").split(",") if t.strip()]


def _merge_tags(user_tags: str, auto_tags: str) -> str:
    u = _split_tags(user_tags)
    a = _split_tags(auto_tags)
    seen, out = set(), []
    for t in u + a:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return ", ".join(out)


def _merge_desc(user_desc: str, auto_desc: str) -> str:
    if user_desc and auto_desc:
        return f"{user_desc} | {auto_desc}"
    return user_desc or auto_desc or ""


def auto_tags_and_description(path: str, topk: int = 5):
    if not AUTO_TAGS:
        return "", ""
    _ensure_labels_ready()
    try:
        avec = embed_audio(path)
        sims = np.dot(_label_embs, avec)
        k = max(1, int(topk))
        order = np.argsort(-sims)
        picked = []
        for idx in order:
            if sims[idx] < AUTO_TAGS_MIN_SIM and len(picked) >= 1:
                break
            picked.append((idx, float(sims[idx])))
            if len(picked) >= k:
                break
        if not picked:
            return "", ""
        labels = [_label_texts[i] for i, _ in picked]
        scores = [s for _, s in picked]
        tags_csv = ", ".join(labels)
        base = os.path.splitext(os.path.basename(path))[0].replace("_", " ").replace("-", " ")
        if CAPTION_MODE == "basic":
            desc = f"{base}. Tags: " + ", ".join(f"{labels[i]} ({scores[i]:.2f})" for i in range(len(labels)))
        else:
            desc = base
        return tags_csv, desc
    except Exception:
        return "", ""


def public_url(path: str) -> str:
    rel = os.path.relpath(path, LIBRARY_DIR)
    return f"/media/{rel}"


ensure_collection(512)

app = FastAPI(title="Audio Semantic Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchResult(BaseModel):
    id: str
    filename: str
    path: str
    score: float
    duration: Optional[float] = None
    tags: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None


@app.on_event("startup")
def _warmup():
    try:
        get_clap()
    except Exception as e:
        print(f"[warmup] CLAP: {e}")


@app.get("/healthz")
def healthz():
    try:
        qd.get_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant not reachable: {e}")
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found")
    return {"ok": True}


@app.get("/media/{subpath:path}")
def media(subpath: str):
    full = os.path.join(LIBRARY_DIR, subpath)
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full)


@app.post("/upload", response_model=SearchResult)
async def upload_audio(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    auto: Optional[int] = Form(1),
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    uid = str(uuid.uuid4())
    save_name = f"{uid}{ext}"
    save_path = os.path.join(LIBRARY_DIR, save_name)
    data = await file.read()
    with open(save_path, "wb") as f:
        f.write(data)
    try:
        y, sr = sf.read(save_path, dtype="float32", always_2d=False)
        duration = float(len(y) / (sr if isinstance(sr, (int, float)) else 48000))
    except Exception:
        duration = None
    atags, adesc = ("", "")
    if AUTO_TAGS and (auto is None or int(auto) == 1):
        atags, adesc = auto_tags_and_description(save_path, AUTO_TAGS_TOPK)
    mode = AUTOTAGS_MODE
    if mode == "overwrite":
        final_tags = atags if AUTO_TAGS and (auto is None or int(auto) == 1) else (tags or "")
        final_desc = adesc if AUTO_TAGS and (auto is None or int(auto) == 1) else (description or f"Auto description: {file.filename}")
    elif mode == "fill_missing":
        final_tags = tags if (tags and tags.strip()) else atags
        final_desc = description if (description and description.strip()) else (adesc or f"Auto description: {file.filename}")
    else:
        final_tags = _merge_tags(tags or "", atags or "")
        final_desc = _merge_desc(description or "", adesc or "")
    vec = embed_audio(save_path)
    ensure_collection(vec.shape[0])
    qd.upsert(
        collection_name=COLLECTION,
        points=[
            qmodels.PointStruct(
                id=uid,
                vector=vec.tolist(),
                payload={
                    "filename": file.filename,
                    "path": save_path,
                    "tags": final_tags,
                    "description": final_desc,
                    "duration": duration or 0.0,
                },
            )
        ],
    )
    with SessionLocal() as db:
        mtime, fsize = file_fingerprint(save_path)
        t = Track(
            id=uid,
            filename=file.filename,
            path=save_path,
            duration=duration,
            tags=final_tags,
            description=final_desc,
            source_mtime=mtime,
            source_size=fsize,
        )
        db.add(t)
        db.commit()
    return SearchResult(
        id=uid,
        filename=file.filename,
        path=save_path,
        score=1.0,
        duration=duration,
        tags=final_tags,
        description=final_desc,
        url=public_url(save_path),
    )


@app.get("/search", response_model=List[SearchResult])
def search(query: str, limit: int = 20):
    vec = embed_text_cached(query)
    ensure_collection(vec.shape[0])
    res = qd.search(
        collection_name=COLLECTION,
        query_vector=vec.tolist(),
        limit=limit,
        with_payload=True,
        search_params=qmodels.SearchParams(hnsw_ef=128),
    )
    terms = expand_terms(query)
    out = []
    with SessionLocal() as db:
        for r in res:
            payload = r.payload or {}
            path = payload.get("path", "")
            t = db.get(Track, str(r.id))
            filename = (t.filename if t else payload.get("filename", ""))
            tags = (t.tags if t else payload.get("tags", ""))
            desc = (t.description if t else payload.get("description", ""))
            sim = float(1 - r.score) if getattr(r, "score", None) is not None else 0.0
            kw = keyword_bonus(terms, " ".join([filename, tags or "", desc or ""]))
            final = SIM_WEIGHT * sim + KW_WEIGHT * kw
            out.append(
                SearchResult(
                    id=str(r.id),
                    filename=filename,
                    path=path,
                    score=final,
                    duration=(t.duration if t else payload.get("duration", 0.0)),
                    tags=tags,
                    description=desc,
                    url=public_url(path) if path else None,
                )
            )
    out.sort(key=lambda x: x.score, reverse=True)
    return out


@app.post("/reindex")
def reindex_folder():
    indexed, updated, removed = [], [], []
    files = []
    for fname in os.listdir(LIBRARY_DIR):
        fpath = os.path.join(LIBRARY_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if os.path.splitext(fname)[1].lower() not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            continue
        files.append(fpath)
    fs_set = set(files)
    with SessionLocal() as db:
        for t in list(db.query(Track).all()):
            if t.path not in fs_set:
                try:
                    qd.delete(collection_name=COLLECTION, points_selector=qmodels.PointIdsList(points=[t.id]))
                except Exception:
                    pass
                db.delete(t)
                removed.append(os.path.basename(t.path))
        db.commit()
        for fpath in files:
            fname = os.path.basename(fpath)
            try:
                mtime, fsize = file_fingerprint(fpath)
            except FileNotFoundError:
                continue
            existing = get_track_by_path(db, fpath)
            track_id = existing.id if existing else str(uuid.uuid4())
            try:
                vec = embed_audio(fpath)
                y, sr = sf.read(fpath, dtype="float32", always_2d=False)
                duration = float(len(y) / (sr if isinstance(sr, (int, float)) else 48000))
            except Exception:
                continue
            atags, adesc = auto_tags_and_description(fpath, AUTO_TAGS_TOPK) if AUTO_TAGS else ("", "")
            base_desc = adesc or f"Auto description: {fname}"
            base_tags = atags or ""
            payload = {
                "filename": fname,
                "path": fpath,
                "tags": (existing.tags if (existing and not AUTO_TAGS_OVERWRITE and existing.tags) else base_tags),
                "description": (existing.description if (existing and not AUTO_TAGS_OVERWRITE and existing.description) else base_desc),
                "duration": duration or 0.0,
            }
            ensure_collection(vec.shape[0])
            qd.upsert(collection_name=COLLECTION, points=[qmodels.PointStruct(id=track_id, vector=vec.tolist(), payload=payload)])
            if existing:
                existing.filename = fname
                existing.duration = duration
                existing.source_mtime = mtime
                existing.source_size = fsize
                if AUTO_TAGS_OVERWRITE or not existing.tags:
                    existing.tags = payload["tags"]
                if AUTO_TAGS_OVERWRITE or not existing.description:
                    existing.description = payload["description"]
                db.commit()
                updated.append(fname)
            else:
                t = Track(
                    id=track_id,
                    filename=fname,
                    path=fpath,
                    duration=duration,
                    tags=payload["tags"],
                    description=payload["description"],
                    source_mtime=mtime,
                    source_size=fsize,
                )
                db.add(t)
                db.commit()
                indexed.append(fname)
    return {"indexed_new": indexed, "updated": updated, "removed_missing": removed}


@app.post("/reindex-incremental")
def reindex_incremental():
    indexed, updated, removed, skipped = [], [], [], []
    current_files = []
    for fname in os.listdir(LIBRARY_DIR):
        fpath = os.path.join(LIBRARY_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if os.path.splitext(fname)[1].lower() not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            continue
        current_files.append(fpath)
    with SessionLocal() as db:
        db_paths = {t.path: t for t in db.query(Track).all()}
        fs_set = set(current_files)
        for path, t in list(db_paths.items()):
            if path not in fs_set:
                try:
                    qd.delete(collection_name=COLLECTION, points_selector=qmodels.PointIdsList(points=[t.id]))
                except Exception:
                    pass
                db.delete(t)
                removed.append(os.path.basename(path))
        db.commit()
        for fpath in current_files:
            fname = os.path.basename(fpath)
            try:
                mtime, fsize = file_fingerprint(fpath)
            except FileNotFoundError:
                continue
            existing = get_track_by_path(db, fpath)
            needs_reembed = False
            if not existing:
                needs_reembed = True
                track_id = str(uuid.uuid4())
            else:
                track_id = existing.id
                if (existing.source_mtime or 0) != mtime or (existing.source_size or 0) != fsize:
                    needs_reembed = True
            if not needs_reembed:
                if AUTO_TAGS and AUTO_TAGS_OVERWRITE and existing:
                    atags, adesc = auto_tags_and_description(fpath, AUTO_TAGS_TOPK)
                    if atags or adesc:
                        existing.tags = atags or existing.tags
                        existing.description = adesc or existing.description
                        db.commit()
                        try:
                            qd.set_payload(
                                collection_name=COLLECTION,
                                payload={"tags": existing.tags or "", "description": existing.description or ""},
                                points=[track_id],
                            )
                        except Exception:
                            pass
                skipped.append(fname)
                continue
            try:
                vec = embed_audio(fpath)
                y, sr = sf.read(fpath, dtype="float32", always_2d=False)
                duration = float(len(y) / (sr if isinstance(sr, (int, float)) else 48000))
            except Exception:
                continue
            if existing and not AUTO_TAGS_OVERWRITE:
                base_tags = existing.tags or ""
                base_desc = existing.description or f"Auto description: {fname}"
            else:
                atags, adesc = auto_tags_and_description(fpath, AUTO_TAGS_TOPK) if AUTO_TAGS else ("", "")
                base_tags = atags or (existing.tags if existing else "") or ""
                base_desc = adesc or (existing.description if existing else f"Auto description: {fname}") or ""
            payload = {
                "filename": fname,
                "path": fpath,
                "tags": base_tags,
                "description": base_desc,
                "duration": duration or 0.0,
            }
            qd.upsert(collection_name=COLLECTION, points=[qmodels.PointStruct(id=track_id, vector=vec.tolist(), payload=payload)])
            if existing:
                existing.filename = fname
                existing.duration = duration
                existing.source_mtime = mtime
                existing.source_size = fsize
                existing.tags = base_tags
                existing.description = base_desc
                db.commit()
                updated.append(fname)
            else:
                t = Track(
                    id=track_id,
                    filename=fname,
                    path=fpath,
                    duration=duration,
                    tags=base_tags,
                    description=base_desc,
                    source_mtime=mtime,
                    source_size=fsize,
                )
                db.add(t)
                db.commit()
                indexed.append(fname)
    return {
        "indexed_new": indexed,
        "updated_changed": updated,
        "removed_missing": removed,
        "skipped_unchanged": skipped,
    }


class UpdateReq(BaseModel):
    tags: Optional[str] = None
    description: Optional[str] = None


@app.patch("/tracks/{track_id}", response_model=SearchResult)
def update_track(track_id: str, body: UpdateReq):
    with SessionLocal() as db:
        t = db.get(Track, track_id)
        if not t:
            raise HTTPException(status_code=404, detail="Not found")
        if body.tags is not None:
            t.tags = body.tags
        if body.description is not None:
            t.description = body.description
        db.commit()
        try:
            qd.set_payload(
                collection_name=COLLECTION,
                payload={"tags": t.tags or "", "description": t.description or ""},
                points=[track_id],
            )
        except Exception:
            pass
        return SearchResult(
            id=t.id,
            filename=t.filename,
            path=t.path,
            score=1.0,
            duration=t.duration,
            tags=t.tags,
            description=t.description,
            url=public_url(t.path),
        )


@app.delete("/tracks/{track_id}")
def delete_track(track_id: str):
    with SessionLocal() as db:
        t = db.get(Track, track_id)
        if not t:
            raise HTTPException(status_code=404, detail="Not found")
        try:
            if os.path.isfile(t.path):
                os.remove(t.path)
        except Exception:
            pass
        try:
            qd.delete(collection_name=COLLECTION, points_selector=qmodels.PointIdsList(points=[track_id]))
        except Exception:
            pass
        db.delete(t)
        db.commit()
    return {"deleted": True, "id": track_id}


class TrackOut(BaseModel):
    id: str
    filename: str
    tags: Optional[str] = None
    description: Optional[str] = None


class BulkItem(BaseModel):
    id: Optional[str] = None
    filename: Optional[str] = None
    tags: Optional[str] = None
    description: Optional[str] = None


class BulkPayload(BaseModel):
    items: List[BulkItem]


@app.get("/tracks", response_model=List[TrackOut])
def list_tracks(limit: int = 1000, offset: int = 0):
    out = []
    with SessionLocal() as db:
        for t in db.query(Track).offset(offset).limit(limit).all():
            out.append(TrackOut(id=t.id, filename=t.filename, tags=t.tags, description=t.description))
    return out


@app.get("/tracks/export")
def export_tracks():
    data = []
    with SessionLocal() as db:
        for t in db.query(Track).all():
            data.append({"id": t.id, "filename": t.filename, "tags": t.tags or "", "description": t.description or ""})
    return {"tracks": data}


@app.post("/tracks/bulk")
def bulk_upsert(body: BulkPayload = Body(...)):
    updated = []
    not_found = []
    with SessionLocal() as db:
        for item in body.items:
            t = None
            if item.id:
                t = db.get(Track, item.id)
            if t is None and item.filename:
                t = db.query(Track).filter(Track.filename == item.filename).first()
            if not t:
                not_found.append({"id": item.id, "filename": item.filename})
                continue
            if item.tags is not None:
                t.tags = item.tags
            if item.description is not None:
                t.description = item.description
            db.commit()
            try:
                qd.set_payload(
                    collection_name=COLLECTION,
                    payload={"tags": t.tags or "", "description": t.description or ""},
                    points=[t.id],
                )
            except Exception:
                pass
            updated.append({"id": t.id, "filename": t.filename})
    return {"updated": updated, "not_found": not_found}


@app.post("/annotate-missing")
def annotate_missing():
    touched = []
    with SessionLocal() as db:
        rows = db.query(Track).all()
        for t in rows:
            needs = (not t.tags or not t.tags.strip()) or (not t.description or not t.description.strip())
            if not AUTO_TAGS or not needs:
                continue
            atags, adesc = auto_tags_and_description(t.path, AUTO_TAGS_TOPK)
            new_tags = t.tags or ""
            new_desc = t.description or ""
            if not new_tags.strip() and atags:
                new_tags = atags
            if not new_desc.strip():
                base = os.path.splitext(os.path.basename(t.filename))[0].replace("_", " ").replace("-", " ")
                new_desc = adesc or base
            if new_tags != (t.tags or "") or new_desc != (t.description or ""):
                t.tags = new_tags
                t.description = new_desc
                db.commit()
                try:
                    qd.set_payload(
                        collection_name=COLLECTION,
                        payload={"tags": t.tags or "", "description": t.description or ""},
                        points=[t.id],
                    )
                except Exception:
                    pass
                touched.append(t.filename)
    return {"annotated": touched}


@app.post("/annotate-force")
def annotate_force():
    touched = []
    with SessionLocal() as db:
        rows = db.query(Track).all()
        for t in rows:
            atags, adesc = auto_tags_and_description(t.path, AUTO_TAGS_TOPK)
            if not atags and not adesc:
                base = os.path.splitext(os.path.basename(t.filename))[0].replace("_", " ").replace("-", " ")
                new_desc = base if not (t.description and t.description.strip()) else t.description
                new_tags = t.tags or ""
            else:
                if AUTO_TAGS_OVERWRITE:
                    new_tags = atags or ""
                    base = os.path.splitext(os.path.basename(t.filename))[0].replace("_", " ").replace("-", " ")
                    new_desc = adesc or base
                else:
                    new_tags = t.tags or atags or ""
                    base = os.path.splitext(os.path.basename(t.filename))[0].replace("_", " ").replace("-", " ")
                    new_desc = t.description or adesc or base
            t.tags = new_tags
            t.description = new_desc
            db.commit()
            try:
                qd.set_payload(
                    collection_name=COLLECTION,
                    payload={"tags": t.tags or "", "description": t.description or ""},
                    points=[t.id],
                )
            except Exception:
                pass
            touched.append(t.filename)
    return {"annotated": touched}


@app.post("/qdrant-gc")
def qdrant_gc():
    kept_ids = set()
    with SessionLocal() as db:
        for t in db.query(Track.id).all():
            kept_ids.add(t[0])
    to_delete = []
    next_page = None
    while True:
        resp = qd.scroll(collection_name=COLLECTION, with_payload=False, with_vectors=False, limit=1024, offset=next_page)
        points, next_page = resp
        if not points:
            break
        for p in points:
            pid = str(p.id)
            if pid not in kept_ids:
                to_delete.append(pid)
        if next_page is None:
            break
    if to_delete:
        qd.delete(collection_name=COLLECTION, points_selector=qmodels.PointIdsList(points=to_delete))
    return {"deleted_orphans": len(to_delete), "kept": len(kept_ids)}


@app.get("/download/{track_id}")
def download_track(track_id: str):
    with SessionLocal() as db:
        t = db.get(Track, track_id)
        if not t or not os.path.isfile(t.path):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(path=t.path, filename=t.filename, media_type="application/octet-stream")
