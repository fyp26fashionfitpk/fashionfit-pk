"""Hardened, atomic, CPU-testable runner for 03D FASHN segmentation_free=True vs False control.

Guarantees:
- Resolves images strictly from verified frozen case & inventory records (refuses missing/changed inputs).
- Freezes protocol.json before generation; refuses silent protocol drift.
- Writes per-case sidecars and outputs atomically (.tmp -> os.replace) so partial failures never lose metadata.
- Resumes ONLY verified successful outputs (matching protocol signature + output SHA-256).
- Flags corrupt or altered outputs explicitly instead of silently overwriting them.
- Generates paired comparison galleries (Donor | Recipient | seg_free=True | seg_free=False) and a separate review CSV.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageOps


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_verified_cases(root: Path):
    def read(path):
        with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    cases=read(root/'05_SPLITS/cases.csv')
    index={r['image_id']:r for r in read(root/'05_SPLITS/frozen.csv')}
    original=json.loads((root/'10_RUNS/benchmark_baseline_fashn_vton_1_5/protocol.json').read_text(encoding='utf-8'))
    expected={r['case_id']:r for r in original['cases']}
    if len(cases)!=20 or {c['case_id'] for c in cases}!=set(expected):raise ValueError('Expected the original twenty evaluated case IDs')
    for c in cases:
        if any(c[k]!=expected[c['case_id']][k] for k in ('scenario','source_image_id','target_image_id','mode')):raise ValueError('Evaluated case definition changed')
        for key in ('source_image_id','target_image_id'):
            r=index[c[key]]
            if r['split']!='test' or sha256_file(Path(r['image_path']))!=original['inputs'][c[key]]['image']:raise ValueError('Non-test or changed evaluated image')
    return cases,index


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_save_image(im: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    im.convert("RGB").save(tmp, format="PNG")
    os.replace(tmp, path)


def write_csv_atomic(path: Path, rows: list[dict], fields: list[str]) -> None:
    buf=io.StringIO(newline='')
    writer=csv.DictWriter(buf,fieldnames=fields,extrasaction='ignore')
    writer.writeheader();writer.writerows(rows)
    atomic_write_text(path,buf.getvalue())


def preserve_review(path: Path, new_rows: list[dict], fields: list[str]) -> None:
    if not path.exists():
        write_csv_atomic(path,new_rows,fields);return
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);existing=list(reader);old_fields=reader.fieldnames or []
    if old_fields!=fields:raise ValueError('Review schema changed; preserve existing review and create a new version')
    ids=[r['output_id'] for r in existing]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate review identities')
    added=[r for r in new_rows if r['output_id'] not in set(ids)]
    if added:write_csv_atomic(path,existing+added,fields)


def run_fashn_segfree_control(
    *,
    cases: list[dict[str, str]],
    inventory: dict[str, dict[str, str]],
    out_dir: Path,
    infer_fn: Callable[[Image.Image, Image.Image, str, bool, int], dict[str, Any]],
    selected_case_ids: list[str] | None = None,
    conditions: tuple[bool, ...] = (True, False),
    seed: int = 42,
    steps: int = 30,
    guidance: float = 1.5,
    model_revision: str = "7720683168567eb5a2a4c67f15116c6e29c83ded",
    runtime_identity: dict | None = None,
) -> dict[str, Any]:
    """Execute or safely resume the controlled FASHN segmentation_free=True/False experiment."""
    if tuple(conditions)!=(True,False) or steps!=30 or guidance!=1.5:
        raise ValueError('This experiment requires both conditions, 30 steps and guidance 1.5; version any alternative explicitly')
    out_dir.mkdir(parents=True, exist_ok=True)
    full_plan=sorted(cases,key=lambda c:c['case_id'])
    if len({c['case_id'] for c in full_plan})!=len(full_plan):raise ValueError('Duplicate case IDs')
    if selected_case_ids is not None and not set(selected_case_ids)<= {c['case_id'] for c in full_plan}:raise ValueError('Unknown selected case IDs')
    plan = [c for c in full_plan if (selected_case_ids is None or c["case_id"] in selected_case_ids)]
    if not plan:
        raise ValueError("No matching cases selected for 03D control.")

    input_hashes: dict[str, str] = {}
    for c in full_plan:
        if c.get('mode') not in {'upper_body','full_outfit','flared_lehenga'}:raise ValueError('Missing/unsupported case mode')
        for key in ("source_image_id", "target_image_id"):
            img_id = c[key]
            if img_id not in inventory:
                raise ValueError(f"Case {c['case_id']} references unknown image_id '{img_id}'.")
            p = Path(inventory[img_id]["image_path"])
            if not p.is_file():
                raise FileNotFoundError(f"Missing frozen input image for {c['case_id']}: {p}")
            input_hashes[img_id] = sha256_file(p)

    protocol = {
        "experiment_id": "03D_FASHN_SegmentationFree_Control_v2",
        "model": "fashn_vton_1_5",
        "model_revision": model_revision,
        "author_code": "7c0f10af3f91ad4048fe9729c470a13ef905d25a",
        "runtime_identity": runtime_identity or {},
        "garment_photo_type": "model",
        "parser_note": "Uses FASHN internal parsing when segmentation_free=False (NOT V4 SegFormer).",
        "conditions_segmentation_free": list(conditions),
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "cases": [{k:c[k] for k in ('case_id','scenario','source_image_id','target_image_id','mode')} for c in full_plan],
        "input_hashes": input_hashes,
    }
    protocol_sig = sha256_bytes(json.dumps(protocol, sort_keys=True).encode("utf-8"))
    proto_path = out_dir / "protocol.json"
    if proto_path.exists():
        existing = json.loads(proto_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError(
                f"Protocol mismatch in {proto_path}. Create a new versioned folder instead of altering frozen protocol."
            )
    else:
        atomic_write_text(proto_path, json.dumps(protocol, indent=2))

    counts = {"planned": len(full_plan) * len(conditions), "scheduled":len(plan)*len(conditions), "succeeded": 0, "resumed": 0, "failed": 0, "corrupt_flagged": 0}
    records: list[dict[str, Any]] = []

    for c in plan:
        cid = c["case_id"]
        person_path = Path(inventory[c["target_image_id"]]["image_path"])
        donor_path = Path(inventory[c["source_image_id"]]["image_path"])
        cat = "tops" if c.get("mode", "upper_body") == "upper_body" else "one-pieces"

        for seg_free in conditions:
            tag = "segfree_true" if seg_free else "segfree_false"
            out_img_path = out_dir / "outputs" / f"{cid}__{tag}.png"
            sidecar_path = out_dir / "outputs" / f"{cid}__{tag}.json"

            expected=dict(case_id=cid,source_image_id=c['source_image_id'],target_image_id=c['target_image_id'],scenario=c['scenario'],category=cat,segmentation_free=seg_free,protocol_signature=protocol_sig)
            if out_img_path.exists() and not sidecar_path.exists():
                flagged=dict(expected,output_id=f'03D__{cid}__{tag}',status='corrupt_or_changed',flagged_reason='orphan_image_without_sidecar')
                atomic_write_text(out_img_path.with_suffix('.CORRUPT_FLAG.json'),json.dumps(flagged,indent=2))
                counts['corrupt_flagged']+=1;records.append(flagged);continue

            # Resume ONLY verified successful outputs; flag corrupt/changed outputs instead of overwriting!
            if sidecar_path.exists():
                prev = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if prev.get("status") == "succeeded":
                    if (
                        out_img_path.is_file()
                        and all(prev.get(k)==v for k,v in expected.items())
                        and prev.get("protocol_signature") == protocol_sig
                        and prev.get("output_sha256") == sha256_file(out_img_path)
                    ):
                        counts["resumed"] += 1
                        counts["succeeded"] += 1
                        records.append(prev)
                        continue
                    # Flag corrupt or tampered output rather than silently overwriting
                    flagged = dict(prev, status="corrupt_or_changed", flagged_reason="output_hash_or_protocol_mismatch")
                    atomic_write_text(out_dir / "outputs" / f"{cid}__{tag}.CORRUPT_FLAG.json", json.dumps(flagged, indent=2))
                    counts["corrupt_flagged"] += 1
                    records.append(flagged)
                    continue

            rec: dict[str, Any] = {
                "output_id": f"03D__{cid}__{tag}",
                "case_id": cid,
                "scenario": c.get("scenario", "unknown"),
                "source_image_id": c["source_image_id"],
                "target_image_id": c["target_image_id"],
                "source_sha256": input_hashes[c["source_image_id"]],
                "target_sha256": input_hashes[c["target_image_id"]],
                "category": cat,
                "garment_photo_type": "model",
                "segmentation_free": seg_free,
                "seed": seed,
                "steps": steps,
                "guidance": guidance,
                "protocol_signature": protocol_sig,
                "status": "failed",
                "error": "",
                "seconds": None,
                "peak_vram_mb": None,
                "output_sha256": None,
            }
            try:
                person = Image.open(person_path).convert("RGB")
                donor = Image.open(donor_path).convert("RGB")
                res = infer_fn(person, donor, cat, seg_free, seed)
                atomic_save_image(res["image"], out_img_path)
                rec.update(
                    status="succeeded",
                    seconds=res.get("seconds"),
                    peak_vram_mb=res.get("peak_vram_mb"),
                    output_sha256=sha256_file(out_img_path),
                )
                counts["succeeded"] += 1
            except Exception as exc:
                rec["error"] = str(exc)
                counts["failed"] += 1
            atomic_write_text(sidecar_path, json.dumps(rec, indent=2))
            records.append(rec)

        # Build paired comparison gallery ONLY if both conditions succeeded and are uncorrupted
        case_recs = {r["segmentation_free"]: r for r in records if r.get("case_id") == cid}
        p_true = out_dir / "outputs" / f"{cid}__segfree_true.png"
        p_false = out_dir / "outputs" / f"{cid}__segfree_false.png"
        if (
            case_recs.get(True, {}).get("status") == "succeeded"
            and case_recs.get(False, {}).get("status") == "succeeded"
            and p_true.is_file()
            and p_false.is_file()
        ):
            gal = Image.new("RGB", (1200, 430), "white")
            draw = ImageDraw.Draw(gal)
            imgs = [
                ("Donor", Image.open(donor_path).convert("RGB")),
                ("Recipient", Image.open(person_path).convert("RGB")),
                ("seg_free=True", Image.open(p_true).convert("RGB")),
                ("seg_free=False", Image.open(p_false).convert("RGB")),
            ]
            for idx, (lbl, im) in enumerate(imgs):
                gal.paste(ImageOps.contain(im, (290, 390)), (idx * 300 + 5, 30))
                draw.text((idx * 300 + 10, 8), f"{cid} | {lbl}", fill="black")
            atomic_save_image(gal, out_dir / "galleries" / f"{cid}_paired.png")

    # A later smoke-only visit must not hide already completed full-run outputs.
    scheduled={c['case_id'] for c in plan}
    for c in full_plan:
        if c['case_id'] in scheduled:continue
        for seg_free in conditions:
            tag='segfree_true' if seg_free else 'segfree_false'
            p=out_dir/'outputs'/f"{c['case_id']}__{tag}.png";meta=p.with_suffix('.json')
            if not meta.exists():
                if p.exists():
                    counts['corrupt_flagged']+=1
                    records.append(dict(output_id=f"03D__{c['case_id']}__{tag}",case_id=c['case_id'],segmentation_free=seg_free,status='corrupt_or_changed'))
                continue
            prev=json.loads(meta.read_text(encoding='utf-8'))
            valid=(prev.get('protocol_signature')==protocol_sig and prev.get('source_image_id')==c['source_image_id'] and prev.get('target_image_id')==c['target_image_id'] and prev.get('segmentation_free')==seg_free)
            if prev.get('status')=='succeeded' and valid and p.exists() and sha256_file(p)==prev.get('output_sha256'):
                counts['succeeded']+=1;records.append(prev)
            elif prev.get('status')=='failed' and valid:
                counts['failed']+=1;records.append(prev)
            else:
                counts['corrupt_flagged']+=1;records.append(dict(prev,status='corrupt_or_changed'))

    # Atomically write manifest CSV and separate review packet CSV.
    manifest_fields=list(dict.fromkeys(k for r in records for k in r)) or ['output_id','status']
    write_csv_atomic(out_dir/'outputs_manifest.csv',records,manifest_fields)
    review_fields=['output_id','case_id','scenario','condition','design_score','silhouette_score','identity_score','artifact_score','scope_score','major_artifact','scope_violation','notes']
    review_rows=[]
    for r in records:
        if r.get("status") == "succeeded":
            cond = "segfree_true" if r["segmentation_free"] else "segfree_false"
            review_rows.append(dict(output_id=r['output_id'],case_id=r['case_id'],scenario=r['scenario'],condition=cond))
    preserve_review(out_dir/'review_packet_03D.csv',review_rows,review_fields)

    counts['not_run']=counts['planned']-counts['succeeded']-counts['failed']-counts['corrupt_flagged']
    summary = {"protocol_signature": protocol_sig, "counts": counts,'complete':counts['succeeded']==counts['planned'],'review_note':'Condition-labelled exploratory review, not blinded human evaluation.'}
    atomic_write_text(out_dir / "run_summary.json", json.dumps(summary, indent=2))
    return summary
