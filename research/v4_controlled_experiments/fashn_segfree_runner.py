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
import json
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageOps


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


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
) -> dict[str, Any]:
    """Execute or safely resume the controlled FASHN segmentation_free=True/False experiment."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = [c for c in cases if (selected_case_ids is None or c["case_id"] in selected_case_ids)]
    if not plan:
        raise ValueError("No matching cases selected for 03D control.")

    input_hashes: dict[str, str] = {}
    for c in plan:
        for key in ("source_image_id", "target_image_id"):
            img_id = c[key]
            if img_id not in inventory:
                raise ValueError(f"Case {c['case_id']} references unknown image_id '{img_id}'.")
            p = Path(inventory[img_id]["image_path"])
            if not p.is_file():
                raise FileNotFoundError(f"Missing frozen input image for {c['case_id']}: {p}")
            input_hashes[img_id] = sha256_file(p)

    protocol = {
        "experiment_id": "03D_FASHN_SegmentationFree_Control_v1",
        "model": "fashn_vton_1_5",
        "model_revision": model_revision,
        "garment_photo_type": "model",
        "parser_note": "Uses FASHN internal parsing when segmentation_free=False (NOT V4 SegFormer).",
        "conditions_segmentation_free": list(conditions),
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "case_ids": [c["case_id"] for c in plan],
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

    counts = {"planned": len(plan) * len(conditions), "succeeded": 0, "resumed": 0, "failed": 0, "corrupt_flagged": 0}
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

            # Resume ONLY verified successful outputs; flag corrupt/changed outputs instead of overwriting!
            if sidecar_path.exists():
                prev = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if prev.get("status") == "succeeded":
                    if (
                        out_img_path.is_file()
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

    # Atomically write manifest CSV and separate review packet CSV
    manifest_fields = list(records[0].keys()) if records else ["output_id", "status"]
    buf_lines = [",".join(manifest_fields)]
    for r in records:
        buf_lines.append(",".join(json.dumps(str(r.get(k, ""))) for k in manifest_fields))
    atomic_write_text(out_dir / "outputs_manifest.csv", "\n".join(buf_lines) + "\n")

    rev_lines = ["output_id,case_id,scenario,condition,design_score,silhouette_score,identity_score,artifact_score,scope_score,major_artifact,scope_violation,notes"]
    for r in records:
        if r.get("status") == "succeeded":
            cond = "segfree_true" if r["segmentation_free"] else "segfree_false"
            rev_lines.append(f"{r['output_id']},{r['case_id']},{r['scenario']},{cond},,,,,,,,")
    atomic_write_text(out_dir / "review_packet_03D.csv", "\n".join(rev_lines) + "\n")

    summary = {"protocol_signature": protocol_sig, "counts": counts}
    atomic_write_text(out_dir / "run_summary.json", json.dumps(summary, indent=2))
    return summary
