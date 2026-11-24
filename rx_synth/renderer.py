#!/usr/bin/env python3


from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from PIL import Image, ImageDraw
from playwright.async_api import BrowserContext, Page, async_playwright

# Always groups rendered word tokens by visual baseline so wrapped inline-block
# spans (.salt-line, .brand-line) split cleanly into separate line-level boxes.
EXTRACT_BOXES_JS = r"""
() => {
  const pageEl = document.querySelector('.page') || document.body;
  const pageRect = pageEl.getBoundingClientRect();
  const originX = pageRect.left;
  const originY = pageRect.top;

  function extractLineBoxes(el) {
    const fullText = (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
    if (!fullText) return [];

    const lines = [];
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
    let node;
    while ((node = walker.nextNode())) {
      const text = node.nodeValue || '';
      const tokenRegex = /\S+/g;
      let match;
      while ((match = tokenRegex.exec(text)) !== null) {
        const range = document.createRange();
        range.setStart(node, match.index);
        range.setEnd(node, match.index + match[0].length);
        const r = range.getBoundingClientRect();
        if (r.width < 0.5 || r.height < 0.5) continue;

        const x1 = r.left - originX;
        const y1 = r.top - originY;
        const x2 = r.right - originX;
        const y2 = r.bottom - originY;
        const midY = (y1 + y2) / 2;

        let targetLine = null;
        for (const ln of lines) {
          const lnMidY = (ln.y1 + ln.y2) / 2;
          const tol = Math.max(5, (ln.y2 - ln.y1) * 0.55);
          if (Math.abs(midY - lnMidY) <= tol) {
            targetLine = ln;
            break;
          }
        }
        if (!targetLine) {
          lines.push({ words: [match[0]], x1, y1, x2, y2 });
        } else {
          targetLine.words.push(match[0]);
          targetLine.x1 = Math.min(targetLine.x1, x1);
          targetLine.y1 = Math.min(targetLine.y1, y1);
          targetLine.x2 = Math.max(targetLine.x2, x2);
          targetLine.y2 = Math.max(targetLine.y2, y2);
        }
      }
    }

    if (lines.length === 0) {
      const r = el.getBoundingClientRect();
      if (r.width >= 2 && r.height >= 2) {
        return [{
          text: fullText,
          x1: r.left - originX,
          y1: r.top - originY,
          x2: r.right - originX,
          y2: r.bottom - originY,
        }];
      }
      return [];
    }

    return lines.map(ln => ({
      text: ln.words.join(' '),
      x1: ln.x1,
      y1: ln.y1,
      x2: ln.x2,
      y2: ln.y2,
    }));
  }

  const boxes = [];
  const taggedEls = pageEl.querySelectorAll('[data-ocr-tag]');
  for (const el of taggedEls) {
    const tag = el.getAttribute('data-ocr-tag');
    const rowAttr = el.getAttribute('data-row-anchor') || el.getAttribute('data-row-index');
    const rowIdx = rowAttr ? parseInt(rowAttr, 10) : null;
    const lineBoxes = extractLineBoxes(el);
    for (const lb of lineBoxes) {
      boxes.push({
        transcription: lb.text,
        tag: tag,
        row_index: rowIdx,
        bbox: [lb.x1, lb.y1, lb.x2, lb.y2],
      });
    }
  }

  const rowAnchors = [];
  const anchorEls = pageEl.querySelectorAll('[data-row-anchor]');
  for (const el of anchorEls) {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    rowAnchors.push({
      row_index: parseInt(el.getAttribute('data-row-anchor'), 10),
      bbox: [
        r.left - originX,
        r.top - originY,
        r.right - originX,
        r.bottom - originY,
      ],
    });
  }

  return {
    page_width: Math.round(pageRect.width),
    page_height: Math.round(pageRect.height),
    boxes: boxes,
    row_anchors: rowAnchors,
  };
}
"""

TAG_COLORS = {
    "BRAND": (220, 38, 38),
    "SALT": (37, 99, 235),
    "DOSE": (22, 163, 74),
    "FREQ": (22, 163, 74),
    "INSTR": (22, 163, 74),
    "DURATION": (22, 163, 74),
    "QTY": (22, 163, 74),
    "FREQ_NOTE": (147, 51, 234),
    "NOISE": (180, 180, 180),
}


def clamp_box_to_quad(
    bbox: list[float], width: int, height: int, pad: int = 2
) -> list[list[int]] | None:
    x1 = max(0, min(width - 1, int(round(bbox[0])) - pad))
    y1 = max(0, min(height - 1, int(round(bbox[1])) - pad))
    x2 = max(0, min(width - 1, int(round(bbox[2])) + pad))
    y2 = max(0, min(height - 1, int(round(bbox[3])) + pad))
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def save_debug_overlay(png_path: Path, record: dict[str, Any], out_path: Path) -> None:
    img = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in record["boxes"]:
        pts = b["points"]
        color = TAG_COLORS.get(b["tag"], (245, 158, 11))
        poly = [tuple(p) for p in pts] + [tuple(pts[0])]
        draw.line(poly, fill=color, width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def load_completed_ids(raw_boxes_path: Path, raw_renders_dir: Path) -> set[str]:
    done: set[str] = set()
    if not raw_boxes_path.exists():
        return done
    with raw_boxes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                img_id = rec.get("image_id")
                if img_id and (raw_renders_dir / f"{img_id}.png").exists():
                    done.add(img_id)
            except json.JSONDecodeError:
                continue
    return done


async def render_single(
    page: Page,
    payload: dict[str, Any],
    jinja_env: Environment,
    raw_renders_dir: Path,
) -> dict[str, Any]:
    template_id = payload["template_id"]
    image_id = payload["image_id"]

    vp_w, vp_h = (1120, 1580) if template_id == "T3_compact_a5" else (1240, 1754)
    current_vp = page.viewport_size
    if not current_vp or current_vp["width"] != vp_w or current_vp["height"] != vp_h:
        await page.set_viewport_size({"width": vp_w, "height": vp_h})

    html = jinja_env.get_template(f"{template_id}.html").render(**payload)
    await page.set_content(html, wait_until="domcontentloaded")
    await page.evaluate("() => document.fonts.ready")

    raw_extract = await page.evaluate(EXTRACT_BOXES_JS)

    out_png = raw_renders_dir / f"{image_id}.png"
    page_loc = page.locator(".page")
    await page_loc.screenshot(path=str(out_png), type="png")

    with Image.open(out_png) as im:
        img_w, img_h = im.size

    clean_boxes: list[dict[str, Any]] = []
    for item in raw_extract["boxes"]:
        quad = clamp_box_to_quad(item["bbox"], img_w, img_h, pad=2)
        if quad is None:
            continue
        clean_boxes.append({
            "transcription": item["transcription"],
            "points": quad,
            "tag": item["tag"],
            "row_index": item["row_index"],
        })

    clean_anchors: list[dict[str, Any]] = []
    for anc in raw_extract["row_anchors"]:
        bx = anc["bbox"]
        clean_anchors.append({
            "row_index": anc["row_index"],
            "bbox": [
                max(0, int(round(bx[0]))),
                max(0, int(round(bx[1]))),
                min(img_w - 1, int(round(bx[2]))),
                min(img_h - 1, int(round(bx[3]))),
            ],
        })

    return {
        "image_id": image_id,
        "raw_image_path": f"raw_renders/{image_id}.png",
        "file_name": payload["file_name"],
        "split": payload["split"],
        "template_id": template_id,
        "language": payload["language"],
        "difficulty": payload["difficulty"],
        "is_hard_negative": payload["is_hard_negative"],
        "width": img_w,
        "height": img_h,
        "boxes": clean_boxes,
        "row_anchors": clean_anchors,
    }


async def worker_loop(
    worker_id: int,
    context: BrowserContext,
    job_queue: asyncio.Queue[dict[str, Any] | None],
    result_queue: asyncio.Queue[dict[str, Any] | None],
    jinja_env: Environment,
    raw_renders_dir: Path,
    recycle_every: int,
) -> None:
    page = await context.new_page()
    rendered_on_page = 0
    try:
        while True:
            payload = await job_queue.get()
            if payload is None:
                job_queue.task_done()
                break

            try:
                record = await render_single(page, payload, jinja_env, raw_renders_dir)
                await result_queue.put(record)
            except Exception as exc:
                await result_queue.put({
                    "error": f"[Worker {worker_id}] Failed {payload.get('image_id')}: {exc}"
                })
            finally:
                job_queue.task_done()

            rendered_on_page += 1
            if rendered_on_page >= recycle_every:
                await page.close()
                page = await context.new_page()
                rendered_on_page = 0
    finally:
        if not page.is_closed():
            await page.close()


async def run_pipeline(args: argparse.Namespace) -> None:
    payloads_path: Path = args.out_dir / "manifests" / "payloads.jsonl"
    raw_boxes_path: Path = args.out_dir / "manifests" / "raw_boxes.jsonl"
    raw_renders_dir: Path = args.out_dir / "raw_renders"
    debug_vis_dir: Path = args.out_dir / "debug_boxes"

    if not payloads_path.exists():
        raise FileNotFoundError(f"Missing {payloads_path}. Run Part 1 (sampler.py) first.")
    if not args.templates_dir.exists():
        raise FileNotFoundError(f"Missing {args.templates_dir}. Run Part 2 (setup_templates.py) first.")

    raw_renders_dir.mkdir(parents=True, exist_ok=True)
    jinja_env = Environment(loader=FileSystemLoader(str(args.templates_dir)), autoescape=True)

    payloads: list[dict[str, Any]] = []
    with payloads_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                payloads.append(json.loads(line))
            if args.limit and len(payloads) >= args.limit:
                break

    completed_ids = load_completed_ids(raw_boxes_path, raw_renders_dir) if args.resume else set()
    pending = [p for p in payloads if p["image_id"] not in completed_ids]

    if not pending:
        print(f"[INFO] All {len(payloads):,} requested images are already rendered.")
        return

    print(
        f"[INFO] Rendering {len(pending):,} images "
        f"(skipping {len(completed_ids):,} already done) with {args.workers} Playwright tabs..."
    )

    job_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    result_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    for p in pending:
        job_queue.put_nowait(p)
    for _ in range(args.workers):
        job_queue.put_nowait(None)

    write_mode = "a" if (args.resume and raw_boxes_path.exists()) else "w"
    start_time = time.time()
    errors: list[str] = []
    written = 0
    vis_saved = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(device_scale_factor=1)

        workers = [
            asyncio.create_task(
                worker_loop(
                    wid, context, job_queue, result_queue,
                    jinja_env, raw_renders_dir, args.recycle_every,
                )
            )
            for wid in range(args.workers)
        ]

        with raw_boxes_path.open(write_mode, encoding="utf-8") as f_out:
            while written + len(errors) < len(pending):
                res = await result_queue.get()
                if res is None:
                    continue
                if "error" in res:
                    errors.append(res["error"])
                    print(f"[WARN] {res['error']}")
                    continue

                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                written += 1

                if vis_saved < args.debug_vis:
                    png_p = raw_renders_dir / f"{res['image_id']}.png"
                    save_debug_overlay(png_p, res, debug_vis_dir / f"vis_{res['image_id']}.png")
                    vis_saved += 1

                if written % args.log_every == 0 or written == len(pending):
                    elapsed = time.time() - start_time
                    rate = written / max(elapsed, 0.001)
                    remaining = (len(pending) - written) / max(rate, 0.001)
                    print(
                        f"  [{written:>4}/{len(pending)}] "
                        f"{rate:5.1f} img/s | elapsed {elapsed:5.1f}s | ETA {remaining:5.1f}s"
                    )

        await asyncio.gather(*workers)
        await context.close()
        await browser.close()

    total_time = time.time() - start_time
    print(f"\n[DONE] Rendered {written:,} PNGs in {total_time:.1f}s ({len(errors)} errors).")
    print(f"  Raw renders : {raw_renders_dir}")
    print(f"  Raw boxes   : {raw_boxes_path}")
    if vis_saved:
        print(f"  Debug boxes : {debug_vis_dir} ({vis_saved} overlays)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 3: Playwright Renderer + Exact BBox Extractor")
    parser.add_argument("--out-dir", type=Path, default=Path("synthetic_rx_dataset"))
    parser.add_argument("--templates-dir", type=Path, default=Path("rx_synth/templates"))
    parser.add_argument("--workers", type=int, default=3, help="Concurrent Chromium tabs (3 is optimal for 8 GB RAM)")
    parser.add_argument("--recycle-every", type=int, default=200, help="Recycle each tab after N renders to keep RAM flat")
    parser.add_argument("--limit", type=int, default=None, help="Render only the first N payloads (great for testing)")
    parser.add_argument("--debug-vis", type=int, default=0, help="Save N bounding-box overlay PNGs to debug_boxes/")
    parser.add_argument("--resume", action="store_true", help="Skip already-rendered images and append cleanly")
    parser.add_argument("--log-every", type=int, default=50, help="Print progress every N images")
    args = parser.parse_args()

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
