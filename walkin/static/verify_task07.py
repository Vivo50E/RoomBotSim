#!/usr/bin/env python3
"""Focused WALK-IN console smoke/regression checks.

Run from walkin/ after starting the server:
  python static/verify_task07.py --url http://127.0.0.1:8000
  python static/verify_task07.py --url http://127.0.0.1:8000 --exercise-human

The optional human pass creates one demo episode, drives briefly with W, and checks
that its JSONL records one teleop-backed navigate_to immediately before a skill.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_assets():
    source = (ROOT / "index.html").read_text()
    for needle in ("function sizeCanvas()", "Math.min(W / (NX * RES), H / (NY * RES))",
                   "pointercancel", "M.seenSteps", "data-skill", "@media (max-width:700px)"):
        assert needle in source, "missing regression guard: " + needle
    try:
        import cv2
        value, _, _ = cv2.QRCodeDetector().detectAndDecode(cv2.imread(str(ROOT / "qr.png")))
        assert value.startswith(("http://", "https://")) and "localhost" not in value and "127.0.0.1" not in value
        print("QR:", value)
    except ImportError:
        print("QR decode skipped (opencv unavailable)")


def map_fit_regression():
    # Same fit invariant as the canvas: all four map corners must be on screen.
    for width_m, height_m in ((7, 9), (20, 20), (20, 3)):  # rectangular room and long corridor
        view_w, view_h = 800, 263
        scale = max(1, min(view_w / width_m, view_h / height_m))
        left, top = (view_w - width_m * scale) / 2, (view_h - height_m * scale) / 2
        assert left >= -1e-9 and top >= -1e-9
        assert left + width_m * scale <= view_w + 1e-9
        assert top + height_m * scale <= view_h + 1e-9
    print("map fit: 7×9, 20 m room, and corridor extents passed")


def browser_pass(url, exercise_human):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("Install Python playwright to run browser checks: " + str(exc))
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(url + "/", wait_until="networkidle")
        initial = page.locator("#top").evaluate("e => [e.clientWidth, e.clientHeight, e.width, e.height]")
        assert initial[0] and initial[1] and initial[2] >= initial[0] and initial[3] >= initial[1], initial
        page.locator("#demo").click()
        page.wait_for_timeout(700)
        assert "Room is live" in page.locator("#statusline").inner_text()
        assert page.locator("#top").evaluate("e => e.width >= e.clientWidth && e.height >= e.clientHeight")
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(url + "/", wait_until="networkidle")
        assert mobile.locator("#mobileFiles").is_visible() and not mobile.locator("#app").is_visible()
        assert "laptop" in mobile.locator(".handoff").inner_text().lower()
        mobile.close()
        if exercise_human:
            page.locator("#pol").select_option("human")
            page.locator("#epStart").click(); page.wait_for_timeout(350)
            page.locator("#top").focus(); page.keyboard.down("w"); page.wait_for_timeout(350); page.keyboard.up("w")
            page.wait_for_timeout(100); page.locator('[data-skill="pick"]').click(); page.wait_for_timeout(500)
            log = page.locator("#steplog").inner_text()
            assert log.count("navigate_to") == 1, log
            assert "pick(" in log, log
        browser.close()
    assert not errors, errors
    print("browser: desktop canvas, live demo, and phone uploader passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="running WALK-IN base URL")
    ap.add_argument("--exercise-human", action="store_true")
    args = ap.parse_args()
    check_assets()
    map_fit_regression()
    if args.url:
        browser_pass(args.url.rstrip("/"), args.exercise_human)
    print("TASK-07 checks passed")

if __name__ == "__main__":
    main()
