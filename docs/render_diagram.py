"""Render an SVG diagram to a 2x PNG with headless Chromium.

Usage: python docs/render_diagram.py images/architecture.svg images/architecture.png
"""
import re
import sys

from playwright.sync_api import sync_playwright

src, out = sys.argv[1], sys.argv[2]
svg = open(src).read()
width = int(re.search(r'<svg[^>]*\swidth="(\d+)"', svg).group(1))
height = int(re.search(r'<svg[^>]*\sheight="(\d+)"', svg).group(1))

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
    page.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
    page.locator("svg").screenshot(path=out)
    browser.close()
