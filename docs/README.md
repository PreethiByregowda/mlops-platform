# Docs

## Architecture diagram

`images/architecture.svg` is the diagram source: hand-written SVG with a fixed layout grid (documented in the comment at the top of the file). Edit it directly; it renders on GitHub as-is.

`images/architecture.png`, the image embedded in the README, is rendered from it with headless Chromium. Regenerate it from the repository root after editing the SVG:

```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/repo -w /repo \
  mcr.microsoft.com/playwright/python:v1.48.0-jammy bash -c \
  "pip install -q --user playwright==1.48.0 && python docs/render_diagram.py images/architecture.svg images/architecture.png"
```

The diagram stays at the level of components and their relationships. Implementation detail (files, ports, flow names, metrics) belongs in the README text.
