"""Build the single-file web app from its sources.

    python3 app/build.py        ->  docs/index.html  (served by GitHub Pages)

template.html holds the page and styles, logic.js the algorithms (the same
A*, Bayes and MDP as the delivery/ package), ui.js the map, animation and
controls. The result has no external dependencies except two Google Fonts.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "docs", "index.html")


def build():
    page = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    logic = open(os.path.join(HERE, "logic.js"), encoding="utf-8").read()
    ui = open(os.path.join(HERE, "ui.js"), encoding="utf-8").read()
    # logic.js also exports its functions for Node (used by check_app.py)
    logic = logic[:logic.index('if (typeof module !== "undefined")')].rstrip() + "\n"
    page = page.replace('style="background:url(#hatch);background:var(--wall-bg)"',
                        'style="background:repeating-linear-gradient(45deg,var(--wall-bg) 0 3px,var(--wall-line) 3px 5px)"')
    page = page.replace("/*LOGIC*/", logic).replace("/*UI*/", ui)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    return OUT


if __name__ == "__main__":
    print("built", build())
