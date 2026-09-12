#!/usr/bin/env python3
"""latex_to_html.py -- render ``report/report.tex`` (+ ``report/chapters/*.tex``) into a
single self-contained HTML book.

Why this file exists
--------------------
The canonical deliverable is ``report/report.tex`` compiled to ``report/report.pdf``.
This sandbox has no TeX engine (and none is installable), and the browser preview must
not depend on a CDN or on network access. So the book is rasterised locally instead:

  * math is typeset with matplotlib's mathtext and embedded as PNG data-URIs,
  * LaTeX structure (parts/sections/theorem environments/lists/floats/tabulars/verbatim)
    is translated to semantic HTML + CSS,
  * ``\\ref``/``\\eqref``/``\\autoref``/``\\cite`` are resolved by a two-pass build,
  * figures are referenced relative to ``report/`` (``figures/*.png``), which the build
    driver populates from ``results/figures/``.

It is a *preview* renderer, not a TeX implementation. Deliberate simplifications:
floats are placed inline where they are written, page breaking is ignored, ``\\part`` is
a banner, unsupported mathtext constructs are decomposed into HTML (cases/matrix/
underbrace/boxed) or fall back to escaped raw TeX (reported on stderr, never silent).

Usage
-----
    python3 report/latex_to_html.py [--out report/report.html] [--dpi 170] [--check-math]
"""
from __future__ import annotations

import argparse
import base64
import html as _html
import io
import os
import re
import sys

# ---------------------------------------------------------------------------
# layout constants (tuned so mathtext at MATH_PT sits on the baseline of the
# BODY_PX serif text around it)
# ---------------------------------------------------------------------------
BODY_PX = 17.0
MATH_PT = 13.0
DISPLAY_PT = 15.0
MATH_DPI = 170
PX_PER_PT = BODY_PX / 12.75  # 12.75pt == 17px == 1rem in the stylesheet


def esc(s: str) -> str:
    return _html.escape(s, quote=False)


# ---------------------------------------------------------------------------
# small LaTeX scanning helpers
# ---------------------------------------------------------------------------
def skip_space(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    return i


def read_group(s: str, i: int):
    """Read one LaTeX argument starting at ``i``. Returns ``(text, next_index)``."""
    i = skip_space(s, i)
    if i >= len(s):
        return "", i
    if s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            c = s[j]
            if c == "\\":
                j += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
            j += 1
        return s[i + 1:], len(s)
    m = re.match(r"\\[A-Za-z]+\*?|.", s[i:], re.S)
    if not m:
        return "", i
    return m.group(0), i + m.end()


def read_optional(s: str, i: int):
    """Read a ``[...]`` optional argument if present. Returns ``(text|None, next_i)``."""
    j = skip_space(s, i)
    if j < len(s) and s[j] == "[":
        depth, k = 0, j
        while k < len(s):
            if s[k] == "\\":
                k += 2
                continue
            if s[k] == "[":
                depth += 1
            elif s[k] == "]":
                depth -= 1
                if depth == 0:
                    return s[j + 1:k], k + 1
            k += 1
        return s[j + 1:], len(s)
    return None, i


def find_env(s: str, start: int, name: str):
    """``start`` points at ``\\begin{name}``. Returns ``(body_start, body_end, after)``."""
    opener = "\\begin{" + name + "}"
    i = start + len(opener)
    pat = re.compile(r"\\begin\{" + re.escape(name) + r"\}|\\end\{" + re.escape(name) + r"\}")
    depth = 1
    for m in pat.finditer(s, i):
        if m.group(0).startswith("\\begin"):
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return i, m.start(), m.end()
    return i, len(s), len(s)


PROTECTED_ENVS = ("cases", "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix",
                  "smallmatrix", "array", "aligned", "split", "gathered", "alignedat")
# environments that only ever exist *inside* math: the block scanner must leave them
# alone so the inline-math scanner sees the whole $...$ span intact
MATH_INNER_ENVS = ("cases", "dcases", "rcases", "matrix", "pmatrix", "bmatrix",
                   "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "aligned",
                   "alignedat", "split", "gathered")
_PROT_RE = re.compile(r"\\(begin|end)\{(" + "|".join(PROTECTED_ENVS) + r")\*?\}")


def split_rows(s: str):
    r"""Split on top-level ``\\`` (ignoring ``\\`` inside cases/matrix/aligned)."""
    parts, buf, i, depth, envdepth = [], [], 0, 0, 0
    n = len(s)
    while i < n:
        m = _PROT_RE.match(s, i)
        if m:
            envdepth += 1 if m.group(1) == "begin" else -1
            buf.append(m.group(0))
            i = m.end()
            continue
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "\\" and depth == 0 and envdepth == 0:
                parts.append("".join(buf))
                buf = []
                i += 2
                opt, j = read_optional(s, i)          # \\[4pt]
                i = j if opt is not None else i
                continue
            buf.append(s[i:i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        buf.append(c)
        i += 1
    parts.append("".join(buf))
    return parts


def split_cells(s: str, sep: str = "&"):
    """Split on top-level ``sep`` (not ``\\&``, not inside braces/environments)."""
    parts, buf, i, depth, envdepth = [], [], 0, 0, 0
    n = len(s)
    while i < n:
        m = _PROT_RE.match(s, i)
        if m:
            envdepth += 1 if m.group(1) == "begin" else -1
            buf.append(m.group(0))
            i = m.end()
            continue
        c = s[i]
        if c == "\\" and i + 1 < n:
            buf.append(s[i:i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == sep and depth == 0 and envdepth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    parts.append("".join(buf))
    return parts


def strip_comments(text: str) -> str:
    """Remove ``%`` comments (not ``\\%``), keeping verbatim blocks intact."""
    out, i, n = [], 0, len(text)
    verb = re.compile(r"\\begin\{(verbatim|Verbatim|lstlisting)\*?\}")
    while i < n:
        m = verb.match(text, i)
        if m:
            b, e, after = find_env(text, i, m.group(1))
            out.append(text[i:after])
            i = after
            continue
        c = text[i]
        if c == "\\" and i + 1 < n:
            out.append(text[i:i + 2])
            i += 2
            continue
        if c == "%":
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# macro expansion (\newcommand / \renewcommand / \def / \DeclareMathOperator)
# ---------------------------------------------------------------------------
class Macros:
    def __init__(self):
        self.simple = {}
        self.withargs = {}

    def load(self, text: str):
        for m in re.finditer(r"\\(?:re)?newcommand\s*\*?\s*\{?\\([A-Za-z]+)\}?\s*(\[\d+\])?", text):
            name = m.group(1)
            nargs = int(m.group(2)[1:-1]) if m.group(2) else 0
            body, _ = read_group(text, m.end())
            if nargs:
                self.withargs[name] = (nargs, body)
            else:
                self.simple[name] = body
        for m in re.finditer(r"\\DeclareMathOperator\s*\*?\s*\{\\([A-Za-z]+)\}", text):
            body, _ = read_group(text, m.end())
            self.simple.setdefault(m.group(1), r"\operatorname{%s}" % body.strip())
        for m in re.finditer(r"\\def\s*\\([A-Za-z]+)", text):
            body, _ = read_group(text, m.end())
            self.simple.setdefault(m.group(1), body)

    def expand(self, s: str, depth: int = 0) -> str:
        if depth > 10:
            return s
        out, i, changed = [], 0, False
        n = len(s)
        while i < n:
            if s[i] == "\\":
                m = re.match(r"\\([A-Za-z]+)", s[i:])
                if m:
                    name = m.group(1)
                    j = i + m.end()
                    if name in self.withargs:
                        nargs, body = self.withargs[name]
                        repl = body
                        for k in range(nargs, 0, -1):
                            arg, j = read_group(s, j)
                            repl = repl.replace("#%d" % k, "{" + arg + "}")
                        out.append(repl)
                        changed = True
                        i = j
                        continue
                    if name in self.simple:
                        out.append(self.simple[name])
                        changed = True
                        i = j
                        continue
                    out.append(m.group(0))
                    i = j
                    continue
                out.append(s[i:i + 2])
                i += 2
                continue
            out.append(s[i])
            i += 1
        res = "".join(out)
        return self.expand(res, depth + 1) if changed else res


# ---------------------------------------------------------------------------
# math rasterisation
# ---------------------------------------------------------------------------
class MathRenderer:
    """matplotlib mathtext -> PNG data-URI, with a translation layer for the
    constructs mathtext does not implement."""

    TRANS = [
        (r"\\hspace\s*\*?\{[^{}]*\}", ""),
        (r"\\(?:displaystyle|textstyle|scriptstyle|limits|nolimits|allowbreak)", ""),
        (r"\\[Bb]igg?[lr]?\s*", ""),
        (r"\\[td]frac", r"\\frac"),
        (r"\\le(?![A-Za-z])", r"\\leq"),
        (r"\\ge(?![A-Za-z])", r"\\geq"),
        (r"\\ne(?![A-Za-z])", r"\\neq"),
        (r"\\bm(?![A-Za-z])", r"\\mathbf"),
        (r"\\boldsymbol", r"\\mathbf"),
        (r"\\lvert|\\rvert", "|"),
        (r"\\lVert|\\rVert", r"\\Vert"),
        (r"\\stackrel\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\2^{\1}"),
        (r"\\overset\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\2^{\1}"),
        (r"\\underset\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\2_{\1}"),
        (r"\\xrightarrow\s*\{([^{}]*)\}", r"\\to^{\1}"),
        (r"\\xleftarrow\s*\{([^{}]*)\}", r"\\leftarrow^{\1}"),
        (r"\\xrightarrow", r"\\to"),
        (r"\\mbox|\\hbox", r"\\text"),
        (r"\\!", ""),
        (r"\\&", r"\\;"),
        (r"\\begin\{(?:aligned|split|gathered)\*?\}|\\end\{(?:aligned|split|gathered)\*?\}", ""),
        (r"\\mathopen|\\mathclose|\\mathrel|\\mathbin|\\mathord|\\mathpunct", ""),
        (r"\\texttt", r"\\mathrm"),
        (r"\\bmod", r"\\;\\mathrm{mod}\\;"),
        (r"\\pmod", r"\\;\\mathrm{mod}"),
        (r"\\\\(?![A-Za-z])", " "),          # leftover row separators
    ]

    def __init__(self, macros: Macros, dpi: int = MATH_DPI, enabled: bool = True,
                 math_dir: str = None, embed: bool = False):
        self.macros = macros
        self.dpi = dpi
        self.enabled = enabled
        self.math_dir = math_dir          # None => always inline as data-URI
        self.embed = embed
        self.cache = {}
        self.failures = []
        self.files = 0
        self._parser = None
        self._fig = None

    # -- lazy matplotlib import so --check-math can run without a backend -----
    def _init_mpl(self):
        if self._parser is not None:
            return True
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.figure import Figure
            from matplotlib.font_manager import FontProperties
            from matplotlib.mathtext import MathTextParser
            self._Figure, self._FontProperties = Figure, FontProperties
            self._parser = MathTextParser("path")
            self._fig = Figure(figsize=(1, 1))
            return True
        except Exception as e:                                     # pragma: no cover
            print("  [math] matplotlib unavailable (%s); math falls back to raw TeX" % e)
            self.enabled = False
            return False

    ONE_ARG = ("sqrt", "mathcal", "mathbf", "mathbb", "mathrm", "mathit", "mathsf",
               "mathtt", "hat", "widehat", "bar", "overline", "tilde", "widetilde",
               "vec", "dot", "ddot", "operatorname", "text", "boldsymbol", "bm",
               "underline", "check", "breve", "acute", "grave", "mathring", "boxed",
               "overset", "underset", "stackrel")
    TWO_ARG = ("frac", "tfrac", "dfrac", "cfrac", "binom", "tbinom")

    @staticmethod
    def _read_token(s: str, i: int):
        """One 'token': a control word (plus its braced argument, if any) or one char."""
        m = re.match(r"\\[A-Za-z]+", s[i:])
        if m:
            j = i + m.end()
            k = skip_space(s, j)
            if k < len(s) and s[k] == "{":
                arg, k2 = read_group(s, k)
                return m.group(0) + "{" + arg + "}", k2
            return m.group(0), j
        return (s[i], i + 1) if i < len(s) else ("", i)

    NO_RECURSE = ("text", "mbox", "hbox", "operatorname")

    def add_braces(self, s: str) -> str:
        """`\sqrt T` -> `\sqrt{T}`, `\frac12` -> `\frac{1}{2}`; recurses into groups
        (mathtext, unlike TeX, insists on braced arguments)."""
        out, i, n = [], 0, len(s)
        while i < n:
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if m:
                name, j = m.group(1), i + m.end()
                if name in self.ONE_ARG:
                    k = skip_space(s, j)
                    if k < n and s[k] not in "{[":
                        tok, k2 = self._read_token(s, k)
                        out.append("\\%s{%s}" % (name, tok))
                        i = k2
                        continue
                    if k < n and s[k] == "{":
                        arg, k2 = read_group(s, k)
                        inner = arg if name in self.NO_RECURSE else self.add_braces(arg)
                        out.append("\\%s{%s}" % (name, inner))
                        i = k2
                        continue
                    out.append(s[i:j]); i = j; continue
                if name in self.TWO_ARG:
                    k = skip_space(s, j)
                    args = []
                    for _ in range(2):
                        k = skip_space(s, k)
                        if k < n and s[k] == "{":
                            a, k = read_group(s, k)
                            args.append(self.add_braces(a))
                        elif k < n:
                            tok, k = self._read_token(s, k)
                            args.append(tok)
                        else:
                            args.append("")
                    out.append("\\%s{%s}{%s}" % (name, args[0], args[1]))
                    i = k
                    continue
                out.append(s[i:j]); i = j; continue
            if s[i] == "\\":                 # \{  \}  \_  \%  ... : keep as a pair
                out.append(s[i:i + 2]); i += 2; continue
            if s[i] == "{":
                arg, k = read_group(s, i)
                out.append("{" + self.add_braces(arg) + "}")
                i = k
                continue
            out.append(s[i]); i += 1
        return "".join(out)

    def translate(self, tex: str) -> str:
        s = re.sub(r"\\\\(?![A-Za-z])", " ", tex)      # leftover row separators
        s = re.sub(r"\\[ \t]+", r"\\;", s)             # control space -> thin space
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"(?:\\;)+$", "", s).strip()        # trailing spacing is noise
        s = re.sub(r"\\+$", "", s).strip()             # dangling control space
        for pat, rep in self.TRANS:
            s = re.sub(pat, rep, s)
        s = self.add_braces(s)
        for pat, rep in self.TRANS:          # macros introduced by add_braces
            s = re.sub(pat, rep, s)
        return s

    def img(self, tex: str, size: float = MATH_PT) -> str:
        """One mathtext image tag for ``tex`` (already macro-expanded, no $)."""
        if not self.enabled or not self._init_mpl():
            return self.fallback(tex)
        code_only = re.match(r"^\\s*\\texttt\\s*\\{(.*)\\}\\s*$", tex.strip(), re.S)
        if code_only:
            inner = code_only.group(1)
            for a, b in ((r"\\_", "_"), (r"\\{", "{"), (r"\\}", "}"), (r"\\%", "%"),
                         (r"\\&", "&"), (r"\\#", "#"), ("~", "\u00a0"), ("--", "\u2013")):
                inner = inner.replace(a, b)
            tag = "<code>%s</code>" % esc(inner)
            self.cache[tex.strip()] = tag
            return tag
        src = self.translate(tex).strip()
        if src in self.cache:
            return self.cache[src]
        prop = self._FontProperties(family="serif", size=size)
        expr = "$" + src + "$"
        scale = PX_PER_PT * (size / MATH_PT)
        try:
            w, h, d, _, _ = self._parser.parse(expr, 72, prop)
            if h <= 0:
                raise ValueError("empty math box")
            self._fig.clf()
            self._fig.set_size_inches(max(w, 1) / 72.0, max(h, 1) / 72.0)
            self._fig.text(0, d / h, expr, fontproperties=prop, color="#161616")
            buf = io.BytesIO()
            self._fig.savefig(buf, dpi=self.dpi, format="png", transparent=True,
                              metadata={"Software": None, "Creation Time": None})
            png = buf.getvalue()
            if self.embed or not self.math_dir:
                src_attr = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
            else:
                import hashlib
                key = hashlib.sha1((src + "|%.2f" % size).encode()).hexdigest()[:20]
                fn = os.path.join(self.math_dir, key + ".png")
                if not os.path.exists(fn):
                    os.makedirs(self.math_dir, exist_ok=True)
                    with open(fn, "wb") as fh:
                        fh.write(png)
                    self.files += 1
                src_attr = "math/%s.png" % key
            tag = ('<img class="math" src="%s" alt="%s" '
                   'style="width:%.2fpx;height:%.2fpx;vertical-align:%.2fpx">'
                   % (src_attr, esc(tex), w * scale, h * scale, -d * scale))
        except Exception as e:
            self.failures.append((tex, str(e).strip().splitlines()[-1][:110]))
            if len(self.failures) <= 40:
                pass
            tag = self.fallback(tex)
        self.cache[src] = tag
        return tag

    @staticmethod
    def fallback(tex: str) -> str:
        return '<span class="tex">%s</span>' % esc(tex.strip())


# ---------------------------------------------------------------------------
# the book
# ---------------------------------------------------------------------------
LIST_ENVS = ("itemize", "enumerate", "description", "compactitem", "compactenum", "asparaenum")
MATH_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*",
             "multline", "multline*", "displaymath", "eqnarray", "eqnarray*", "flalign",
             "flalign*")
FLOAT_ENVS = ("figure", "figure*", "table", "table*", "threeparttable", "sidewaystable")
THEOREM_ENVS = ("definition", "assumption", "theorem", "proposition", "lemma",
                "corollary", "remark", "example", "intuition", "observation", "claim")
TABLE_ENVS = ("tabular", "tabular*", "longtable", "array", "tabularx")
PASSTHRU_ENVS = ("center", "flushleft", "flushright", "raggedright", "minipage",
                 "subfigure", "multicols", "small", "footnotesize", "abstract",
                 "quote", "quotation", "spacing", "adjustwidth", "document",
                 "titlepage", "wrapfigure", "sloppypar", "onehalfspacing")
VERB_ENVS = ("verbatim", "Verbatim", "lstlisting", "minted")
IGNORE_ENVS = ("comment", "landscape")

ACCENTS = {
    "v": {"S": "\u0160", "s": "\u0161", "C": "\u010c", "c": "\u010d",
          "Z": "\u017d", "z": "\u017e", "r": "\u0159", "R": "\u0158",
          "e": "\u011b", "E": "\u011a", "n": "\u0148", "N": "\u0147",
          "d": "\u010f", "t": "\u0165", "a": "\u00e4", "o": "\u00f6",
          "u": "\u00fc"},
    "\'": {"o": "\u00f3", "O": "\u00d3", "a": "\u00e1", "A": "\u00c1",
            "e": "\u00e9", "E": "\u00c9", "i": "\u00ed", "I": "\u00cd",
            "u": "\u00fa", "U": "\u00da", "n": "\u00f1", "N": "\u00d1",
            "y": "\u00fd", "Y": "\u00dd", "c": "\u0107", "C": "\u0106",
            "s": "\u015b", "S": "\u015a", "z": "\u017a", "Z": "\u0179"},
    "`": {"a": "\u00e0", "A": "\u00c0", "e": "\u00e8", "E": "\u00c8",
          "o": "\u00f2", "O": "\u00d2", "u": "\u00f9", "U": "\u00d9"},
    "^": {"o": "\u00f4", "O": "\u00d4", "a": "\u00e2", "A": "\u00c2",
          "e": "\u00ea", "E": "\u00ca", "i": "\u00ee", "I": "\u00ce"},
    "\"": {"a": "\u00e4", "A": "\u00c4", "o": "\u00f6", "O": "\u00d6",
             "u": "\u00fc", "U": "\u00dc", "y": "\u00ff"},
    "~": {"n": "\u00f1", "N": "\u00d1", "a": "\u00e3", "o": "\u00f5",
          "A": "\u00c3", "O": "\u00d5"},
    "c": {"c": "\u00e7", "C": "\u00c7", "s": "\u015f", "S": "\u015e"},
    "u": {"a": "\u0103", "A": "\u0102", "g": "\u011f", "G": "\u011e"},
    "=": {"a": "\u0101", "A": "\u0100", "e": "\u0113", "o": "\u014d"},
    "r": {"a": "\u00e5", "A": "\u00c5", "u": "\u016f", "U": "\u016e"},
    "H": {"o": "\u0151", "O": "\u0150", "u": "\u0171", "U": "\u0170"},
    ".": {"z": "\u017c", "Z": "\u017b", "e": "\u0117"},
    "k": {"a": "\u0105", "e": "\u0119", "A": "\u0104", "E": "\u0118"},
}
TEXT_SYMBOLS = {
    "textbackslash": "\\", "TeX": "TeX", "LaTeX": "LaTeX", "BibTeX": "BibTeX",
    "S": "\u00a7", "P": "\u00b6", "dag": "\u2020",
    "ddag": "\u2021", "copyright": "\u00a9", "pounds": "\u00a3",
    "euro": "\u20ac", "yen": "\u00a5", "textregistered": "\u00ae",
    "texttrademark": "\u2122", "textellipsis": "\u2026", "textbullet": "\u2022",
    "textdegree": "\u00b0", "textpm": "\u00b1", "texttimes": "\u00d7",
    "textendash": "\u2013", "textemdash": "\u2014", "ss": "\u00df",
    "ae": "\u00e6", "AE": "\u00c6", "oe": "\u0153", "OE": "\u0152",
    "aa": "\u00e5", "AA": "\u00c5", "o": "\u00f8", "O": "\u00d8",
    "l": "\u0142", "L": "\u0141", "i": "\u0131",
}

AUTOREF = {
    "part": "Part", "section": "Section", "subsection": "Section",
    "subsubsection": "Section", "paragraph": "Section", "equation": "Equation",
    "figure": "Figure", "table": "Table", "definition": "Definition",
    "assumption": "Assumption", "theorem": "Theorem", "proposition": "Proposition",
    "lemma": "Lemma", "corollary": "Corollary", "remark": "Remark",
    "example": "Example", "intuition": "Intuition", "appendix": "Appendix",
}

ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def to_roman(k: int) -> str:
    if 0 < k < len(ROMAN):
        return ROMAN[k]
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
            (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, sym in vals:
        while k >= v:
            out += sym
            k -= v
    return out or "0"


def to_letter(k: int) -> str:
    return chr(ord("A") + (k - 1) % 26) if k > 0 else "A"


class Book:
    BLOCK_RE = re.compile(r"""
      (?P<env>\\begin\{[A-Za-z*]+\})
    | (?P<heading>\\(?:subsubsection|subparagraph|subsection|paragraph|section|part)\*?(?![A-Za-z]))
    | (?P<dispopen>(?<!\\)\\\[)
    | (?P<cmd>\\(?:label|newpage|clearpage|pagebreak|vspace|hspace|bigskip|medskip|smallskip|noindent|hrule|rule|maketitle|tableofcontents|listoffigures|listoftables|printbibliography|appendix|input|include|centering|item|caption|includegraphics|bibliography|bibliographystyle)\b)
    | (?P<blank>\n[ \t]*\n)
    """, re.X)

    def __init__(self, report_dir: str, dpi: int = MATH_DPI, embed_figures: bool = False,
                 embed_math: bool = False):
        self.dir = report_dir
        self.embed_figures = embed_figures
        self.macros = Macros()
        self.math = MathRenderer(self.macros, dpi=dpi,
                                 math_dir=None if embed_math
                                 else os.path.join(report_dir, "math"),
                                 embed=embed_math)
        self.dry = True
        self.reset_state()

    # -- state ---------------------------------------------------------------
    def reset_state(self):
        self.labels = {}          # label -> (kind, number, anchor)
        self.current = ("section", "", "")
        self.counters = {}
        self.appendix = False
        self.toc = []
        self.lof = []
        self.lot = []
        self.cited = []           # citation keys in order of first use
        self.collecting = True    # pass 1 collects TOC/float lists; pass 2 reuses them
        self.bib = {}
        self.title_html = ""
        self.author_html = ""
        self.date_html = ""
        self.anchors = 0

    def ctr(self, name: str) -> int:
        return self.counters.get(name, 0)

    def step(self, name: str, reset=()):
        self.counters[name] = self.counters.get(name, 0) + 1
        for r in reset:
            self.counters[r] = 0
        return self.counters[name]

    def new_anchor(self, hint: str = "a") -> str:
        self.anchors += 1
        return "ax-%s-%d" % (re.sub(r"\W+", "", hint)[:12], self.anchors)

    # -- loading -------------------------------------------------------------
    def load(self) -> str:
        root = os.path.join(self.dir, "report.tex")
        with open(root, encoding="utf-8") as f:
            raw = f.read()
        self.macros.load(raw)
        self.raw_root = raw
        body = self.expand_inputs(raw, depth=0)
        body = strip_comments(body)
        m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", body, re.S)
        if m:
            body = m.group(1)
        self.load_bib()
        return body

    def expand_inputs(self, text: str, depth: int) -> str:
        if depth > 6:
            return text
        out, i = [], 0
        while True:
            m = re.compile(r"\\input\s*\{([^}]*)\}|\\include\s*\{([^}]*)\}").search(text, i)
            if not m:
                out.append(text[i:])
                break
            out.append(text[i:m.start()])
            name = (m.group(1) or m.group(2)).strip()
            path = self.resolve(name)
            if path and os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    sub = f.read()
                self.macros.load(sub)
                out.append("\n%% ---- from %s ----\n" % os.path.relpath(path, self.dir))
                out.append(self.expand_inputs(sub, depth + 1))
            else:
                out.append("[missing input: %s]" % name)
            i = m.end()
        return "".join(out)

    def resolve(self, name: str):
        for cand in (name, name + ".tex"):
            p = os.path.join(self.dir, cand)
            if os.path.exists(p):
                return p
        return None

    def load_bib(self):
        path = os.path.join(self.dir, "references.bib")
        if not os.path.exists(path):
            return
        txt = open(path, encoding="utf-8").read()
        for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\s*\}", txt, re.S):
            typ, key, body = m.group(1).lower(), m.group(2), m.group(3)
            fields = {}
            for fm in re.finditer(r"(\w+)\s*=\s*(\{.*?\}|\d+)", body, re.S):
                fields[fm.group(1).lower()] = fm.group(2).strip("{}").replace("\n", " ")
            fields["_type"] = typ
            self.bib[key] = fields

    # -- references ----------------------------------------------------------
    def cite_key(self, key: str) -> int:
        if key not in self.cited:
            self.cited.append(key)
        return self.cited.index(key) + 1

    def fmt_bib(self, key: str) -> str:
        f = self.bib.get(key)
        if not f:
            return key
        auth = f.get("author", "").replace(" and ", "; ")
        auth = re.sub(r"[{}\\]", "", auth)
        year = f.get("year", "")
        title = re.sub(r"[{}\\]", "", f.get("title", ""))
        rest = f.get("journal") or f.get("booktitle") or f.get("publisher") or ""
        rest = re.sub(r"[{}\\]", "", rest)
        vol = f.get("volume", "")
        num = f.get("number", "")
        pages = re.sub(r"[{}\\]", "", f.get("pages", ""))
        bits = []
        if vol:
            bits.append("vol.&nbsp;%s" % esc(vol))
        if num:
            bits.append("no.&nbsp;%s" % esc(num))
        if pages:
            bits.append("pp.&nbsp;%s" % esc(pages.replace("--", "&ndash;")))
        tail = (", " + ", ".join(bits)) if bits else ""
        return "%s (%s). &ldquo;%s.&rdquo; <em>%s</em>%s." % (
            esc(auth), esc(year), esc(title), esc(rest), tail)

    def ref_html(self, key: str, kind: str = "ref") -> str:
        entry = self.labels.get(key)
        if entry is None:
            if self.dry:
                return "<span class='ref-unresolved'>%s</span>" % esc(key)
            return "<span class='ref-unresolved'>%s</span>" % esc(key)
        lkind, number, anchor = entry
        if kind == "eqref":
            text = "(%s)" % number
        elif kind == "autoref":
            text = "%s&nbsp;%s" % (AUTOREF.get(lkind, lkind.capitalize()), number)
        else:
            text = number
        href = ("#%s" % anchor) if anchor else "#"
        return '<a class="ref" href="%s">%s</a>' % (href, text)

    # -- inline --------------------------------------------------------------
    INLINE_CMD = re.compile(r"\\([A-Za-z]+\*?|.)", re.S)

    def render_inline(self, s: str) -> str:
        out, i, n, text = [], 0, len(s), []

        def flush():
            if not text:
                return
            t = "".join(text)
            text.clear()
            t = esc(t)
            t = t.replace("---", "&mdash;").replace("--", "&ndash;")
            t = t.replace("``", "\u201c").replace("''", "\u201d")
            t = t.replace("`", "\u2018").replace("'", "\u2019")
            out.append(t)

        while i < n:
            c = s[i]
            if c == "\\" and i + 1 < n:
                if s[i + 1] == "\\":
                    flush()
                    j = i + 2
                    _opt, j = read_optional(s, j)
                    out.append("<br>")
                    i = j
                    continue
                m = self.INLINE_CMD.match(s, i)
                name = m.group(1)
                j = m.end()
                handled = self.inline_command(name, s, j, out, flush)
                if handled is not None:
                    i = handled
                    continue
                # unknown command: drop the name, keep going
                if name not in (",", ";", ":", "!", " ", "~"):
                    self.note_unknown(name, s[max(0, i - 70):i + 70])
                i = j
                continue
            if c == "$":
                flush()
                i = self.inline_math(s, i, out)
                continue
            if c == "~":
                text.append("\u00a0")
                i += 1
                continue
            if c in "{}":
                i += 1
                continue
            text.append(c)
            i += 1
        flush()
        return "".join(out)

    _unknown = {}

    def note_unknown(self, name: str, ctx: str = ""):
        if name not in Book._unknown:
            Book._unknown[name] = re.sub(r"\s+", " ", ctx)[:110]

    def inline_command(self, name, s, j, out, flush):
        """Return new index if handled, else None."""
        two_arg = {
            "textbf": "strong", "bf": "strong", "textit": "em", "emph": "em",
            "texttt": "code", "code": "code", "textsc": "sc", "underline": "u",
            "text": "rm", "mbox": "rm", "textnormal": "span",
        }
        if name in two_arg:
            arg, k = read_group(s, j)
            flush()
            tag = two_arg[name]
            inner = self.render_inline(arg)
            out.append("<%s>%s</%s>" % (tag, inner, tag) if tag not in ("sc", "rm", "span")
                       else '<span class="%s">%s</span>' % (tag, inner))
            return k
        if name in ("ref", "eqref", "autoref", "Cref", "cref", "pageref"):
            arg, k = read_group(s, j)
            flush()
            kind = "eqref" if name == "eqref" else ("autoref" if name in ("autoref", "Cref", "cref") else "ref")
            out.append(self.ref_html(arg.strip(), kind))
            return k
        if name in ("cite", "citep", "citet", "parencite", "textcite", "autocite"):
            _opt, j2 = read_optional(s, j)
            arg, k = read_group(s, j2)
            flush()
            keys = [x.strip() for x in arg.split(",") if x.strip()]
            nums = [self.cite_key(x) for x in keys]
            links = ", ".join('<a class="ref" href="#bib-%s">%d</a>' % (esc(x), num)
                              for x, num in zip(keys, nums))
            out.append("[%s]" % links)
            return k
        if name == "label":
            arg, k = read_group(s, j)
            flush()
            self.set_label(arg.strip())
            return k
        if name == "footnote":
            arg, k = read_group(s, j)
            flush()
            out.append('<sup class="fn">%s</sup>' % self.render_inline(arg))
            return k
        if name == "includegraphics":
            opt, j2 = read_optional(s, j)
            arg, k = read_group(s, j2)
            flush()
            out.append(self.image_html(arg.strip(), opt))
            return k
        if name in ACCENTS:
            arg, k = read_group(s, j)
            flush()
            out.append(ACCENTS[name].get(arg.strip(), arg.strip()))
            return k
        if name in TEXT_SYMBOLS:
            flush(); out.append(TEXT_SYMBOLS[name]); return j
        if name in ("addcontentsline", "index"):
            _a, k = read_group(s, j)
            _b, k = read_group(s, k)
            _c, k = read_group(s, k)
            flush(); return k
        if name in ("renewcommand", "newcommand", "providecommand", "def", "let",
                    "DeclareMathOperator", "newenvironment", "renewenvironment"):
            k = j
            if name in ("newcommand", "renewcommand", "providecommand",
                        "newenvironment", "renewenvironment"):
                _a, k = read_group(s, k)
                _opt, k2 = read_optional(s, k)
                if _opt is not None:
                    k = k2
            _b, k = read_group(s, k)
            if name in ("newenvironment", "renewenvironment"):
                _c, k = read_group(s, k)
            flush(); return k
        if name in ("setlength", "addtolength", "settowidth", "setcounter",
                    "addtocounter", "counterwithin"):
            _a, k = read_group(s, j)
            _b, k = read_group(s, k)
            flush(); return k
        if name == "checkmark":
            flush(); out.append('<span class="ok">&#10003;</span>'); return j
        if name in ("dots", "ldots"):
            flush(); out.append("&hellip;"); return j
        if name == "cdots":
            flush(); out.append("&middot;&middot;&middot;"); return j
        if name == "%":
            flush(); out.append("%"); return j
        if name in ("&", "_", "$", "#", "{", "}"):
            flush(); out.append(esc(name)); return j
        if name.strip() == "":                 # \ followed by a space/newline
            flush(); out.append("&nbsp;"); return j
        if name in (" ", ",", ";", ":", "!", "~", "quad", "qquad"):
            flush()
            out.append("&thinsp;" if name in ("quad", "qquad") else ("&nbsp;" if name == "~" else " "))
            return j
        if name in ("textsuperscript",):
            arg, k = read_group(s, j); flush(); out.append("<sup>%s</sup>" % self.render_inline(arg)); return k
        if name == "textsubscript":
            arg, k = read_group(s, j); flush(); out.append("<sub>%s</sub>" % self.render_inline(arg)); return k
        if name in ("url", "href"):
            arg, k = read_group(s, j)
            if name == "href":
                arg2, k = read_group(s, k)
                flush(); out.append('<a href="%s">%s</a>' % (esc(arg), self.render_inline(arg2)))
            else:
                flush(); out.append('<a href="%s"><code>%s</code></a>' % (esc(arg), esc(arg)))
            return k
        if name in ("item", "hfill", "hfil", "noindent", "par", "smallskip", "medskip",
                    "bigskip", "hspace", "vspace", "rule", "centering", "raggedright",
                    "renewcommand", "setlength", "normalfont", "bfseries", "itshape",
                    "small", "large", "Large", "normalsize", "footnotesize", "scriptsize",
                    "ttfamily", "scshape", "quad", "qquad", "hspace*", "vspace*",
                    "phantom", "hphantom", "vphantom", "kern", "hskip", "unskip",
                    "displaystyle", "protect", "clearpage", "newpage"):
            return j
        return None

    def inline_math(self, s: str, i: int, out: list) -> int:
        """Handle ``$...$`` / ``$$...$$`` starting at index ``i``."""
        if s.startswith("$$", i):
            k = s.find("$$", i + 2)
            if k < 0:
                k = len(s)
            body = s[i + 2:k]
            out.append(self.display_math_html(body, numbered=False, number=""))
            return k + 2
        j = i + 1
        buf = []
        while j < len(s):
            if s[j] == "\\" and j + 1 < len(s):
                buf.append(s[j:j + 2]); j += 2; continue
            if s[j] == "$":
                break
            buf.append(s[j]); j += 1
        tex = "".join(buf)
        expanded = self.macros.expand(tex)
        if self.dry:
            out.append('<span class="mathph"></span>')
        else:
            out.append(self.math_html(expanded, MATH_PT))
        return min(j + 1, len(s))

    # -- math composition ----------------------------------------------------
    SPECIAL = re.compile(r"\\underbrace|\\overbrace|\\boxed|\\begin\{(cases|bmatrix|pmatrix|matrix|Bmatrix|vmatrix|smallmatrix|array)\}")

    def math_html(self, tex: str, size: float) -> str:
        tex = tex.strip()
        if not tex:
            return ""
        m = self.SPECIAL.search(tex)
        if not m:
            if self.dry:
                return '<span class="mathph"></span>'
            return self.math.img(tex, size)
        pre, post = tex[:m.start()], tex[m.end():]
        kind = m.group(0)
        if kind.startswith("\\underbrace") or kind.startswith("\\overbrace"):
            arg, k = read_group(tex, m.end())
            sub, k2 = None, k
            k2 = skip_space(tex, k)
            if k2 < len(tex) and tex[k2] == "_":
                sub, k = read_group(tex, k2 + 1)
            elif k2 < len(tex) and tex[k2] == "^":
                sub, k = read_group(tex, k2 + 1)
            post = tex[k:]
            return (self.math_html(pre, size)
                    + self.brace_stack(arg, sub, size, over=kind.startswith("\\overbrace"))
                    + self.math_html(post, size))
        if kind.startswith("\\boxed"):
            arg, k = read_group(tex, m.end())
            post = tex[k:]
            return (self.math_html(pre, size)
                    + '<span class="boxed">%s</span>' % self.math_html(arg, size)
                    + self.math_html(post, size))
        # cases / matrix environments
        envname = m.group(1)
        b, e, after = find_env(tex, m.start(), envname)
        inner, post = tex[b:e], tex[after:]
        return (self.math_html(pre, size)
                + self.env_matrix_html(envname, inner, size)
                + self.math_html(post, size))

    def math_pieces(self, tex: str):
        """Yield the atomic mathtext strings a math expression decomposes into
        (cases/matrix rows and cells, underbrace parts, boxed content, plain math)."""
        tex = (tex or "").strip()
        if not tex:
            return
        m = self.SPECIAL.search(tex)
        if not m:
            yield tex
            return
        pre, kind = tex[:m.start()], m.group(0)
        if kind.startswith("\\underbrace") or kind.startswith("\\overbrace"):
            arg, k = read_group(tex, m.end())
            sub, k = None, skip_space(tex, k)
            if k < len(tex) and tex[k] in "_^":
                sub, k = read_group(tex, k + 1)
            post = tex[k:]
            for part in (pre, arg, sub, post):
                if part:
                    yield from self.math_pieces(part)
            return
        if kind.startswith("\\boxed"):
            arg, k = read_group(tex, m.end())
            for part in (pre, arg, tex[k:]):
                if part:
                    yield from self.math_pieces(part)
            return
        env = m.group(1)
        _b, _e, after = find_env(tex, m.start(), env)
        inner, post = tex[_b:_e], tex[after:]
        for part in (pre, post):
            if part:
                yield from self.math_pieces(part)
        for row in split_rows(inner):
            for cell in split_cells(row):
                if cell.strip():
                    yield from self.math_pieces(cell)

    def brace_stack(self, top: str, sub, size: float, over: bool = False) -> str:
        t = self.math_html(top, size)
        b = self.math_html(sub, size * 0.85) if sub else ""
        char = "&#9182;" if over else "&#9183;"
        cls = "obrace" if over else "ubrace"
        return ('<span class="%s"><span class="ub-main">%s</span>'
                '<span class="ub-mark">%s</span><span class="ub-sub">%s</span></span>'
                % (cls, t, char, b))

    def env_matrix_html(self, env: str, inner: str, size: float) -> str:
        rows = split_rows(inner)
        cls = {"cases": "cases", "bmatrix": "bmat", "Bmatrix": "bmat",
               "pmatrix": "pmat", "matrix": "nomat", "vmatrix": "vmat",
               "smallmatrix": "nomat", "array": "nomat"}.get(env, "nomat")
        html_rows = []
        for r in rows:
            if not r.strip():
                continue
            cells = split_cells(r)
            tds = "".join('<td class="mcell%s">%s</td>'
                          % (" cond" if (cls == "cases" and ci > 0) else "",
                             self.math_html(c, size * 0.97))
                          for ci, c in enumerate(cells))
            html_rows.append("<tr>%s</tr>" % tds)
        brace = ""
        if cls == "cases":
            brace = '<span class="cases-brace">{</span>'
        return ('<span class="menv %s">%s<table class="mtab">%s</table>%s</span>'
                % (cls, brace, "".join(html_rows),
                   '<span class="cases-brace right">}</span>' if cls in ("bmat", "pmat") else ""))

    # -- display math --------------------------------------------------------
    def display_math_html(self, body: str, numbered: bool, number: str,
                          env: str = "equation") -> str:
        body = self.macros.expand(body)
        rows = split_rows(body)
        html_rows = []
        any_number = False
        for r in rows:
            r = r.strip()
            if not r:
                continue
            labels = re.findall(r"\\label\s*\{([^}]*)\}", r)
            r = re.sub(r"\\label\s*\{[^}]*\}", "", r)
            nonumber = bool(re.search(r"\\(?:nonumber|notag)\b", r))
            r = re.sub(r"\\(?:nonumber|notag)\b", "", r)
            cells = split_cells(r)
            tds = ""
            if len(cells) == 1:
                tds = '<td class="eq-full">%s</td>' % self.math_html(cells[0], DISPLAY_PT)
            else:
                for ci, cell in enumerate(cells):
                    align = "eq-right" if ci % 2 == 0 else "eq-left"
                    tds += '<td class="%s">%s</td>' % (align, self.math_html(cell, DISPLAY_PT))
            num_cell = ""
            show_no = numbered and not nonumber and number
            if show_no:
                num_cell = '<td class="eq-no">(%s)</td>' % esc(number)
                any_number = True
            anchor = labels[0] if labels else ""
            for lb in labels:
                self.set_label(lb, kind="equation", number=number, anchor=anchor)
            html_rows.append('<tr%s>%s%s</tr>'
                             % (' id="%s"' % esc(anchor) if anchor else "", tds, num_cell))
        if not html_rows:
            return ""
        return ('<div class="eqn"><table class="eqtab">%s</table></div>'
                % "".join(html_rows))

    # -- block level ---------------------------------------------------------
    def render_blocks(self, s: str) -> str:
        out, buf, pos = [], [], 0
        n = len(s)

        def flush():
            txt = "".join(buf)
            buf.clear()
            if txt.strip():
                out.append("<p>%s</p>" % self.render_inline(txt))

        while pos < n:
            m = self.BLOCK_RE.search(s, pos)
            if not m:
                buf.append(s[pos:])
                break
            buf.append(s[pos:m.start()])
            if m.group("blank"):
                flush()
                pos = m.end()
                continue
            keep = self.peek_env(s, m)
            if keep is not None:                 # math-internal env: not a block
                buf.append(s[m.start():keep])
                pos = keep
                continue
            flush()
            pos = self.dispatch_block(s, m, out, pos)
        flush()
        return "\n".join(out)

    @staticmethod
    def peek_env(s, m):
        """If this token is a math-internal environment, return the index just after
        its \end{...}; the block scanner then leaves it inside the paragraph text."""
        if not m.group("env"):
            return None
        name = re.match(r"\\begin\{([A-Za-z*]+)\}", m.group("env")).group(1)
        if name not in MATH_INNER_ENVS:
            return None
        _b, _e, after = find_env(s, m.start(), name)
        return after

    def dispatch_block(self, s, m, out, pos) -> int:
        if m.group("env"):
            name = re.match(r"\\begin\{([A-Za-z*]+)\}", m.group("env")).group(1)
            b, e, after = find_env(s, m.start(), name)
            if name in MATH_INNER_ENVS:
                return None          # leave inside the paragraph for the math scanner
            opt, b2 = read_optional(s, b)
            self.handle_env(name, s[b2 if opt is not None else b:e], out, opt)
            return after
        if m.group("heading"):
            kind = re.match(r"\\([a-z]+)(\*?)$", m.group("heading")).groups()
            title, k = read_group(s, m.end())
            self.handle_heading(kind[0], bool(kind[1]), title, out)
            return k
        if m.group("dispopen"):
            mm = re.compile(r"(?<!\\)\\]").search(s, m.end())
            k = mm.start() if mm else len(s)
            out.append(self.display_math_html(s[m.end():k], numbered=False, number=""))
            return k + 2
        cmd = m.group("cmd")
        name = re.match(r"\\([A-Za-z]+)", cmd).group(1)
        return self.handle_block_cmd(name, s, m.end(), out)

    def handle_block_cmd(self, name, s, j, out) -> int:
        if name == "label":
            arg, k = read_group(s, j)
            self.set_label(arg.strip())
            return k
        if name == "input" or name == "include":
            arg, k = read_group(s, j)
            path = self.resolve(arg.strip())
            if path and os.path.exists(path):
                txt = strip_comments(open(path, encoding="utf-8").read())
                out.append(self.render_blocks(txt))
            return k
        if name == "caption":
            arg, k = read_group(s, j)
            out.append('<div class="caption">%s</div>' % self.render_inline(arg))
            return k
        if name == "includegraphics":
            opt, j2 = read_optional(s, j)
            arg, k = read_group(s, j2)
            out.append('<div class="imgline">%s</div>' % self.image_html(arg.strip(), opt))
            return k
        if name == "item":
            out.append("<br>")
            return j
        if name in ("maketitle",):
            out.append(self.title_page_html())
            return j
        if name == "tableofcontents":
            out.append("<!--TOC-->")
            return j
        if name == "listoffigures":
            out.append("<!--LOF-->")
            return j
        if name == "listoftables":
            out.append("<!--LOT-->")
            return j
        if name in ("printbibliography", "bibliography"):
            _opt, j2 = read_optional(s, j)
            if name == "printbibliography":
                out.append("<!--BIB-->")
                return j2
            _arg, k = read_group(s, j2)
            out.append("<!--BIB-->")
            return k
        if name == "appendix":
            self.appendix = True
            self.counters["section"] = 0
            out.append('<div class="appendix-start"></div>')
            return j
        if name in ("rule", "hrule"):
            out.append("<hr>")
            _opt, j2 = read_optional(s, j)
            _a, k = read_group(s, j2)
            _b, k = read_group(s, k)
            return k
        if name in ("vspace", "hspace"):
            _a, k = read_group(s, j)
            return k
        if name in ("newpage", "clearpage", "pagebreak"):
            out.append('<hr class="pagebreak">')
            return j
        return j

    def handle_heading(self, kind, starred, title, out):
        title_html = self.render_inline(title)
        title_plain = title
        anchor = self.new_anchor(kind)
        if kind == "part":
            num = to_roman(self.step("part"))
            self.current = ("part", num, anchor)
            out.append('<div class="part" id="%s"><div class="kicker">Part %s</div>'
                       '<div class="t">%s</div></div>' % (anchor, num, title_html))
            if self.collecting:
                self.toc.append((0, "Part %s" % num, title_plain, anchor))
            return
        if kind == "section":
            num = ""
            if not starred:
                k = self.step("section", reset=("subsection", "subsubsection"))
                num = to_letter(k) if self.appendix else str(k)
            for t in THEOREM_ENVS:
                self.counters[t] = 0
            self.current = ("appendix" if self.appendix else "section", num, anchor)
            out.append('<h2 class="sec" id="%s"><span class="no">%s</span>%s</h2>'
                       % (anchor, num if starred is False else "", title_html))
            if self.collecting:
                self.toc.append((1, "" if starred else num, title_plain, anchor))
            return
        if kind == "subsection":
            sec = self.sec_number()
            num = "" if starred else "%s.%d" % (
                sec, self.step("subsection", reset=("subsubsection",)))
            self.current = ("subsection", num, anchor)
            out.append('<h3 id="%s"><span class="no">%s</span>%s</h3>'
                       % (anchor, "" if starred else num, title_html))
            if self.collecting and not starred:
                self.toc.append((2, num, title_plain, anchor))
            return
        if kind == "subsubsection":
            sec = self.sec_number()
            num = "" if starred else "%s.%d.%d" % (
                sec, self.ctr("subsection"), self.step("subsubsection"))
            self.current = ("subsubsection", num, anchor)
            out.append('<h4 id="%s"><span class="no">%s</span>%s</h4>'
                       % (anchor, "" if starred else num, title_html))
            if self.collecting and not starred:
                self.toc.append((3, num, title_plain, anchor))
            return
        # paragraph / subparagraph: run-in heading
        out.append('<h5 class="runin">%s</h5>' % title_html)

    def sec_number(self) -> str:
        k = self.ctr("section")
        return to_letter(k) if self.appendix else str(k)

    def plain_title(self, title: str) -> str:
        """Title with math shown as raw TeX (keeps the TOC cheap and readable)."""
        t = re.sub(r"\$([^$]*)\$", r'<span class="tex">\1</span>', title)
        t = re.sub(r"\\(?:label)\{[^}]*\}", "", t)
        t = self.macros.expand(t)
        t = re.sub(r"\\(?:textbf|emph|textit|code|texttt)\{([^{}]*)\}", r"\1", t)
        t = re.sub(r"\\[A-Za-z]+\*?", "", t)
        return esc(t).replace("---", "&mdash;").replace("--", "&ndash;").strip()

    def set_label(self, name: str, kind: str = None, number: str = None, anchor: str = None):
        k, num, anc = self.current
        if kind is not None:
            k = kind
        if number is not None:
            num = number
        if anchor:
            anc = anchor
        if not anc:
            anc = self.new_anchor(name)
        self.labels[name] = (k, num, anc)

    # -- environments --------------------------------------------------------
    def handle_env(self, name, body, out, opt=None):
        if name in LIST_ENVS:
            out.append(self.render_list(name, body))
        elif name in MATH_ENVS:
            numbered = not name.endswith("*") and name != "displaymath"
            if name in ("align", "align*", "eqnarray", "eqnarray*", "gather", "gather*",
                        "multline", "multline*", "flalign", "flalign*"):
                rows = split_rows(body)
                blocks = []
                for r in rows:
                    if not r.strip():
                        continue
                    num = str(self.step("equation")) if numbered and not re.search(
                        r"\\(?:nonumber|notag)\b", r) else ""
                    self.current = ("equation", num, "")
                    blocks.append(self.display_math_html(r, numbered=numbered, number=num))
                out.append('<div class="eqngroup">%s</div>' % "".join(blocks))
            else:
                num = str(self.step("equation")) if numbered else ""
                self.current = ("equation", num, "")
                out.append(self.display_math_html(body, numbered=numbered, number=num))
        elif name in ("figure", "figure*"):
            out.append(self.render_float("figure", body, opt))
        elif name in ("table", "table*", "threeparttable", "sidewaystable"):
            out.append(self.render_float("table", body, opt))
        elif name in TABLE_ENVS or name == "array":
            spec, rest = self.tabular_spec(body)
            out.append(self.render_tabular(spec, rest))
        elif name in VERB_ENVS:
            out.append('<pre class="code">%s</pre>' % esc(body.rstrip("\n")))
        elif name in THEOREM_ENVS:
            out.append(self.render_theorem(name, body, opt))
        elif name == "proof":
            head = opt or "Proof"
            inner = self.render_blocks(body).strip()
            out.append('<div class="proof"><span class="proof-head">%s.</span> %s '
                       '<span class="qed">&#9633;</span></div>' % (esc(head), inner))
        elif name == "tablenotes":
            items = self.split_items(body)
            lis = "".join("<li>%s</li>" % self.render_inline(x) for x in items)
            out.append('<div class="tablenotes"><ul>%s</ul></div>' % lis)
        elif name == "abstract":
            out.append('<div class="abstract"><div class="abs-head">Abstract</div>%s</div>'
                       % self.render_blocks(body))
        elif name in IGNORE_ENVS:
            out.append("<!-- ignored %s -->" % name)
        elif name in PASSTHRU_ENVS:
            cls = " blockquote" if name in ("quote", "quotation") else ""
            out.append('<div class="env-%s%s">%s</div>'
                       % (name.replace("*", ""), cls, self.render_blocks(body)))
        else:
            out.append('<div class="env-unknown" data-env="%s">%s</div>'
                       % (esc(name), self.render_blocks(body)))

    def render_list(self, name, body) -> str:
        items = self.split_items(body)
        tag = "ol" if "enum" in name else "ul"
        if name == "description":
            lis = []
            for it in items:
                m = re.match(r"\s*\[([^\]]*)\](.*)", it, re.S)
                if m:
                    lis.append("<dt>%s</dt><dd>%s</dd>"
                               % (self.render_inline(m.group(1)), self.render_blocks(m.group(2)).strip()))
                else:
                    lis.append("<dd>%s</dd>" % self.render_blocks(it).strip())
            return '<dl class="desc">%s</dl>' % "".join(lis)
        lis = "".join("<li>%s</li>" % self.render_blocks(it).strip() for it in items)
        return "<%s>%s</%s>" % (tag, lis, tag)

    def split_items(self, body: str):
        items, buf, i, depth, envdepth = [], [], 0, 0, 0
        n = len(body)
        env_re = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")
        while i < n:
            m = env_re.match(body, i)
            if m:
                envdepth += 1 if m.group(1) == "begin" else -1
                buf.append(m.group(0)); i = m.end(); continue
            if body[i] == "\\" and i + 1 < n:
                if body[i + 1:i + 5] == "item" and depth == 0 and envdepth == 0:
                    nxt = body[i + 5:i + 6]
                    if nxt == "" or not (nxt.isalpha() or nxt == "*"):
                        if buf:
                            items.append("".join(buf))
                        buf = []
                        i += 5
                        continue
                buf.append(body[i:i + 2]); i += 2; continue
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
            buf.append(body[i]); i += 1
        if buf:
            items.append("".join(buf))
        return [x for x in items if x.strip()]

    def render_theorem(self, name, body, opt) -> str:
        k = self.step(name)
        number = "%s.%d" % (self.sec_number(), k)
        anchor = self.new_anchor(name)
        self.current = (name, number, anchor)
        head = name.capitalize()
        note = ""
        if opt:
            note = ' <span class="thm-note">(%s)</span>' % self.render_inline(opt)
        labels = re.findall(r"\\label\s*\{([^}]*)\}", body)
        body = re.sub(r"^\s*\\label\s*\{[^}]*\}\s*", "", body)
        for lb in labels:
            self.set_label(lb, kind=name, number=number, anchor=anchor)
        cls = "remark" if name in ("remark", "intuition", "example", "observation") else "thm"
        return ('<div class="thm thm-%s" id="%s"><div class="thm-head">%s %s%s</div>'
                '<div class="thm-body">%s</div></div>'
                % (name, anchor, head, number, note, self.render_blocks(body).strip()))

    # -- floats --------------------------------------------------------------
    def render_float(self, kind, body, opt) -> str:
        caption, labels, cleaned = self.extract_float_meta(body)
        num = str(self.step(kind))
        anchor = self.new_anchor(kind)
        self.current = (kind, num, anchor)
        for lb in labels:
            self.set_label(lb, kind=kind, number=num, anchor=anchor)
        inner = self.render_blocks(cleaned).strip()
        cap_html = ""
        if caption:
            cap_plain = caption
            cap_html = ('<figcaption><span class="cap-no">%s %s.</span> %s</figcaption>'
                        % (kind.capitalize(), num, self.render_inline(caption)))
            if self.collecting:
                (self.lof if kind == "figure" else self.lot).append((num, cap_plain, anchor))
        if kind == "figure":
            return ('<figure class="fig" id="%s">%s%s</figure>' % (anchor, inner, cap_html))
        return ('<figure class="tab" id="%s">%s%s</figure>' % (anchor, cap_html, inner))

    def extract_float_meta(self, body: str):
        """Pull top-level \\caption / \\label out of a float body."""
        caption, labels = None, []
        spans, i, depth, envdepth = [], 0, 0, 0
        n = len(body)
        env_re = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")
        while i < n:
            m = env_re.match(body, i)
            if m:
                envdepth += 1 if m.group(1) == "begin" else -1
                i = m.end(); continue
            c = body[i]
            if c == "\\" and i + 1 < n:
                mm = re.match(r"\\(caption|label)\*?", body[i:])
                if mm and depth == 0 and envdepth == 0:
                    arg, k = read_group(body, i + mm.end())
                    if mm.group(1) == "caption":
                        if caption is None:
                            caption = arg
                    else:
                        labels.append(arg.strip())
                    spans.append((i, k))
                    i = k
                    continue
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        cleaned = body
        for a, b in reversed(spans):
            cleaned = cleaned[:a] + cleaned[b:]
        return caption, labels, cleaned

    # -- tabular -------------------------------------------------------------
    def tabular_spec(self, body: str):
        i = skip_space(body, 0)
        opt, i2 = read_optional(body, i)
        spec, k = read_group(body, i2 if opt is not None else i)
        return spec, body[k:]

    def render_tabular(self, spec: str, body: str) -> str:
        """booktabs-style tabular -> <table class="data"> with rule-driven borders."""
        cols = self.parse_spec(spec)
        lead_re = re.compile(
            r"\\(?:toprule|midrule|bottomrule|hline|cmidrule(?:\([^)]*\))?\{[^}]*\}"
            r"|cline\{[^}]*\}|addlinespace(?:\{[^}]*\})?|endhead|endfirsthead"
            r"|endfoot|endlastfoot)\s*")
        trail_re = re.compile(
            r"\\(?:toprule|midrule|bottomrule|hline|cmidrule(?:\([^)]*\))?\{[^}]*\}"
            r"|cline\{[^}]*\})\s*$")

        def rule_of(k):
            return {"toprule": "top", "hline": "top", "midrule": "mid",
                    "cmidrule": "mid", "cline": "mid", "bottomrule": "bot"}.get(k)

        rows, pending, bottom = [], None, False
        for seg in split_rows(body):
            seg = seg.strip()
            if not seg:
                continue
            lead = None
            while True:
                m = lead_re.match(seg)
                if not m:
                    break
                lead = rule_of(seg[m.start() + 1:m.end()].split("{")[0].split("(")[0].strip()) or lead
                seg = seg[m.end():].strip()
            trail = None
            while True:
                m = trail_re.search(seg)
                if not m:
                    break
                txt = seg[m.start() + 1:].split("{")[0].split("(")[0].strip()
                trail = rule_of(txt) or trail
                seg = seg[:m.start()].strip()
            if not seg:
                if lead == "bot" or trail == "bot":
                    bottom = True
                elif lead or trail:
                    pending = lead or trail
                continue
            rows.append([self.render_cells(seg, cols), lead or pending])
            pending = trail if trail != "bot" else None
            if trail == "bot":
                bottom = True
        if not rows:
            return ""
        if bottom:
            rows[-1][1] = "bot"
        html = []
        for cells, rule in rows:
            cls = {"top": "r-top", "mid": "r-mid", "bot": "r-bot"}.get(rule, "")
            html.append('<tr%s>%s</tr>' % (' class="%s"' % cls if cls else "", cells))
        return '<table class="data">%s</table>' % "".join(html)

    def parse_spec(self, spec: str):
        """Column spec -> list of alignments ('l'|'c'|'r'), one per column."""
        cols, i, n = [], 0, len(spec or "")
        while i < n:
            c = spec[i]
            if c in "lcr":
                cols.append(c)
                i += 1
            elif c in "pmb":
                _arg, i = read_group(spec, i + 1)
                cols.append("l")
            elif c in "><@!":
                _arg, i = read_group(spec, i + 1)
            elif c == "*":
                rep, k = read_group(spec, i + 1)
                sub, k = read_group(spec, k)
                try:
                    times = int(rep)
                except ValueError:
                    times = 1
                cols.extend(self.parse_spec(sub) * times)
                i = k
            else:
                i += 1
        return cols

    def render_cells(self, row: str, cols) -> str:
        out, col = [], 0
        for p in split_cells(row):
            p = p.strip()
            align = cols[col] if col < len(cols) else "l"
            m = re.match(r"\\multicolumn\s*", p)
            if m:
                n_txt, k = read_group(p, m.end())
                mspec, k = read_group(p, k)
                try:
                    span = int(n_txt)
                except ValueError:
                    span = 1
                align = next((c for c in mspec if c in "lcr"), align)
                out.append('<td colspan="%d" class="c-%s">%s</td>'
                           % (span, align, self.cell_html(p[k:])))
                col += span
                continue
            m = re.match(r"\\multirow\s*", p)
            if m:
                n_txt, k = read_group(p, m.end())
                _w, k = read_group(p, k)
                try:
                    span = int(n_txt)
                except ValueError:
                    span = 1
                out.append('<td rowspan="%d" class="c-%s">%s</td>'
                           % (span, align, self.cell_html(p[k:])))
                col += 1
                continue
            out.append('<td class="c-%s">%s</td>' % (align, self.cell_html(p)))
            col += 1
        return "".join(out)

    def cell_html(self, txt: str) -> str:
        txt = re.sub(r"\\(?:bottomrule|toprule|midrule|hline)\s*$", "", txt.strip())
        txt = txt.replace("\\tabularnewline", "")
        inner = self.render_blocks(txt).strip()
        if inner.startswith("<p>") and inner.endswith("</p>") and "<p>" not in inner[3:-4]:
            inner = inner[3:-4]
        return inner

    # -- images --------------------------------------------------------------
    def image_html(self, path: str, opt) -> str:
        width = ""
        if opt:
            m = re.search(r"width\s*=\s*([0-9.]+)\s*\\(?:linewidth|textwidth|columnwidth)", opt)
            if m:
                width = ' style="width:%g%%"' % (float(m.group(1)) * 100)
            else:
                m = re.search(r"width\s*=\s*([0-9.]+)pt", opt)
                if m:
                    width = ' style="width:%gpx"' % float(m.group(1))
        full = os.path.join(self.dir, path)
        if self.embed_figures and os.path.exists(full):
            with open(full, "rb") as f:
                uri = base64.b64encode(f.read()).decode("ascii")
            ext = os.path.splitext(full)[1].lstrip(".").lower() or "png"
            src = "data:image/%s;base64,%s" % (ext, uri)
        else:
            src = path
        alt = esc(os.path.basename(path))
        if not os.path.exists(full):
            return ('<span class="missing-figure">[figure not found: %s]</span>' % esc(path))
        return '<img class="plot" src="%s" alt="%s"%s loading="lazy">' % (src, alt, width)

    # -- title page ----------------------------------------------------------
    def title_page_html(self) -> str:
        def grab(cmd):
            m = re.search(r"\\%s\s*" % cmd, self.raw_root)
            if not m:
                return ""
            arg, _ = read_group(self.raw_root, m.end())
            arg = arg.replace("\\bfseries", "")
            arg = re.sub(r"\\(?:large|Large|normalsize|small|bfseries|itshape)\b", "", arg)
            arg = re.sub(r"\[\d+pt\]", "", arg)
            return self.render_inline(arg)
        self.title_html = grab("title")
        self.author_html = grab("author")
        import datetime
        date = datetime.date.today().strftime("%d %B %Y")
        return ('<header class="titlepage"><div class="rule-top"></div>'
                '<h1 class="booktitle">%s</h1>'
                '<div class="bookauthor">%s</div>'
                '<div class="bookdate">%s</div>'
                '<div class="rule-bot"></div>'
                '<p class="preview-note">HTML preview of the canonical LaTeX source '
                '<code>report/report.tex</code>. Math is rasterised locally with '
                'matplotlib&rsquo;s mathtext; every figure and table is generated from '
                '<code>results/</code> by the checked-in scripts. Compile the '
                '<code>.tex</code> with <code>latexmk -pdf</code> for the paginated '
                'book.</p></header>'
                % (self.title_html, self.author_html, date))


# ---------------------------------------------------------------------------
# stylesheet
# ---------------------------------------------------------------------------
CSS = r"""
:root{
  --ink:#191919; --muted:#63605a; --rule:#ddd8cf; --accent:#7c2d2d;
  --box:#faf8f4; --box2:#f4f1ea; --code-bg:#f5f4f0; --link:#28527a;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:#fff;color:var(--ink);
     font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,
                 "Times New Roman",serif;
     font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}
#layout{display:flex;align-items:flex-start}
nav#sidebar{position:sticky;top:0;flex:0 0 320px;width:320px;max-height:100vh;overflow-y:auto;
     padding:20px 12px 60px 20px;border-right:1px solid var(--rule);background:#fbfaf7;
     font-size:13.2px;line-height:1.42}
nav#sidebar .brand{font-weight:700;font-size:14px;letter-spacing:.02em;margin-bottom:2px}
nav#sidebar .brandsub{color:var(--muted);font-size:11.5px;margin-bottom:14px}
nav#sidebar ul{list-style:none;margin:0;padding:0}
nav#sidebar li{margin:0}
nav#sidebar a{color:#33312d;text-decoration:none;display:block;padding:2.5px 6px;border-radius:3px}
nav#sidebar a:hover{background:#efece4;color:var(--accent)}
nav#sidebar .l0{margin-top:12px;font-weight:700;text-transform:uppercase;letter-spacing:.09em;
     font-size:11px;color:var(--accent)}
nav#sidebar .l0 a{color:var(--accent)}
nav#sidebar .l2 a{padding-left:20px;color:#55524c;font-size:12.4px}
nav#sidebar .l3 a{padding-left:34px;color:#7a776f;font-size:11.8px}
nav#sidebar .num{color:#a09c93;margin-right:5px;font-variant-numeric:tabular-nums}
main{flex:1 1 auto;min-width:0;max-width:56em;padding:34px 46px 140px;margin:0 auto}
p{margin:.75em 0;text-align:justify;hyphens:auto}
a{color:var(--link)}
h1{font-size:1.85em;line-height:1.2;margin:1.5em 0 .5em}
h2.sec{font-size:1.42em;line-height:1.25;margin:2.3em 0 .6em;padding-bottom:.22em;
     border-bottom:1px solid var(--rule)}
h3{font-size:1.15em;margin:1.7em 0 .4em}
h4{font-size:1.02em;margin:1.4em 0 .3em}
h5.runin{font-size:.98em;margin:1.25em 0 -.5em;font-style:italic}
.sec .no,h3 .no,h4 .no{color:var(--muted);font-weight:400;margin-right:.55em;
     font-variant-numeric:tabular-nums}
.part{margin:3.6em 0 2em;padding:1.5em 1em 1.3em;text-align:center;
     border-top:3px double var(--rule);border-bottom:3px double var(--rule);
     background:linear-gradient(#fff,var(--box))}
.part .kicker{font-size:.74em;letter-spacing:.26em;text-transform:uppercase;color:var(--muted)}
.part .t{font-size:1.95em;font-weight:700;line-height:1.2;margin-top:.28em}
.appendix-start{margin:3em 0;border-top:1px solid var(--rule)}
hr{border:0;border-top:1px solid var(--rule);margin:2.2em auto;width:38%}
hr.pagebreak{width:100%;border-top:1px dashed var(--rule);margin:2.6em 0}
.titlepage{text-align:center;padding:2.4em 0 1.6em;margin-bottom:2em}
.titlepage .rule-top,.titlepage .rule-bot{height:3px;border-top:1px solid #2b2b2b;
     border-bottom:1px solid #2b2b2b;margin:0 auto;width:70%}
.booktitle{font-size:2.05em;line-height:1.22;margin:.9em 0 .5em}
.bookauthor{color:#3a3833;font-size:1em;margin-top:.4em}
.bookdate{color:var(--muted);font-size:.9em;margin-top:.5em}
.preview-note{max-width:38em;margin:2.2em auto 0;color:var(--muted);font-size:.82em;
     text-align:left;border-left:2px solid var(--rule);padding-left:1em}
ul,ol{margin:.7em 0 .9em;padding-left:1.5em}
li{margin:.3em 0}
dl.desc{margin:.8em 0}
dl.desc dt{font-weight:700;margin-top:.8em}
dl.desc dd{margin:.15em 0 .3em 1.2em}
strong{font-weight:700}
em{font-style:italic}
code,.tex,pre.code{font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,
     "DejaVu Sans Mono",monospace}
code{background:var(--code-bg);padding:.08em .32em;border-radius:3px;font-size:.855em;
     color:#3a2f2a;word-break:break-word}
pre.code{background:var(--code-bg);border:1px solid var(--rule);border-left:3px solid #c9c3b6;
     border-radius:4px;padding:.85em 1em;overflow-x:auto;font-size:.795em;line-height:1.5;
     margin:1em 0;white-space:pre}
pre.code code{background:none;padding:0;font-size:1em}
.tex{font-size:.86em;color:#4a4640;background:#f7f6f2;padding:.05em .25em;border-radius:3px}
blockquote,.env-quote,.env-quotation{margin:1.1em 1.4em;padding:.2em 1.1em;
     border-left:3px solid var(--rule);color:#3d3a35;font-size:.97em}
.thm{margin:1.35em 0;padding:.85em 1.15em .9em;background:var(--box);
     border:1px solid #e9e4da;border-left:3px solid var(--accent);border-radius:0 4px 4px 0}
.thm-head{font-weight:700;color:var(--accent);font-size:.94em;letter-spacing:.02em;
     margin-bottom:.3em}
.thm-note{font-weight:400;font-style:italic;color:var(--muted)}
.thm-remark,.thm-intuition,.thm-example,.thm-observation{border-left-color:#9a958a;
     background:#fcfbf9}
.thm-remark .thm-head,.thm-intuition .thm-head,.thm-example .thm-head{color:#5d584f}
.thm-body p:first-child{margin-top:0}
.thm-body p:last-child{margin-bottom:0}
.proof{margin:1.15em 0;padding:.1em 0 .1em 1em;border-left:2px solid #e6e2d9;color:#3a3833}
.proof-head{font-style:italic;font-weight:700}
.qed{float:right}
.abstract{margin:1.4em 0;padding:.9em 1.2em;background:var(--box2);border-radius:4px}
.abs-head{font-weight:700;letter-spacing:.06em;text-transform:uppercase;font-size:.8em;
     color:var(--muted);margin-bottom:.4em}
figure.fig,figure.tab{margin:1.7em 0;padding:0;text-align:center}
figure.fig img.plot,figure.tab img.plot{max-width:100%;height:auto}
figcaption{font-size:.845em;color:var(--muted);text-align:left;margin:.6em auto 0;
     max-width:46em;line-height:1.5}
.cap-no{font-weight:700;color:#4b4740}
.imgline{margin:.9em 0;text-align:center}
.missing-figure{color:#a33;font-size:.85em}
table.data{border-collapse:collapse;margin:.5em auto;font-size:.845em;line-height:1.45;
     text-align:left}
table.data td,table.data th{padding:.3em .62em;vertical-align:top;border:0}
table.data tr.r-top td{border-top:2px solid #2b2b2b}
table.data tr.r-mid td{border-top:1px solid #9b968c}
table.data tr.r-bot td{border-bottom:2px solid #2b2b2b}
table.data td.c-l{text-align:left}table.data td.c-c{text-align:center}
table.data td.c-r{text-align:right}
table.data tr:first-child td{border-top:2px solid #2b2b2b}
table.data tr:last-child td{border-bottom:2px solid #2b2b2b}
.tablenotes{font-size:.8em;color:var(--muted);text-align:left;max-width:46em;margin:.4em auto 0}
.tablenotes ul{padding-left:1.1em;margin:.2em 0}
.eqn{margin:1.05em 0;overflow-x:auto;overflow-y:hidden;padding:.15em 0}
.eqngroup .eqn{margin:.35em 0}
table.eqtab{border-collapse:collapse;margin:0 auto;min-width:60%}
table.eqtab td{padding:.1em .35em;border:0;vertical-align:middle}
td.eq-right{text-align:right}td.eq-left{text-align:left}td.eq-full{text-align:center}
td.eq-no{text-align:right;color:var(--muted);font-size:.9em;white-space:nowrap;
     padding-left:1.4em}
img.math{display:inline-block;vertical-align:baseline}
.eqn img.math{display:inline-block}
span.menv{display:inline-block;vertical-align:middle}
table.mtab{border-collapse:collapse;display:inline-table;vertical-align:middle}
table.mtab td.mcell{padding:.08em .4em;text-align:center;vertical-align:middle}
table.mtab td.cond{text-align:left;padding-left:.7em}
.menv.bmat,.menv.pmat,.menv.vmat{position:relative;padding:0 .42em}
.menv.bmat{border-left:1.4px solid #2b2b2b;border-right:1.4px solid #2b2b2b;border-radius:2px}
.menv.pmat{border-left:1.4px solid #2b2b2b;border-right:1.4px solid #2b2b2b;
     border-radius:50%/8px}
.menv.vmat{border-left:1.4px solid #2b2b2b;border-right:1.4px solid #2b2b2b}
.menv.cases{display:inline-flex;align-items:stretch;vertical-align:middle}
.cases-brace{font-size:1.9em;font-weight:100;line-height:.9;transform:scaleY(1.6);
     transform-origin:center;color:#2b2b2b;margin-right:.15em}
.cases-brace.right{display:none}
span.ubrace,span.obrace{display:inline-flex;flex-direction:column;align-items:center;
     vertical-align:baseline;margin:0 .15em}
span.ubrace{flex-direction:column}
.ub-mark{font-size:1.15em;line-height:.55;transform:scaleY(.8);color:#2b2b2b}
.ub-sub{font-size:.85em;line-height:1}
span.boxed{display:inline-block;border:1.3px solid #2b2b2b;padding:.12em .4em;margin:0 .1em}
.ok{color:#2f6b3a;font-weight:700}
.ref{text-decoration:none;border-bottom:1px dotted #9ab}
.ref-unresolved{color:#b00;font-size:.85em}
.fn{font-size:.75em;color:var(--muted)}
.toc-block{margin:1.6em 0;padding:1.1em 1.3em;background:var(--box);border-radius:4px;
     border:1px solid #ece7dd}
.toc-block h2{margin-top:0}
.toc-block ol{list-style:none;padding-left:0;margin:.3em 0}
.toc-block li{margin:.16em 0}
.toc-block .t0{font-weight:700;text-transform:uppercase;letter-spacing:.08em;font-size:.85em;
     color:var(--accent);margin-top:.9em}
.toc-block .t1{margin-left:.4em}
.toc-block .t2{margin-left:1.9em;font-size:.94em;color:#4a4740}
.toc-block .t3{margin-left:3.4em;font-size:.88em;color:#6b675f}
.toc-block .n{color:#a09c93;margin-right:.5em;font-variant-numeric:tabular-nums}
.refs{font-size:.9em}
.refs ol{padding-left:1.6em}
.refs li{margin:.45em 0;text-align:left}
.lof table{border-collapse:collapse;margin:.6em 0;font-size:.9em;width:100%}
.lof td{padding:.22em .5em;border-bottom:1px dotted var(--rule);vertical-align:top}
.lof td.n{white-space:nowrap;color:var(--muted);width:5.5em}
.build-note{margin-top:4em;padding-top:1em;border-top:1px solid var(--rule);
     color:var(--muted);font-size:.8em}
@media (max-width:1100px){
  nav#sidebar{display:none}
  main{padding:24px 20px 90px}
}
@media print{
  nav#sidebar{display:none}
  main{max-width:none;padding:0}
  .eqn{overflow:visible}
  p{text-align:left}
}
"""


def build_sidebar(toc, book, max_level: int = 1) -> str:
    items = []
    for level, num, title, anchor in [t for t in toc if t[0] <= max_level]:
        n = '<span class="num">%s</span>' % esc(num) if num else ""
        items.append('<li class="l%d"><a href="#%s">%s%s</a></li>'
                     % (level, esc(anchor), n, book.render_inline(title)))
    return ('<nav id="sidebar"><div class="brand">Multi-Asset Statistical Arbitrage</div>'
            '<div class="brandsub">a book-length build report</div><ul>%s</ul></nav>'
            % "".join(items))


def build_toc_block(toc, book, max_level: int = 2) -> str:
    lis = []
    for level, num, title, anchor in [t for t in toc if t[0] <= max_level]:
        n = '<span class="n">%s</span>' % esc(num) if num else ""
        lis.append('<li class="t%d"><a href="#%s">%s%s</a></li>'
                   % (level, esc(anchor), n, book.render_inline(title)))
    return ('<div class="toc-block"><h2>Contents</h2><ol>%s</ol></div>' % "".join(lis))


def build_float_list(entries, kind, book) -> str:
    rows = "".join('<tr><td class="n"><a href="#%s">%s %s</a></td><td>%s</td></tr>'
                   % (esc(a), kind, esc(n), book.render_inline(t)) for n, t, a in entries)
    return ('<div class="lof"><table>%s</table></div>' % rows) if rows else \
        '<div class="lof"><em>none</em></div>'


def build_refs(book: Book) -> str:
    if not book.cited:
        return ""
    lis = []
    for i, key in enumerate(book.cited, 1):
        lis.append('<li id="bib-%s">%s</li>' % (esc(key), book.fmt_bib(key)))
    return ('<section class="refs"><h2 class="sec"><span class="no"></span>References</h2>'
            '<ol>%s</ol></section>' % "".join(lis))


def render_book(report_dir: str, out_path: str = None, dpi: int = MATH_DPI,
                embed_figures: bool = False, embed_math: bool = False,
                verbose: bool = True) -> str:
    """Two-pass render of report/report.tex to an HTML book."""
    book = Book(report_dir, dpi=dpi, embed_figures=embed_figures, embed_math=embed_math)
    body = book.load()

    # ---- pass 1: numbering, labels, TOC, citation order (no rasterisation) ----
    book.dry = True
    book.collecting = True
    book.reset_state()
    book.render_blocks(body)
    toc, lof, lot, cited = list(book.toc), list(book.lof), list(book.lot), list(book.cited)
    labels_pass1 = dict(book.labels)

    # ---- pass 2: real render, references now resolvable ---------------------
    book.dry = False
    book.reset_state()
    book.collecting = False               # pass 2 must not duplicate the TOC/lists
    book.toc, book.lof, book.lot, book.cited = toc, lof, lot, cited
    book.labels = labels_pass1          # forward references resolve from pass 1
    main = book.render_blocks(body)

    main = (main
            .replace("<!--TOC-->", build_toc_block(book.toc, book))
            .replace("<!--LOF-->", build_float_list(book.lof, "Figure", book))
            .replace("<!--LOT-->", build_float_list(book.lot, "Table", book))
            .replace("<!--BIB-->", build_refs(book)))
    import datetime
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_math = len(book.math.cache)
    note = ('<div class="build-note">Rendered from <code>report/report.tex</code> and '
            '<code>report/chapters/*.tex</code> on %s. %d math expressions rasterised '
            'with matplotlib mathtext at %d&nbsp;dpi; %d cross-references resolved; '
            '%d citations. Figures are loaded from <code>report/figures/</code>, which '
            'the build copies from <code>results/figures/</code>.'
            '</div>' % (stamp, n_math, dpi, len(book.labels), len(book.cited)))

    html_doc = ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                "<title>Multi-Asset Statistical Arbitrage &mdash; a book-length build report</title>\n"
                "<style>%s</style>\n</head>\n<body>\n<div id=\"layout\">\n%s\n<main>\n%s\n%s\n"
                "</main>\n</div>\n</body>\n</html>\n"
                % (CSS, build_sidebar(book.toc, book), main, note))

    out_path = out_path or os.path.join(report_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    if verbose:
        size_mb = os.path.getsize(out_path) / 1e6
        print("      -> %s (%.1f MB)" % (os.path.relpath(out_path,
                                                         os.path.dirname(report_dir)), size_mb))
        print("      %d unique math expressions (%d new PNG files), %d labels, "
              "%d TOC entries, %d figures, %d tables, %d citations"
              % (n_math, book.math.files, len(labels_pass1), len(book.toc),
                 len(book.lof), len(book.lot), len(book.cited)))
        if book.math.failures:
            uniq = {}
            for tex, err in book.math.failures:
                uniq.setdefault(err, []).append(tex)
            print("      ! %d math expressions could not be rasterised (%d distinct errors):"
                  % (len(book.math.failures), len(uniq)))
            for err, texs in list(uniq.items())[:12]:
                print("        - %s  e.g. %r" % (err, texs[0][:70]))
        if book._unknown:
            print("      ! unhandled LaTeX commands (dropped), with context:")
            for name, ctx in sorted(book._unknown.items()):
                print("        \\%-14s ...%s..." % (name, ctx))
    return out_path


def check_math(report_dir: str) -> int:
    """Run every math expression in the book through the real translation pipeline
    (macro expansion -> translation -> structural decomposition -> mathtext parse)
    and report what cannot be rasterised. This is what keeps the LaTeX mathtext-safe."""
    import glob
    book = Book(report_dir)
    macros = book.macros
    files = ([os.path.join(report_dir, "report.tex")]
             + sorted(glob.glob(os.path.join(report_dir, "chapters", "*.tex")))
             + sorted(glob.glob(os.path.join(report_dir, "tables", "*.tex"))))
    texts = []
    for f in files:
        raw = open(f, encoding="utf-8").read()
        macros.load(raw)
        raw = re.sub(r"\\begin\{(?:verbatim|Verbatim|lstlisting)\*?\}.*?"
                     r"\\end\{(?:verbatim|Verbatim|lstlisting)\*?\}", " ", raw, flags=re.S)
        texts.append((os.path.basename(f), strip_comments(raw)))
    mr = book.math
    mr._init_mpl()
    prop = mr._FontProperties(family="serif", size=MATH_PT)

    def parse_ok(tex):
        try:
            mr._parser.parse("$" + mr.translate(tex) + "$", 72, prop)
            return None
        except Exception as e:
            return str(e).strip().splitlines()[-1][:110]

    bad, total = [], 0
    inline_re = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)", re.S)
    disp_re = re.compile(
        r"\\begin\{(equation|align|gather|multline|flalign)\*?\}(.*?)\\end\{\1\*?\}"
        r"|\\\[(.+?)\\\]", re.S)
    drop_re = re.compile(r"\\(?:label\{[^}]*\}|nonumber|notag)")

    for fname, t in texts:
        for m in disp_re.finditer(t):
            body = m.group(2) if m.group(2) is not None else m.group(3)
            for row in split_rows(body or ""):
                for cell in split_cells(drop_re.sub("", row)):
                    for piece in book.math_pieces(macros.expand(cell)):
                        total += 1
                        err = parse_ok(piece)
                        if err:
                            bad.append((fname, piece, err))
        for m in inline_re.finditer(disp_re.sub(" ", t)):
            tex = m.group(1)
            if "\\begin{" in tex:
                continue
            for piece in book.math_pieces(macros.expand(tex)):
                total += 1
                err = parse_ok(piece)
                if err:
                    bad.append((fname, piece, err))

    print("math expressions checked: %d   failures: %d" % (total, len(bad)))
    seen = set()
    for fname, piece, err in bad:
        if piece in seen:
            continue
        seen.add(piece)
        print("  [%s] %-62r %s" % (fname, piece[:62], err))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=MATH_DPI)
    ap.add_argument("--embed-figures", action="store_true",
                    help="inline figure PNGs as data-URIs (bigger file, fully portable)")
    ap.add_argument("--embed-math", action="store_true",
                    help="inline math PNGs as data-URIs instead of writing report/math/")
    ap.add_argument("--portable", action="store_true",
                    help="single self-contained file: embed figures AND math")
    ap.add_argument("--check-math", action="store_true",
                    help="only report which math expressions mathtext cannot render")
    args = ap.parse_args(argv)
    if args.check_math:
        return check_math(args.report_dir)
    render_book(args.report_dir, args.out, dpi=args.dpi,
                embed_figures=args.embed_figures or args.portable,
                embed_math=args.embed_math or args.portable)
    return 0


if __name__ == "__main__":
    sys.exit(main())
