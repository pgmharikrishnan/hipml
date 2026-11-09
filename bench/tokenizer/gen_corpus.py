#!/usr/bin/env python3
"""Generate the WP48 tokenizer differential corpus.

Deterministic: same Python 3 + same seed -> same file. Output lines are
``hex(utf8-text) TAB label`` so newlines, NULs, bidi controls and lone
combining marks survive the line format.

Coverage:
  * every assigned codepoint of every CJK / Hangul / kana / Tangut block,
    chunked to 512 codepoints, plus block boundaries;
  * every assigned emoji codepoint, all 676 regional-indicator flags, ZWJ
    families/professions/skin tones, keycaps, tag flags, VS15/VS16;
  * every Unicode combining mark (Mn/Mc/Me) as ``a<mark>b``, standalone and
    in clusters;
  * RTL scripts (Hebrew, Arabic, Syriac, Thaana, NKo, Samaritan, Mandaic,
    Arabic presentation forms, Hebrew presentation forms), bidi controls;
  * box drawing, block elements, geometric shapes, arrows, math symbols;
  * mixed ASCII/Unicode identifiers;
  * whitespace edge cases (leading/trailing/runs, tabs, CR/LF/CRLF, NBSP,
    thin space, ideographic space, ZWSP, BOM, NEL, LS/PS);
  * 2,000 random valid-UTF-8 strings, including unassigned codepoints.

Usage: python3 bench/tokenizer/gen_corpus.py [OUT]   (default corpus.txt)
"""

import random
import sys
import unicodedata

OUT = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
rng = random.Random(20261109)

records = []  # (label, text)


def add(label, text):
    records.append((label, text))


def assigned(cp):
    if 0xD800 <= cp <= 0xDFFF:
        return False
    try:
        return unicodedata.category(chr(cp)) != "Cn"
    except ValueError:
        return False


BLOCKS = [
    # CJK radicals and ideographs
    ("cjk-radicals", 0x2E80, 0x2EFF),
    ("kangxi", 0x2F00, 0x2FDF),
    ("cjk-strokes", 0x31C0, 0x31EF),
    ("cjk-uro", 0x4E00, 0x9FFF),
    ("cjk-ext-a", 0x3400, 0x4DBF),
    ("cjk-ext-b", 0x20000, 0x2A6DF),
    ("cjk-ext-c", 0x2A700, 0x2B73F),
    ("cjk-ext-d", 0x2B740, 0x2B81F),
    ("cjk-ext-e", 0x2B820, 0x2CEAF),
    ("cjk-ext-f", 0x2CEB0, 0x2EBEF),
    ("cjk-ext-i", 0x2EBF0, 0x2EE5F),
    ("cjk-ext-g", 0x30000, 0x3134F),
    ("cjk-ext-h", 0x31350, 0x323AF),
    ("cjk-ext-j", 0x323B0, 0x3347F),
    ("cjk-compat", 0xF900, 0xFAFF),
    ("cjk-compat-supp", 0x2F800, 0x2FA1F),
    ("tangut", 0x17000, 0x187FF),
    ("tangut-comp", 0x18800, 0x18AFF),
    ("tangut-supp", 0x18D00, 0x18D8F),
    ("ideographic-symbols", 0x16FE0, 0x16FFF),
    ("enclosed-ideographic", 0x1F200, 0x1F2FF),
    # kana and hangul
    ("hiragana", 0x3040, 0x309F),
    ("katakana", 0x30A0, 0x30FF),
    ("kana-ext", 0x31F0, 0x31FF),
    ("kana-supp", 0x1B000, 0x1B16F),
    ("kana-ext-a", 0x1AFF0, 0x1AFFF),
    ("halfwidth-katakana", 0xFF66, 0xFF9F),
    ("bopomofo", 0x3100, 0x312F),
    ("bopomofo-ext", 0x31A0, 0x31BF),
    ("hangul-jamo", 0x1100, 0x11FF),
    ("hangul-compat-jamo", 0x3130, 0x318F),
    ("hangul-jamo-ext-a", 0xA960, 0xA97F),
    ("hangul-jamo-ext-b", 0xD7B0, 0xD7FF),
    ("hangul-syllables", 0xAC00, 0xD7A3),
    # RTL scripts
    ("hebrew", 0x0590, 0x05FF),
    ("hebrew-pres", 0xFB1D, 0xFB4F),
    ("arabic", 0x0600, 0x06FF),
    ("arabic-supp", 0x0750, 0x077F),
    ("arabic-ext-a", 0x08A0, 0x08FF),
    ("arabic-ext-b", 0x0870, 0x089F),
    ("arabic-ext-c", 0x10E60, 0x10E7F),
    ("arabic-pres-a", 0xFB50, 0xFDFF),
    ("arabic-pres-b", 0xFE70, 0xFEFF),
    ("syriac", 0x0700, 0x074F),
    ("syriac-supp", 0x0860, 0x086F),
    ("thaana", 0x0780, 0x07BF),
    ("nko", 0x07C0, 0x07FF),
    ("samaritan", 0x0800, 0x083F),
    ("mandaic", 0x0840, 0x085F),
    ("adlam", 0x1E900, 0x1E95F),
    # symbols an agent emits
    ("box-drawing", 0x2500, 0x257F),
    ("block-elements", 0x2580, 0x259F),
    ("geometric", 0x25A0, 0x25FF),
    ("arrows", 0x2190, 0x21FF),
    ("math", 0x2200, 0x22FF),
    ("tech", 0x2300, 0x23FF),
    ("legacy-computing", 0x1FB00, 0x1FBFF),
    ("dingbats", 0x2700, 0x27BF),
    ("misc-symbols", 0x2600, 0x26FF),
]


def chunks(cps, size):
    for i in range(0, len(cps), size):
        yield cps[i:i + size]


for name, lo, hi in BLOCKS:
    cps = [cp for cp in range(lo, hi + 1) if assigned(cp)]
    if not cps:
        continue
    # boundaries, then 512-codepoint chunks over the whole block
    add(name + "-first", "".join(chr(cp) for cp in cps[:8]))
    add(name + "-last", "".join(chr(cp) for cp in cps[-8:]))
    for k, part in enumerate(chunks(cps, 512)):
        add(f"{name}-{k:03d}", "".join(chr(cp) for cp in part))

# every emoji-ish codepoint, in chunks
emoji_cps = []
for lo, hi in [(0x1F000, 0x1FAFF), (0x2600, 0x27BF), (0x2B00, 0x2BFF),
               (0x1F1E6, 0x1F1FF), (0xFE00, 0xFE0F), (0x200D, 0x200D),
               (0x1F3FB, 0x1F3FF), (0xE0020, 0xE007F)]:
    emoji_cps.extend(cp for cp in range(lo, hi + 1) if assigned(cp))
for k, part in enumerate(chunks(emoji_cps, 256)):
    add(f"emoji-chunk-{k:03d}", "".join(chr(cp) for cp in part))

ZWJ = "\u200d"
VS16 = "\ufe0f"
SKIN = ["\U0001F3FB", "\U0001F3FC", "\U0001F3FD", "\U0001F3FE", "\U0001F3FF"]
curated_emoji = [
    "\U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466",  # family
    "\U0001F468\u200D\U0001F469\u200D\U0001F467",
    "\U0001F469\u200D\U0001F4BB",  # woman technologist
    "\U0001F468\u200D\U0001F4BB",
    "\U0001F469" + SKIN[2] + "\u200D\U0001F4BB",
    "\U0001F468" + SKIN[4] + "\u200D\U0001F680",
    "\U0001F9D1" + SKIN[1] + "\u200D\U0001F9B0",
    "\U0001F3F3\ufe0f\u200D\U0001F308",  # rainbow flag
    "\U0001F3F4\u200D\u2620\ufe0f",      # pirate flag
    "\U0001F441\ufe0f\u200D\U0001F5E8\ufe0f",
    "\U0001F9D1\u200D\U0001F91D\u200D\U0001F9D1",
    "\U0001F469\u200D\u2764\ufe0f\u200D\U0001F468",
    "\U0001F491\u200D\U0001F468",
    "\U0001F5E3\ufe0f", "\u2764\ufe0f", "\u2764", "\u263A\ufe0f", "\u263A",
    "1\ufe0f\u20e3", "2\ufe0f\u20e3", "#\ufe0f\u20e3", "*\ufe0f\u20e3",
    "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F",
    "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F",
    "\U0001F3F4\U000E0067\U000E0062\U000E0077\U000E006C\U000E0073\U000E007F",
    "\u00a9\ufe0f", "\u00ae\ufe0f", "\u2122\ufe0f",
    "a" + ZWJ + "b", "a" + ZWJ + ZWJ + "b", ZWJ + "a", "a" + ZWJ,
    ZWJ, ZWJ * 4, "\u200c", "\u200c\u200d",
    "\U0001F600\u200D\U0001F600", "\U0001F600" + ZWJ,
    "emoji:\U0001F600:end", "x\U0001F44D" + SKIN[0] + "y",
]
for i, s in enumerate(curated_emoji):
    add(f"emoji-curated-{i:03d}", s)

# regional-indicator flags: all 26x26 pairs, plus a few valid ones
letters = [chr(0x1F1E6 + i) for i in range(26)]
for a in letters:
    for b in letters:
        add("flag-pair", a + b)

# every combining mark in the whole repertoire
marks = [cp for cp in range(0x110000)
         if assigned(cp) and unicodedata.category(chr(cp)) in ("Mn", "Mc", "Me")]
for cp in marks:
    m = chr(cp)
    add(f"mark-{cp:05X}", "a" + m + "b")
for k, part in enumerate(chunks(marks, 128)):
    add(f"mark-chunk-{k:03d}", "".join(chr(cp) for cp in part))
for i in range(200):
    n = rng.randrange(2, 7)
    base = chr(rng.choice([0x61, 0x65, 0x3B1, 0x430, 0x5D0, 0x627]))
    add("mark-cluster", base + "".join(chr(rng.choice(marks)) for _ in range(n)))

# bidi and RTL edge cases
bidi_controls = [0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
                 0x2066, 0x2067, 0x2068, 0x2069, 0x061C, 0x200B, 0xFEFF]
for cp in bidi_controls:
    c = chr(cp)
    add(f"bidi-{cp:04X}", "abc" + c + "\u05d0\u05d1\u05d2" + c + "123")
for label, text in [
    ("rtl-arabic-diacritics", "\u0645\u064e\u0631\u0652\u062d\u064e\u0628\u064b\u0627"),
    ("rtl-hebrew-niqqud", "\u05e9\u05b8\u05c1\u05dc\u05d5\u05b9\u05dd"),
    ("rtl-mixed", "\u05e9\u05dc\u05d5\u05dd hello 123 \u4e16\u754c"),
    ("rtl-mixed2", "def f(x):  # \u0645\u062a\u063a\u064a\u0631\n    return x"),
    ("rtl-numbers", "\u0661\u0662\u0663\u0664 \u06f1\u06f2\u06f3"),
    ("rtl-parens", "(\u05d0\u05d1) [\u0627\u0628] {\u05d2\u05d3}"),
    ("rtl-emoji", "\u05e9\u05dc\u05d5\u05dd \U0001F600 \u0645\u0631\u062d\u0628\u0627"),
]:
    add(label, text)

# identifiers mixing ASCII and non-ASCII letters
scripts = ["\u00e9", "\u00fc", "\u00f1", "\u03b1", "\u03b2", "\u0430", "\u0431",
           "\u05d0", "\u0627", "\u4e16", "\u754c", "\u3042", "\u30a2",
           "\uac00", "\uac01", "\u0e01", "\u0e02", "\u0915", "\u0937"]
ides = [
    "caf\u00e9", "na\u00efve_identifier", "Stra\u00dfe", "ma\u00f1ana",
    "variables_\u00e9t\u00e9", "\u03b1\u03b2\u03b3_function", "x_\u4e16\u754c",
    "fn_\u3042\u3044", "let_ \u00e9", "\u00e9l\u00e9ment", "e\u0301le\u0301ment",
    "snake_case_\u00e9", "UPPER_\u00c9", "a1b2_\u00e9\u00e8",
    "$_weird_\u00e9", "_leading\u00e9", "trailing\u00e9_",
    "\U0001F600_id", "id\U0001F600", "\u4e16\u754c_42", "42_\u4e16\u754c",
    "\\u00e9", "x\\n_y", "emoji\u200djoin", "zw\u200bj",
]
for i, s in enumerate(ides):
    add(f"ident-curated-{i:03d}", s)
for i in range(300):
    n = rng.randrange(1, 5)
    tok = "".join(rng.choice(scripts) for _ in range(n))
    pattern = rng.randrange(5)
    if pattern == 0:
        s = tok
    elif pattern == 1:
        s = "abc_" + tok + "_123"
    elif pattern == 2:
        s = tok + "_" + tok
    elif pattern == 3:
        s = "def " + tok + "(x): return x + 1"
    else:
        s = tok + str(rng.randrange(1000))
    add("ident-random", s)

# whitespace edge cases
WS = [" ", "\t", "\n", "\r", "\r\n", "\u00a0", "\u1680", "\u2000", "\u2001",
      "\u2002", "\u2003", "\u2004", "\u2005", "\u2006", "\u2007", "\u2008",
      "\u2009", "\u200a", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000",
      "\u000b", "\u000c", "\u0085", "\u200b", "\ufeff"]
for i, w in enumerate(WS):
    add(f"ws-{i:02d}-alone", w)
    add(f"ws-{i:02d}-lead", w + "word")
    add(f"ws-{i:02d}-trail", "word" + w)
    add(f"ws-{i:02d}-mid", "a" + w + "b")
    add(f"ws-{i:02d}-run", "a" + w * 3 + "b")
    add(f"ws-{i:02d}-edges", w * 2 + "x y" + w * 2)
for i in range(120):
    a = rng.choice(WS)
    b = rng.choice(WS)
    c = rng.choice(WS)
    add("ws-random", a + "alpha" + b + "beta" + c + "gamma")
add("ws-empty", "")
add("ws-spaces", " " * 17)
add("ws-newlines", "\n" * 9)
add("ws-tabs", "\t\t\t")
add("ws-mixed-run", " \t\n\r\u00a0\u3000 \r\n\t ")
add("ws-code-indent", "\n\tif x:\n\t\treturn 1\n")
add("ws-crlf-code", "fn main() {\r\n    println!(\"hi\");\r\n}\r\n")
add("ws-markdown", "# Title\n\n- a\n  - b\n\n```rust\nlet x = 1;\n```\n")

# random valid UTF-8 strings, including unassigned codepoints
def rand_cp():
    while True:
        band = rng.randrange(4)
        if band == 0:
            cp = rng.randrange(0x00, 0x7F)
        elif band == 1:
            cp = rng.randrange(0x80, 0x3000)
        elif band == 2:
            cp = rng.randrange(0x3000, 0x10000)
        else:
            cp = rng.randrange(0x10000, 0x110000)
        if not (0xD800 <= cp <= 0xDFFF):
            return cp


for i in range(2000):
    n = rng.randrange(1, 25)
    s = "".join(chr(rand_cp()) for _ in range(n))
    add("random-utf8", s)

# every assigned codepoint in Unicode, chunked: an exhaustive category sweep
# over scripts the targeted blocks above do not name.
all_cps = [cp for cp in range(0x110000) if assigned(cp)]
for k, part in enumerate(chunks(all_cps, 512)):
    add(f"all-assigned-{k:03d}", "".join(chr(cp) for cp in part))

# a realistic agent transcript fragment: code plus non-ASCII
add("agent-code-1", "fn caf\u00e9(x: &\u4e16\u754c) -> Result<(), E> {\n    // \u5bf9\u8c61\u540d\n    Ok(())\n}")
add("agent-code-2", "const GREETING: &str = \"\u3053\u3093\u306b\u3061\u306f \U0001F44B\";")
add("agent-code-3", "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\n\u2502 table \u2502\n\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")
add("agent-json", "{\"name\":\"caf\u00e9\",\"path\":\"\u6587\u4ef6/\u30c6\u30b9\u30c8.rs\"}")

seen = set()
uniq = []
for label, text in records:
    if text in seen:
        continue
    seen.add(text)
    uniq.append((label, text))

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for label, text in uniq:
        f.write(text.encode("utf-8").hex() + "\t" + label + "\n")

from collections import Counter
print(f"{len(uniq)} strings -> {OUT}")
for label, n in Counter(l.split("-")[0] for l, _ in uniq).most_common():
    print(f"  {label}: {n}")
