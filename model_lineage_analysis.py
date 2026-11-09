"""Model lineage analysis: base vs. derivative models and who publishes them.

Questions (from the Oct 22 2026 meeting):
  1. Of the most-downloaded models (top 0.01% = 313 of ~3.1M), which are base
     models and which are derivatives (fine-tunes, quantizations, adapters,
     merges)? Who publishes them?
  2. Across the full metadata pull, whose base models do derivatives trace
     back to? (Tests the idea that a few large labs have more reach than their
     own upload counts suggest.)

Lineage comes from Hugging Face's self-reported `base_model` field
(cardData.base_model, plus the `base_model:<relation>:<id>` tags). A model
with no declared parent is NOT necessarily a base model: many older models
(BERT-era fine-tunes) never filled in the field. Results are a lower bound
on derivative counts.

Publisher types come from the hand-labeled lookup
`model_lineage/author_types.csv`, which labels the account that published a
model. Accounts not in the table are reported as "Unlabeled (long tail)".

Two evidence-backed correction files are applied on top (every row cites a
public source; a metadata-only baseline is printed for comparison):
  - `undeclared_parents.csv`: parent links the model card documents but the
    base_model field omits (e.g. nreimers/MiniLM-L6 is a pruned copy of
    microsoft/MiniLM-L12).
  - `publisher_overrides.csv`: accounts or individual models that publish a
    company's weights (e.g. intfloat = Microsoft's E5 releases).

Run:  python3 model_lineage_analysis.py
Needs the LFS data files (`git lfs pull`). `raw_data/models_popular.jsonl` is
optional (gitignored) but adds lineage fields for most of the top-313.
"""

# %% Imports and paths
import collections
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path

DOWNLOADS_CSV = "raw_data/hf_model_downloads_10_08_26.csv"
METADATA_JSONL = [
    "clean_license_data/clean_licenses.jsonl",
    "clean_license_data/unresolved_licenses.jsonl",
]
OPTIONAL_METADATA_JSONL = "raw_data/models_popular.jsonl"
AUTHOR_TYPES_CSV = "model_lineage/author_types.csv"
# Evidence-backed corrections (each row cites a public source):
UNDECLARED_PARENTS_CSV = "model_lineage/undeclared_parents.csv"   # parent links the metadata omits
PUBLISHER_OVERRIDES_CSV = "model_lineage/publisher_overrides.csv"  # accounts/models published on a company's behalf
ORG_CLASSIFICATION_CSV = "model_lineage/org_classification.csv"    # market cap + compute flag + ECI name per company
ECI_SNAPSHOT_CSV = "model_lineage/eci_scores_snapshot.csv"         # Epoch Capabilities Index, CC BY 4.0, accessed 2026-11-08

# Frontier developer = passes EITHER test (EU AI Act Annex XIII lists both kinds of criteria):
#   capability test: released a model within CAPABILITY_GAP_ECI points of the best model available at
#                    its release date, on Epoch's Capabilities Index (ECI). At the 2025-26 pace of
#                    ~16-18 ECI points/year, 5 points ~= 3-4 months behind the leader.
#   compute test:    trained a model with > 1e25 FLOP (EU AI Act Art. 51(2)), per Epoch's compute field.
CAPABILITY_GAP_ECI = 5.0
CAPABILITY_START = "2024-01-01"  # ECI coverage is thin before 2024, which would make early gaps trivially small
OUT_DIR = "model_lineage/output"
TOP_N = 313  # top 0.01% of models by all-time downloads

RELATIONS = ["finetune", "quantized", "adapter", "merge"]
UNLABELED = "Unlabeled (long tail)"
# Companies are split by two public, checkable tests (see org_classification.csv):
#   Big Tech  = publicly listed tech company with market cap >= $100B
#   Frontier  = has trained a model with > 1e25 training FLOP (EU AI Act Art. 51
#               presumption of high-impact capabilities), per Epoch AI's compute field
BIG_FRONTIER = "Big Tech + frontier developer"
BIG_ONLY = "Big Tech, not frontier"
FRONTIER_ONLY = "Frontier developer, not Big Tech"
OTHER_CO = "Other companies & startups"
COMPANY = "Company"  # label used in author_types / publisher_overrides; resolved via org_classification
BIG_TECH_MCAP_USD_BN = 100
PUBLISHER_TYPES = [
    BIG_FRONTIER,
    BIG_ONLY,
    FRONTIER_ONLY,
    OTHER_CO,
    "Repackagers (quantize/convert/re-host)",
    "Academic & nonprofit",
    "Individuals & community",
    "Open-source projects & Hugging Face",
    "Needs review",
    UNLABELED,
]
# Figures merge the three Big Tech / frontier groups into one category to keep
# the diagrams readable; tables keep the full four-way company split.
BIG_OR_FRONTIER = "Big Tech or frontier developer"
FIGURE_MERGE = {BIG_FRONTIER: BIG_OR_FRONTIER, BIG_ONLY: BIG_OR_FRONTIER, FRONTIER_ONLY: BIG_OR_FRONTIER}
FIGURE_ORDER = [
    BIG_OR_FRONTIER,
    OTHER_CO,
    "Repackagers (quantize/convert/re-host)",
    "Academic & nonprofit",
    "Individuals & community",
    "Open-source projects & Hugging Face",
    "Needs review",
    UNLABELED,
]
# Fixed categorical order (validated palette, slots 1-6 in order); grays for the
# two "not labeled" buckets so they recede.
TYPE_COLORS = {
    BIG_OR_FRONTIER: "#2a78d6",
    OTHER_CO: "#eb6834",
    "Repackagers (quantize/convert/re-host)": "#1baf7a",
    "Academic & nonprofit": "#eda100",
    "Individuals & community": "#e87ba4",
    "Open-source projects & Hugging Face": "#008300",
    "Needs review": "#a3a29c",
    UNLABELED: "#cfcec9",
}


def for_figure(types: pd.Series) -> pd.Series:
    """Collapse the Big Tech / frontier groups into one figure category."""
    return types.replace(FIGURE_MERGE)
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

# Old un-namespaced Hub ids (e.g. "bert-base-uncased") now redirect to an org.
LEGACY_PREFIXES = [
    ("distilroberta", "distilbert"), ("distilgpt2", "distilbert"),
    ("distilbert", "distilbert"), ("xlm-roberta", "FacebookAI"),
    ("roberta", "FacebookAI"), ("bert-", "google-bert"), ("gpt2", "openai-community"),
    ("t5-", "google-t5"), ("albert-", "albert"), ("qwen", "Qwen"),
    ("gpt-oss", "openai"), ("glm-", "zai-org"), ("exaone", "LGAI-EXAONE"),
    ("minimax", "MiniMaxAI"), ("wav2vec2", "facebook"), ("krea", "krea"),
]


def publisher_of(model_id: str) -> str:
    """Account that published a model id, mapping legacy un-namespaced ids."""
    if "/" in model_id:
        return model_id.split("/")[0]
    low = model_id.lower()
    for prefix, org in LEGACY_PREFIXES:
        if low.startswith(prefix):
            return org
    return model_id


def as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [v for v in value if isinstance(v, str)]


# %% Load metadata into one table (one row per model)
def load_models() -> pd.DataFrame:
    recs = {}
    tag_re = re.compile(r"base_model:(finetune|quantized|adapter|merge):")
    for path in METADATA_JSONL:
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                card = d.get("card_data") or {}
                rels = collections.Counter(
                    m.group(1) for t in (d.get("tags") or []) if (m := tag_re.match(t))
                )
                recs[d["id"]] = {
                    "id": d["id"],
                    "bases": as_list(card.get("base_model")),
                    "relation": rels.most_common(1)[0][0] if rels else None,
                    "pipeline_tag": card.get("pipeline_tag"),
                }
    if os.path.exists(OPTIONAL_METADATA_JSONL):
        with open(OPTIONAL_METADATA_JSONL, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                bases = as_list(d.get("base_model"))
                r = recs.get(d["id"])
                if r is None:
                    recs[d["id"]] = {
                        "id": d["id"], "bases": bases,
                        "relation": d.get("base_model_relation"),
                        "pipeline_tag": d.get("pipeline_tag"),
                    }
                else:
                    r["bases"] = r["bases"] or bases
                    r["relation"] = r["relation"] or d.get("base_model_relation")
                    r["pipeline_tag"] = r["pipeline_tag"] or d.get("pipeline_tag")
    else:
        print(f"Note: {OPTIONAL_METADATA_JSONL} not found; lineage coverage of the top {TOP_N} will be lower.")

    models = pd.DataFrame(recs.values())
    downloads = pd.read_csv(DOWNLOADS_CSV).set_index("id")["downloads_all_time"]
    models["downloads"] = models["id"].map(downloads).fillna(0).astype("int64")
    models["publisher"] = models["id"].map(publisher_of)
    models["has_parent"] = models["bases"].str.len() > 0
    models.loc[models["has_parent"] & models["relation"].isna(), "relation"] = "unspecified"
    return models, downloads


def read_csv_str(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def eci_gaps() -> pd.DataFrame:
    """Each ECI model's gap to the best model released on or before its release date."""
    e = pd.read_csv(ECI_SNAPSHOT_CSV, parse_dates=["release_date"]).sort_values(["release_date", "eci"])
    e["sota_at_release"] = e["eci"].cummax()
    e["gap_to_frontier"] = e["sota_at_release"] - e["eci"]
    return e


def classify_frontier(orgs: pd.DataFrame) -> pd.DataFrame:
    """Add capability_frontier / capability_evidence and frontier = capability OR compute."""
    e = eci_gaps()
    e = e[e["release_date"] >= CAPABILITY_START].assign(org=e["organization"].str.split(", ")).explode("org")
    orgs = orgs.copy()
    cap, ev = [], []
    for org, names in orgs["eci_organization"].items():
        m = e[e["org"].isin([n for n in names.split(";") if n])].sort_values("gap_to_frontier")
        if len(m):
            b = m.iloc[0]
            cap.append("y" if b["gap_to_frontier"] <= CAPABILITY_GAP_ECI else "n")
            ev.append(f"{'best' if cap[-1] == 'y' else 'closest'}: {b['model']} ({b['release_date']:%Y-%m}), "
                      f"ECI {b['eci']:.1f} vs frontier {b['sota_at_release']:.1f} (gap {b['gap_to_frontier']:.1f})")
        else:
            cap.append("n")
            ev.append("no model on Epoch's ECI leaderboard")
    orgs["capability_frontier"] = cap
    orgs["capability_evidence"] = ev
    orgs["frontier"] = np.where((orgs["capability_frontier"] == "y") | (orgs["compute_frontier"] == "y"), "y", "n")
    return orgs


def apply_undeclared_parents(models: pd.DataFrame, downloads: pd.Series) -> pd.DataFrame:
    """Add parent links that the Hub metadata omits but the model card itself documents.

    Only fills models with NO declared parent; never overrides a declared one.
    """
    models = models.copy()
    models["parent_source"] = np.where(models["has_parent"], "declared", "")
    extra = read_csv_str(UNDECLARED_PARENTS_CSV)
    idx = {m: i for i, m in enumerate(models["id"])}
    new_rows = []
    for r in extra.itertuples():
        if r.model_id in idx:
            i = models.index[idx[r.model_id]]
            if models.at[i, "has_parent"]:
                continue
            models.at[i, "bases"] = [r.parent_id]
            models.at[i, "relation"] = r.relation
            models.at[i, "has_parent"] = True
            models.at[i, "parent_source"] = "documented override"
        else:  # below the 100-download floor or otherwise not pulled
            new_rows.append({"id": r.model_id, "bases": [r.parent_id], "relation": r.relation,
                             "pipeline_tag": None, "downloads": int(downloads.get(r.model_id, 0)),
                             "publisher": publisher_of(r.model_id), "has_parent": True,
                             "parent_source": "documented override"})
    if new_rows:
        models = pd.concat([models, pd.DataFrame(new_rows)], ignore_index=True)
    return models


class Attribution:
    """Who a model is attributed to: model override -> account override -> author_types -> unlabeled."""

    def __init__(self, types: pd.DataFrame, overrides: pd.DataFrame | None, orgs: pd.DataFrame):
        self.types = types
        o = overrides if overrides is not None else pd.DataFrame(columns=["level", "target", "org", "publisher_type"])
        self.by_model = o[o["level"] == "model"].set_index("target")
        self.by_account = o[o["level"] == "account"].set_index("target")
        self.orgs = orgs

    def _one(self, model_id: str, field: str) -> str:
        if model_id in self.by_model.index:
            return self.by_model.at[model_id, field]
        pub = publisher_of(model_id)
        if pub in self.by_account.index:
            return self.by_account.at[pub, field]
        if pub in self.types.index and self.types.at[pub, field]:
            return self.types.at[pub, field]
        return UNLABELED if field == "publisher_type" else pub

    def company_group(self, org: str) -> str:
        """Split a company into the four groups using its two flags; unclassified orgs -> Other."""
        if org not in self.orgs.index:
            return OTHER_CO
        big = self.orgs.at[org, "big_tech"] == "y"
        frontier = self.orgs.at[org, "frontier"] == "y"  # computed in classify_frontier()
        return {(True, True): BIG_FRONTIER, (True, False): BIG_ONLY,
                (False, True): FRONTIER_ONLY, (False, False): OTHER_CO}[(big, frontier)]

    def _type(self, model_id: str) -> str:
        t = self._one(model_id, "publisher_type")
        return self.company_group(self._one(model_id, "org")) if t == COMPANY else t

    def type(self, model_ids: pd.Series) -> pd.Series:
        return model_ids.map(self._type)

    def org(self, model_ids: pd.Series) -> pd.Series:
        return model_ids.map(lambda m: self._one(m, "org"))


# %% Root resolution: follow declared parents up to the top of the chain
def add_roots(models: pd.DataFrame) -> pd.DataFrame:
    parent = {r.id: r.bases[0] for r in models.itertuples() if r.bases}

    def root(model_id):
        seen, cur = set(), model_id
        while cur in parent and cur not in seen:
            seen.add(cur)
            cur = parent[cur]
        return cur

    kids = models[models["has_parent"]].copy()
    kids["root"] = kids["id"].map(root)
    kids["root_publisher"] = kids["root"].map(publisher_of)
    return kids


# %% Q1: the top-313
def top_models_table(models, downloads, attr):
    top = downloads.sort_values(ascending=False).head(TOP_N)
    t = pd.DataFrame({"id": top.index, "downloads": top.values})
    t["rank"] = np.arange(1, len(t) + 1)
    t["publisher"] = t["id"].map(publisher_of)
    t["publisher_type"] = attr.type(t["id"])
    t["org"] = attr.org(t["id"])
    meta = models.set_index("id")
    t["in_metadata"] = t["id"].isin(meta.index)
    t["declared_parents"] = t["id"].map(lambda i: ";".join(meta.at[i, "bases"]) if i in meta.index else "")
    t["relation"] = t["id"].map(lambda i: meta.at[i, "relation"] if i in meta.index else None)
    t["lineage"] = np.select(
        [~t["in_metadata"], t["declared_parents"] != ""],
        ["Not in pulled metadata", "Declared derivative"],
        "No declared parent",
    )
    return t, downloads.sum()


# %% Figures
def style_axes(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d6d5d0")
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.xaxis.grid(True, color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_top_models(top, path):
    order = [p for p in FIGURE_ORDER if p in set(top["publisher_type"])]
    lineages = ["No declared parent", "Declared derivative", "Not in pulled metadata"]
    shades = {"No declared parent": 1.0, "Declared derivative": 0.55, "Not in pulled metadata": 0.2}
    counts = top.pivot_table(index="publisher_type", columns="lineage", values="id",
                             aggfunc="count", fill_value=0).reindex(order).fillna(0)
    dl_share = top.groupby("publisher_type")["downloads"].sum() / top["downloads"].sum()
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for lin in lineages:
        vals = counts[lin].values if lin in counts else np.zeros(len(order))
        for yi, p, l, v in zip(y, order, left, vals):
            if v:
                ax.barh(yi, v - 0.3, left=l, height=0.62, color=TYPE_COLORS[p],
                        alpha=shades[lin], edgecolor="none")
        left += vals
    for yi, p, tot in zip(y, order, left):
        share = dl_share.get(p, 0)
        share_txt = f"{share:.0%}" if share >= 0.005 else "<1%"
        ax.text(tot + 1.5, yi, f"{int(tot)} models · {share_txt} of top-{TOP_N} downloads",
                va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, order, fontsize=9.5, color=INK)
    ax.set_xlim(0, left.max() * 1.55)
    ax.set_xlabel("Number of models in the top 313", color=INK_2, fontsize=9)
    style_axes(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#6b6a65", alpha=shades[l], label=l) for l in lineages],
              loc="lower right", frameon=False, fontsize=8.5, title="Shade = lineage", title_fontsize=8.5)
    ax.set_title(f"Who publishes the top {TOP_N} most-downloaded models, and do they declare a parent?",
                 loc="left", fontsize=11.5, color=INK, pad=12)
    fig.text(0.01, 0.01, "Top 0.01% of ~3.1M Hub models by all-time downloads (Oct 8 2026 pull). "
             "'No declared parent' ≠ base model: many older fine-tunes never filled in base_model.",
             fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _band(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha):
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, alpha=alpha, edgecolor="none"))


def fig_lineage_flow(kids, path):
    """Three-column flow: base publisher type -> kind of derivative -> derivative publisher type."""
    cols = ["root_type", "relation", "child_type"]
    rel_order = RELATIONS + sorted(set(kids["relation"]) - set(RELATIONS) - {"unspecified"}) + ["unspecified"]
    orders = [
        [p for p in FIGURE_ORDER if p in set(kids["root_type"])],
        [r for r in rel_order if r in set(kids["relation"])],
        [p for p in FIGURE_ORDER if p in set(kids["child_type"])],
    ]
    total = len(kids)
    gap, node_w = 0.016, 0.022
    xs = [0.0, 0.5, 1.0]
    # node positions (stacked top->bottom, fraction of height)
    pos = []
    for col, order in zip(cols, orders):
        sizes = kids[col].value_counts().reindex(order).fillna(0) / total
        scale = (1 - gap * (len(order) - 1))
        top, d = 1.0, {}
        for k in order:
            h = sizes[k] * scale
            d[k] = [top, top - h]
            top -= h + gap
        pos.append(d)
    fig, ax = plt.subplots(figsize=(12, 6.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.05)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    # bands: colored by base publisher type, stacked in a consistent order
    for i in range(2):
        a, b = cols[i], cols[i + 1]
        flows = kids.groupby([a, b, "root_type"]).size() / total
        out_cursor = {k: v[0] for k, v in pos[i].items()}
        in_cursor = {k: v[0] for k, v in pos[i + 1].items()}
        scale0 = (1 - gap * (len(orders[i]) - 1))
        scale1 = (1 - gap * (len(orders[i + 1]) - 1))
        for ka in orders[i]:
            for kb in orders[i + 1]:
                for rt in orders[0]:
                    v = flows.get((ka, kb, rt), 0)
                    if not v:
                        continue
                    h0, h1 = v * scale0, v * scale1
                    y0, y1 = out_cursor[ka], in_cursor[kb]
                    _band(ax, xs[i] + node_w, y0, y0 - h0, xs[i + 1], y1, y1 - h1,
                          TYPE_COLORS[rt], 0.38)
                    out_cursor[ka] -= h0
                    in_cursor[kb] -= h1
    # nodes and labels
    min_gap = 0.03  # minimum vertical distance between adjacent labels (axes units)
    for ci, (col, order) in enumerate(zip(cols, orders)):
        counts = kids[col].value_counts()
        # label positions: node centres, pushed apart top-to-bottom so small nodes don't collide
        ys = []
        for k in order:
            y_top, y_bot = pos[ci][k]
            y = (y_top + y_bot) / 2
            if ys and ys[-1] - y < min_gap:
                y = ys[-1] - min_gap
            ys.append(y)
        for k, y in zip(order, ys):
            y_top, y_bot = pos[ci][k]
            color = "#6b6a65" if col == "relation" else TYPE_COLORS[k]
            ax.add_patch(plt.Rectangle((xs[ci], y_bot), node_w, y_top - y_bot, color=color, lw=0))
            pct = counts[k] / total
            label = f"{k}  {counts[k]:,} ({pct:.0%})" if pct >= 0.005 else f"{k}  {counts[k]:,} (<1%)"
            if ci == 0:
                ax.text(xs[ci] - 0.01, y, label, ha="right", va="center", fontsize=8.5, color=INK)
            elif ci == 1:
                ax.text(xs[ci] + node_w + 0.008, y, label, ha="left", va="center",
                        fontsize=8.5, color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.2))
            else:
                ax.text(xs[ci] + node_w + 0.01, y, label, ha="left", va="center", fontsize=8.5, color=INK)
    for x, h in zip(xs, ["Publisher of the root base model", "Kind of derivative", "Publisher of the derivative"]):
        ax.text(x + node_w / 2, 1.045, h, ha="center", fontsize=9.5, color=INK_2, weight="bold")
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(-0.01, 1.07)
    fig.suptitle(f"Where {total:,} declared derivative models trace back to (band color = root publisher type)",
                 x=0.02, y=0.985, ha="left", fontsize=12, color=INK)
    fig.text(0.02, 0.015, "Models with ≥100 downloads in the pulled metadata that declare a base_model. "
             "Each derivative is followed up its declared chain to the root. Counts, not downloads.",
             fontsize=7.5, color=INK_2)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_top_roots(roots, path, k=20):
    r = roots.head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(np.arange(len(r)), r["n_derivatives"] - 0.6, height=0.68,
            color=[TYPE_COLORS[t] for t in r["publisher_type"]])
    for yi, (n, rel) in enumerate(zip(r["n_derivatives"], r["top_relation"])):
        ax.text(n + 3, yi, f"{n:,}  (mostly {rel})", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(np.arange(len(r)), r.index, fontsize=8.5, color=INK)
    ax.set_xlim(0, r["n_derivatives"].max() * 1.3)
    ax.set_xlabel("Declared derivatives (all generations) in the pulled metadata", color=INK_2, fontsize=9)
    style_axes(ax)
    from matplotlib.patches import Patch
    present = [p for p in FIGURE_ORDER if p in set(r["publisher_type"])]
    ax.legend(handles=[Patch(color=TYPE_COLORS[p], label=p) for p in present],
              loc="lower right", frameon=False, fontsize=8)
    ax.set_title(f"Top {k} root base models by number of derivatives", loc="left", fontsize=11.5, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# %% Main
def derivatives_by_type(kids: pd.DataFrame) -> pd.DataFrame:
    by_type = kids.groupby("root_type").agg(derivatives=("id", "size"), derivative_downloads=("downloads", "sum"))
    by_type["share_of_derivatives"] = by_type["derivatives"] / len(kids)
    by_type["share_of_derivative_downloads"] = by_type["derivative_downloads"] / kids["downloads"].sum()
    return by_type.reindex([p for p in PUBLISHER_TYPES if p in by_type.index])


def run(verbose: bool = True) -> dict:
    """Run the full analysis, write outputs to OUT_DIR and return the key tables."""
    say = print if verbose else (lambda *a, **k: None)
    os.makedirs(OUT_DIR, exist_ok=True)
    types = read_csv_str(AUTHOR_TYPES_CSV).set_index("author")
    orgs = classify_frontier(read_csv_str(ORG_CLASSIFICATION_CSV).set_index("org"))
    orgs.to_csv(f"{OUT_DIR}/frontier_tests_by_company.csv")
    raw_models, downloads = load_models()
    say(f"Models with metadata: {len(raw_models):,}; declaring a parent: {raw_models['has_parent'].sum():,} "
          f"({raw_models['has_parent'].mean():.1%})")

    # Baseline: Hub metadata only (no documented corrections), kept for comparison
    base_attr = Attribution(types, None, orgs)
    base_kids = add_roots(raw_models)
    base_kids["root_type"] = base_attr.type(base_kids["root"])
    baseline = derivatives_by_type(base_kids)

    # Main run: apply evidence-backed corrections
    attr = Attribution(types, read_csv_str(PUBLISHER_OVERRIDES_CSV), orgs)
    models = apply_undeclared_parents(raw_models, downloads)
    say(f"Documented undeclared parents applied: {(models['parent_source'] == 'documented override').sum()}")

    # Q1
    top, all_dl = top_models_table(models, downloads, attr)
    top.to_csv(f"{OUT_DIR}/top{TOP_N}_models.csv", index=False)
    say(f"\nTop {TOP_N}: {top['downloads'].sum() / all_dl:.1%} of all Hub downloads")
    say(top["lineage"].value_counts().to_string())
    q1 = top.groupby("publisher_type").agg(models=("id", "size"), downloads=("downloads", "sum"))
    q1["download_share"] = q1["downloads"] / q1["downloads"].sum()
    q1 = q1.reindex([p for p in PUBLISHER_TYPES if p in q1.index])
    q1.to_csv(f"{OUT_DIR}/top{TOP_N}_by_publisher_type.csv")
    say(q1.to_string())
    fig_top_models(top.assign(publisher_type=for_figure(top["publisher_type"])), f"{OUT_DIR}/fig1_top{TOP_N}_publishers.png")

    # Q2
    kids = add_roots(models)
    kids["root_type"] = attr.type(kids["root"])
    kids["root_org"] = attr.org(kids["root"])
    kids["child_type"] = attr.type(kids["id"])
    kids.drop(columns=["bases"]).to_csv(f"{OUT_DIR}/derivatives_with_roots.csv", index=False)

    by_type = derivatives_by_type(kids)
    by_type.to_csv(f"{OUT_DIR}/derivatives_by_root_publisher_type.csv")
    say("\nDerivatives by root publisher type:\n" + by_type.to_string())

    effect = pd.DataFrame({
        "share_of_derivatives_metadata_only": baseline["share_of_derivatives"],
        "share_of_derivatives_with_corrections": by_type["share_of_derivatives"],
        "share_of_derivative_downloads_metadata_only": baseline["share_of_derivative_downloads"],
        "share_of_derivative_downloads_with_corrections": by_type["share_of_derivative_downloads"],
    }).reindex([p for p in PUBLISHER_TYPES if p in by_type.index or p in baseline.index]).fillna(0)
    effect.to_csv(f"{OUT_DIR}/effect_of_documented_corrections.csv")
    say("\nEffect of documented corrections:\n" + effect.round(3).to_string())

    # Headline under each test separately (the two flags overlap)
    n_all, d_all = len(kids), kids["downloads"].sum()
    per_test = {}
    for name, groups in [("Big Tech (market cap >= $100B)", {BIG_FRONTIER, BIG_ONLY}),
                         ("Frontier developer (capability or compute)", {BIG_FRONTIER, FRONTIER_ONLY}),
                         ("Either", {BIG_FRONTIER, BIG_ONLY, FRONTIER_ONLY})]:
        m = kids["root_type"].isin(groups)
        per_test[name] = {"share_of_derivatives": m.mean(),
                          "share_of_derivative_downloads": kids.loc[m, "downloads"].sum() / d_all}
        say(f"{name:34s} share of derivatives {m.mean():.1%}, of derivative downloads "
              f"{kids.loc[m, 'downloads'].sum() / d_all:.1%}")

    # Company-level table: the two flags next to each company's downstream reach
    co = kids[kids["root_type"].isin({BIG_FRONTIER, BIG_ONLY, FRONTIER_ONLY, OTHER_CO})]
    co_tab = co.groupby("root_org").agg(derivatives=("id", "size"), derivative_downloads=("downloads", "sum"))
    co_tab["group"] = co_tab.index.map(attr.company_group)
    co_tab = co_tab.join(orgs[["listing", "market_cap_usd_bn", "big_tech", "capability_frontier", "compute_frontier",
                                  "frontier", "hq_country"]], how="left")
    co_tab.sort_values("derivatives", ascending=False).to_csv(f"{OUT_DIR}/derivatives_by_company.csv")

    # How much of the ecosystem's *own uploads* do these publishers account for?
    models["type"] = attr.type(models["id"])
    own = models.groupby("type").size() / len(models)
    reach = pd.DataFrame({"share_of_all_models_published": own,
                          "share_of_derivatives_rooted_in_their_models": by_type["share_of_derivatives"]})
    reach = reach.reindex([p for p in PUBLISHER_TYPES if p in reach.index])
    reach.to_csv(f"{OUT_DIR}/reach_vs_own_uploads.csv")
    say("\nOwn uploads vs. downstream reach:\n" + reach.to_string())

    by_org = (kids.groupby("root_org").agg(derivatives=("id", "size"), derivative_downloads=("downloads", "sum"))
              .sort_values("derivatives", ascending=False))
    by_org.head(50).to_csv(f"{OUT_DIR}/derivatives_by_root_org_top50.csv")

    roots = kids.groupby("root").agg(n_derivatives=("id", "size"), derivative_downloads=("downloads", "sum"),
                                     top_relation=("relation", lambda s: s.value_counts().index[0]))
    roots["publisher"] = roots.index.map(publisher_of)
    roots["publisher_type"] = attr.type(roots.index.to_series())
    roots = roots.sort_values("n_derivatives", ascending=False)
    roots.head(100).to_csv(f"{OUT_DIR}/top100_root_models.csv")

    fig_lineage_flow(kids.assign(root_type=for_figure(kids["root_type"]), child_type=for_figure(kids["child_type"])), f"{OUT_DIR}/fig2_lineage_flow.png")
    fig_top_roots(roots.assign(publisher_type=for_figure(roots["publisher_type"])), f"{OUT_DIR}/fig3_top_root_models.png")
    say(f"\nWrote outputs to {OUT_DIR}/")
    return {"models": models, "top": top, "top_by_type": q1, "kids": kids, "by_type": by_type,
            "baseline": baseline, "effect": effect, "per_test": pd.DataFrame(per_test).T,
            "companies": co_tab.sort_values("derivatives", ascending=False), "reach": reach,
            "by_org": by_org, "roots": roots, "orgs": orgs}


def main():
    run()




if __name__ == "__main__":
    main()
