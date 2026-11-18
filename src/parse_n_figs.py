"""
Parses sheets of the scoping review Excel workbook and generates summary plots,
cross-tabulated breakdowns, and LaTeX tables.
"""

import argparse
from ast import literal_eval
from collections import Counter
from itertools import chain
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DTYPE_CLUSTER_LABELS = {
    "only_poldocs": "Only Policy Docs",
    "only_poldocs_and_admin": "Only Policy & Admin Docs",
    "the_rest": "Remaining Dtype Combos",
}

TECH_CLUSTER_LABELS = {
    "only_corenlp": "Only CoreNLP",
    "only_deepnlp": "Only DeepNLP",
    "only_genai": "Only GenAI",
    "incl_kgonto": "Includes KG / Ontologies",
    "the_rest": "Remaining Tech Combos",
}

DTYPE_LABELS = {
    "POLDOCS": "Public Policy Documents",
    "ADMINDOCS": "Public Administration Records",
    "SCIENTIF": "Scientific Literature",
    "MASSMEDIA": "News & Mass Media",
    "PUBLICOPIN": "Social Media & Public Input",
    "QUANT": "Quantitative Data",
    "SEPDATA": "Quantitative Data",
    "OTHER": "Other",
}

DOMAIN_LABELS = {
    "MACECON": "Macroeconomics",
    "CIVRIGHT": "Civil Rights",
    "HEALTH": "Health",
    "AGRI": "Agriculture",
    "LABOR": "Labor",
    "EDUCAT": "Education",
    "ENVIRON": "Environment",
    "ENERGY": "Energy",
    "IMMIGR": "Immigration",
    "TRANSPORT": "Transportation",
    "LAW": "Law and Crime",
    "SOCWELF": "Social Welfare",
    "HOUSING": "Housing",
    "DOMCOMM": "Domestic Commerce",
    "DEFENS": "Defense",
    "TECH": "Technology",
    "FORTRD": "Foreign Trade",
    "INTLAFF": "International Affairs",
    "GOVTOPS": "Government Operations",
    "PUBLAND": "Public Lands",
    "CULTUR": "Culture",
    "DOMAGN": "Domain Agnostic",
    "POLINS": "Political Insights",
    "OTHER": "Other",
}

TASK_LABELS = {
    "PROB_ANALYSIS": "Problem Analysis",
    "POLDOC_POLDES": "PDA: Design",
    "POLDOC_OTHER": "PDA: General",
    "INSGHT_PUBLIC": "PI: Public Opinion",
    "INSGHT_GOVT": "PI: Government",
    "FORECAST": "Forecasting",
    "PRESCRIPT": "Prescription",
    "MONITOR": "Monitoring",
    "EVALUATE": "Evaluation",
    "OTHER": "Other",
}

TECH_METHODS = {
    "CORENLP": "Core NLP",
    "DEEPNLP": "Deep NLP",
    "KGONTO": "KGs & Ontologies",
    "GENAI": "Generative AI",
}

LIST_COLUMNS = [
    "META TASK LIST",
    "POLICY AREA LIST",
    "META TECH LIST",
    "META DATA LIST",
]

# Per-column metadata: (singular_label, plural_label, short_slug, default_bins, label_replacements)
PLOT_COUNT_CONFIGS = {
    "META TASK LIST": {
        "singular": "Task",
        "plural": "Tasks",
        "slug": "task",
        "n_bins": 3,
        "overflow": True,
        "replacements": TASK_LABELS,
    },
    "POLICY AREA LIST": {
        "singular": "Domain",
        "plural": "Domains",
        "slug": "area",
        "n_bins": 3,
        "overflow": True,
        "replacements": DOMAIN_LABELS,
    },
    "META TECH LIST": {
        "singular": "Tech class",
        "plural": "Tech classes",
        "slug": "tech",
        "n_bins": 3,
        "overflow": True,
        "replacements": TECH_METHODS,
    },
    "META DATA LIST": {
        "singular": "Data Type",
        "plural": "Data Types",
        "slug": "dtype",
        "n_bins": 4,
        "overflow": True,
        "replacements": DTYPE_LABELS,
    },
}

def get_label_map(colname):
    """Helper to fetch the display label dictionary for a given column."""
    return PLOT_COUNT_CONFIGS.get(colname, {}).get("replacements", {})

# ---------------------------------------------------------------------------
# Data Loading & Core Counting Utilities
# ---------------------------------------------------------------------------

def _parse_list_cell(val):
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    return list(literal_eval(str(val)))


def load_sheet(excel_path, sheet_name="POOL 8", index_col="IDX", list_cols=LIST_COLUMNS):
    """Loads an Excel sheet and parses stringified list columns once."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name, index_col=index_col)
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list_cell)
    return df


def count_items(df, colname, replacements=None):
    """Returns a Counter of items across a pre-parsed list column."""
    replacements = replacements or {}
    items = (replacements.get(x, x) for x in chain.from_iterable(df[colname]))
    return Counter(items)


def sorted_item_counts(df, colname, replacements=None):
    """Returns [(item, count), ...] sorted descending by frequency."""
    return count_items(df, colname, replacements).most_common()


def get_label_paperids(df, colname, replacements=None):
    """Maps each item label in `colname` to the list of paper IDs containing it."""
    replacements = replacements or {}
    label_pids = {}
    for pid, items in df[colname].items():
        for raw_val in items:
            val = replacements.get(raw_val, raw_val)
            label_pids.setdefault(val, []).append(pid)
    return label_pids


# ---------------------------------------------------------------------------
# Breakdown & Binning Builders (Output: PID groups or 2D Count Matrices)
# ---------------------------------------------------------------------------

def bucket_by_item_count(df, colname, n_bins=3, overflow=True, singular="Item", plural="Items"):
    """
    Groups paper IDs by how many items they have in `colname`.
    - overflow=True:  ['1 Item', '2 Items', '3+ Items'] (for n_bins=3)
    - overflow=False: ['1 Item', '2 Items', '3 Items']
    """
    def bin_label(k):
        if overflow and k >= n_bins:
            return f"{n_bins}+ {singular if n_bins == 1 else plural}"
        return f"{k} {singular if k == 1 else plural}"

    labels = [bin_label(k) for k in range(1, n_bins + 1)]
    buckets = {lbl: [] for lbl in labels}

    for pid, items in df[colname].items():
        k = len(items)
        if k == 0:
            continue
        if overflow and k >= n_bins:
            buckets[labels[-1]].append(pid)
        elif 1 <= k <= n_bins:
            buckets[bin_label(k)].append(pid)

    return buckets


def build_count_bin_matrix(df, colname, n_bins=3, overflow=True, singular="Item", plural="Items", replacements=None):
    """Builds a DataFrame (rows=sorted items, cols=count bins) for stacked plotting."""
    replacements = replacements or {}
    overall = sorted_item_counts(df, colname, replacements)
    item_order = [item for item, _ in overall]
    buckets = bucket_by_item_count(df, colname, n_bins, overflow, singular, plural)

    data = {}
    for bin_name, pids in buckets.items():
        sub_counts = count_items(df.loc[pids], colname, replacements) if pids else Counter()
        data[bin_name] = [sub_counts.get(item, 0) for item in item_order]

    return pd.DataFrame(data, index=item_order)


def build_group_breakdown_matrix(df, colname, pid_groups, group_labels=None, replacements=None):
    """Builds a DataFrame (rows=sorted items in `colname`, cols=custom PID groups)."""
    replacements = replacements or {}
    group_labels = group_labels or {k: k for k in pid_groups}
    overall = sorted_item_counts(df, colname, replacements)
    item_order = [item for item, _ in overall]

    data = {}
    for key, display_name in group_labels.items():
        pids = pid_groups.get(key, [])
        sub_counts = count_items(df.loc[pids], colname, replacements) if pids else Counter()
        data[display_name] = [sub_counts.get(item, 0) for item in item_order]

    return pd.DataFrame(data, index=item_order)


def build_column_cross_matrix(df, x_col, stack_col, x_replacements=None, stack_replacements=None):
    """Builds a DataFrame (rows=items in `x_col`, cols=items in `stack_col`)."""
    x_sorted = sorted_item_counts(df, x_col, x_replacements)
    x_order = [item for item, _ in x_sorted]
    x_pid_map = get_label_paperids(df, x_col, x_replacements)

    stack_sorted = sorted_item_counts(df, stack_col, stack_replacements)
    stack_order = [item for item, _ in stack_sorted]

    rows = []
    for x_item in x_order:
        pids = x_pid_map.get(x_item, [])
        sub_counts = count_items(df.loc[pids], stack_col, stack_replacements) if pids else Counter()
        rows.append([sub_counts.get(s_item, 0) for s_item in stack_order])

    return pd.DataFrame(rows, index=x_order, columns=stack_order)


def build_pid_cross_matrix(x_groups, s_groups, x_labels=None, s_labels=None):
    """Builds a DataFrame of set-intersection counts between two PID bucket dicts."""
    x_labels = x_labels or {k: k for k in x_groups}
    s_labels = s_labels or {k: k for k in s_groups}
    s_sets = {k: set(v) for k, v in s_groups.items()}

    rows = []
    for x_key in x_labels:
        x_set = set(x_groups.get(x_key, []))
        rows.append([len(x_set & s_sets.get(s_key, set())) for s_key in s_labels])

    return pd.DataFrame(rows, index=list(x_labels.values()), columns=list(s_labels.values()))


# ---------------------------------------------------------------------------
# Unified Plotting Engine
# ---------------------------------------------------------------------------

def plot_bar_counts(sorted_tuples, title="", filename="fig_bar.png", palette="deep", show=False):
    """Plots a simple sorted bar chart from [(label, count), ...]."""
    sns.set_theme(style="whitegrid")
    names = [i[0] for i in sorted_tuples]
    counts = [i[1] for i in sorted_tuples]

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(x=names, y=counts, hue=names, palette=palette, legend=False, ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fontsize=12, padding=2)
    if counts:
        ax.set_ylim(0, max(counts) * 1.12)

    sns.despine(ax=ax)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_stacked_bar(matrix_df, title="", filename="fig_stacked.png", palette="deep", figsize=(7, 7), show=False):
    """
    Unified stacked bar chart renderer.
    - matrix_df.index   -> x-axis categories
    - matrix_df.columns -> stacked bar segments (bottom to top)
    """
    sns.set_theme(style="whitegrid")
    x_names = list(matrix_df.index)
    stack_cols = list(matrix_df.columns)
    colors = sns.color_palette(palette, n_colors=len(stack_cols))

    fig, ax = plt.subplots(figsize=figsize)
    bottom = [0] * len(x_names)
    last_container = None

    for idx, col in enumerate(stack_cols):
        counts = matrix_df[col].tolist()
        r, g, b = colors[idx]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "#222222" if luminance > 0.6 else "white"

        last_container = ax.bar(
            x_names,
            counts,
            bottom=bottom,
            label=col,
            color=colors[idx],
            edgecolor="white",
            linewidth=0.8,
            width=0.65,
        )
        seg_labels = [str(c) if c > 0 else "" for c in counts]
        ax.bar_label(
            last_container,
            labels=seg_labels,
            label_type="center",
            fontsize=9,
            color=text_color,
            fontweight="bold",
        )
        bottom = [b + c for b, c in zip(bottom, counts)]

    if last_container is not None and bottom:
        ax.bar_label(
            last_container,
            labels=[str(t) for t in bottom],
            label_type="edge",
            fontsize=12,
            padding=2,
        )
        ax.set_ylim(0, max(bottom) * 1.15)

    ax.xaxis.grid(False)
    sns.despine(ax=ax)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    ax.legend(frameon=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Domain-Specific Breakdowns & Queries
# ---------------------------------------------------------------------------

def get_grouping_breakdown(df, query):
    """Evaluates scoping-review slice queries over pre-parsed list columns."""
    if query == "task_poldesoth":
        tasks = {pid: set(vals) for pid, vals in df["META TASK LIST"].items()}
        incl_des = [p for p, s in tasks.items() if "POLDOC_POLDES" in s]
        incl_oth = [p for p, s in tasks.items() if "POLDOC_OTHER" in s]
        return {
            "pids": {
                "INCL_poldoc_poldes_OR_poldoc_other": [p for p, s in tasks.items() if s & {"POLDOC_POLDES", "POLDOC_OTHER"}],
                "ONLY_poldoc_poldes_OR_poldoc_other": [p for p, s in tasks.items() if s and s <= {"POLDOC_POLDES", "POLDOC_OTHER"}],
                "INCL_poldoc_poldes": incl_des,
                "INCL_poldoc_other": incl_oth,
                "ONLY_poldoc_poldes": [p for p, s in tasks.items() if s == {"POLDOC_POLDES"}],
                "ONLY_poldoc_other": [p for p, s in tasks.items() if s == {"POLDOC_OTHER"}],
                "INCL_poldoc_poldes_AND_poldoc_other": [p for p, s in tasks.items() if {"POLDOC_POLDES", "POLDOC_OTHER"} <= s],
                "ONLY_poldoc_poldes_AND_poldoc_other": [p for p, s in tasks.items() if s == {"POLDOC_POLDES", "POLDOC_OTHER"}],
            }
        }

    if query == "dtype_typecounts":
        dtypes = df["META DATA LIST"]
        return {
            "pids": {
                "1_dtypes": [p for p, v in dtypes.items() if v == ["POLDOCS"]],
                "+1_dtypes": [p for p, v in dtypes.items() if len(v) > 1],
                "2_dtypes": [p for p, v in dtypes.items() if len(v) == 2],
                "3_dtypes": [p for p, v in dtypes.items() if len(v) == 3],
                "4_dtypes": [p for p, v in dtypes.items() if len(v) == 4],
                ">4_dtypes": [p for p, v in dtypes.items() if len(v) > 4],
            }
        }

    if query == "dtype_overlaps":
        dtypes = {pid: set(vals) for pid, vals in df["META DATA LIST"].items()}
        pairs = [("SCIENTIF", "sci"), ("ADMINDOCS", "admin"), ("SEPDATA", "quant"), ("MASSMEDIA", "mass"), ("PUBLICOPIN", "pubop")]
        triples = [
            ("ADMINDOCS", "SCIENTIF", "admin_AND_sci"),
            ("ADMINDOCS", "SEPDATA", "admin_AND_sep"),
            ("ADMINDOCS", "MASSMEDIA", "admin_AND_mass"),
            ("ADMINDOCS", "PUBLICOPIN", "admin_AND_pubop"),
            ("SEPDATA", "SCIENTIF", "quant_AND_sci"),
            ("SEPDATA", "MASSMEDIA", "quant_AND_mass"),
            ("SEPDATA", "PUBLICOPIN", "quant_AND_pubop"),
            ("SCIENTIF", "MASSMEDIA", "sci_AND_mass"),
            ("SCIENTIF", "PUBLICOPIN", "sci_AND_pubop"),
            ("MASSMEDIA", "PUBLICOPIN", "mass_AND_pubop"),
        ]
        pids_out = {f"ONLY_pd_AND_{slug}": [p for p, s in dtypes.items() if len(s) == 2 and tag in s] for tag, slug in pairs}
        pids_out.update({f"ONLY_pd_AND_{slug}": [p for p, s in dtypes.items() if len(s) == 3 and {a, b} <= s] for a, b, slug in triples})
        return {
            "pids": pids_out,
            "the_rest": {"4_dtypes+pd": {p: sorted(s) for p, s in dtypes.items() if len(s) == 4}},
        }

    if query == "tech_typecounts":
        techs = {pid: set(vals) for pid, vals in df["META TECH LIST"].items()}
        return {
            "pids": {
                "1_tech": [p for p, s in techs.items() if len(s) == 1],
                "2_tech": [p for p, s in techs.items() if len(s) == 2],
                "3_tech": [p for p, s in techs.items() if len(s) == 3],
                "4_tech": [p for p, s in techs.items() if len(s) == 4],
                "ONLY_corenlp": [p for p, s in techs.items() if s == {"CORENLP"}],
                "ONLY_deepnlp": [p for p, s in techs.items() if s == {"DEEPNLP"}],
                "ONLY_genai": [p for p, s in techs.items() if s == {"GENAI"}],
                "ONLY_kgonto": [p for p, s in techs.items() if s == {"KGONTO"}],
                "INCL_kgonto": [p for p, s in techs.items() if "KGONTO" in s],
            }
        }

    if query == "tech_overlaps":
        techs = {pid: set(vals) for pid, vals in df["META TECH LIST"].items()}
        pairs = [
            ("CORENLP", "DEEPNLP", "corenlp_AND_deepnlp"),
            ("CORENLP", "GENAI", "corenlp_AND_genai"),
            ("CORENLP", "KGONTO", "corenlp_AND_kgonto"),
            ("DEEPNLP", "GENAI", "deepnlp_AND_genai"),
            ("DEEPNLP", "KGONTO", "deepnlp_AND_kgonto"),
            ("GENAI", "KGONTO", "genai_AND_kgonto"),
        ]
        return {
            "pids": {f"ONLY_{slug}": [p for p, s in techs.items() if s == {a, b}] for a, b, slug in pairs},
            "the_rest": {"3_tech": {p: sorted(s) for p, s in techs.items() if len(s) == 3}},
        }

    if query == "domagn_only":
        return {"pids": {"domagn_only": [p for p, v in df["POLICY AREA LIST"].items() if v == ["DOMAGN"]]}}

    raise ValueError(f"Unknown query: {query}")


def get_desired_dtype_breakdown(df):
    only_poldocs = get_grouping_breakdown(df, "dtype_typecounts")["pids"]["1_dtypes"]
    only_pd_admin = get_grouping_breakdown(df, "dtype_overlaps")["pids"]["ONLY_pd_AND_admin"]
    excluded = set(only_poldocs) | set(only_pd_admin)
    return {
        "only_poldocs": only_poldocs,
        "only_poldocs_and_admin": only_pd_admin,
        "the_rest": [idx for idx in df.index if idx not in excluded],
    }


def get_desired_tech_breakdown(df):
    tc = get_grouping_breakdown(df, "tech_typecounts")["pids"]
    groups = {
        "only_corenlp": tc["ONLY_corenlp"],
        "only_deepnlp": tc["ONLY_deepnlp"],
        "only_genai": tc["ONLY_genai"],
        "incl_kgonto": tc["INCL_kgonto"],
    }
    excluded = set().union(*groups.values())
    groups["the_rest"] = [idx for idx in df.index if idx not in excluded]
    return groups


def create_latex_table_content(df):
    """Generates LaTeX rows for all papers in `df`."""
    lines = []
    for pid, row in df.iterrows():
        cols = []
        for colname in ["META TECH LIST", "META DATA LIST", "META TASK LIST", "POLICY AREA LIST"]:
            repl = PLOT_COUNT_CONFIGS[colname]["replacements"]
            vals = [repl.get(v, v).replace("_", r"\_") for v in row[colname]]
            cols.append(", ".join(vals))
        pidx = str(pid).lower().replace(" ", "_")
        lines.append(rf"\shortcite{{{pidx}}} & {' & '.join(cols)} \\")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# High-Level Batch Runners
# ---------------------------------------------------------------------------

def generate_simple_bar_plots(df, output_dir, suffix="", palette="deep", show=False):
    for colname, cfg in PLOT_COUNT_CONFIGS.items():
        counts = sorted_item_counts(df, colname, cfg["replacements"])
        tag = f"_{suffix}" if suffix else ""
        plot_bar_counts(
            counts,
            filename=Path(output_dir) / f"fig_bar_meta_{cfg['slug']}{tag}.png",
            palette=palette,
            show=show,
        )

def generate_stacked_count_plots(df, output_dir, bins_override=None, suffix="", palette="deep", figsize=(7,7), show=False):
    """
    Generates stacked count-bin bar plots for all configured columns.
    `bins_override` allows per-column bin customization, e.g. {"META TASK LIST": 4, "POLICY AREA LIST": 2}.
    """
    bins_override = bins_override or {}
    for colname, cfg in PLOT_COUNT_CONFIGS.items():
        if cfg['slug'] in ["data", "dtype"]:
            figsize = (7,8)
        n_bins = bins_override.get(colname, cfg["n_bins"])
        matrix = build_count_bin_matrix(
            df,
            colname=colname,
            n_bins=n_bins,
            overflow=cfg["overflow"],
            singular=cfg["singular"],
            plural=cfg["plural"],
            replacements=cfg["replacements"],
        )
        tag = f"_{suffix}" if suffix else ""
        plot_stacked_bar(
            matrix,
            filename=Path(output_dir) / f"fig_stacked_bar_meta_{cfg['slug']}_{n_bins}bins{tag}.png",
            palette=palette,
            figsize=figsize,
            show=show,
        )


def run_pipeline(
    excel_path="inputs/Systematic_Review_Worksheet.xlsx",
    sheet_name="POOL 8",
    output_root="outputs",
    bins_override=None,
    palette="deep",
    show=False,
):
    safe_sheet = sheet_name.replace(" ", "_")
    output_dir = Path(output_root) / safe_sheet
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_sheet(excel_path, sheet_name=sheet_name)

    # 1. Simple frequency bar charts
    #generate_simple_bar_plots(df, output_dir, palette=palette, show=show)

    # 2. Stacked item-count bin charts (customizable bins per column)
    generate_stacked_count_plots(df, output_dir, bins_override=bins_override, palette=palette, show=show)

    # 3. Domain-specific breakdowns
    dtype_bd = get_desired_dtype_breakdown(df)
    tech_bd = get_desired_tech_breakdown(df)
    
    for colname, slug in [("META TASK LIST", "task")]:#, ("META TECH LIST", "tech"), ("POLICY AREA LIST", "area")]:
        mat = build_group_breakdown_matrix(df, colname, dtype_bd, DTYPE_CLUSTER_LABELS, get_label_map(colname))
        plot_stacked_bar(mat, filename=output_dir / f"fig_bar_{slug}_by_dtype.png", palette=palette, show=show)

    for colname, slug in [("META TASK LIST", "task")]:#, ("POLICY AREA LIST", "area"), ("META DATA LIST", "data")]:
        mat = build_group_breakdown_matrix(df, colname, tech_bd, TECH_CLUSTER_LABELS, get_label_map(colname))
        plot_stacked_bar(mat, filename=output_dir / f"fig_bar_{slug}_by_tech.png", palette=palette, show=show)

    # 4. Cross-breakdowns]:#
    '''
    plot_stacked_bar(
        build_pid_cross_matrix(dtype_bd, tech_bd, DTYPE_CLUSTER_LABELS, TECH_CLUSTER_LABELS),
        filename=output_dir / "fig_stacked_bar_dtype_by_tech_group.png",
        palette=palette,
        show=show,
    )
    plot_stacked_bar(
        build_pid_cross_matrix(tech_bd, dtype_bd, TECH_CLUSTER_LABELS, DTYPE_CLUSTER_LABELS),
        filename=output_dir / "fig_stacked_bar_tech_by_dtype_group.png",
        palette=palette,
        show=show,
    )
    
    plot_stacked_bar(
        build_column_cross_matrix(
            df,
            "POLICY AREA LIST",
            "META TASK LIST",
            x_replacements=get_label_map("POLICY AREA LIST"),
            stack_replacements=get_label_map("META TASK LIST"),
        ),
        filename=output_dir / "fig_stacked_bar_area_by_task.png",
        palette=palette,
        show=show,
    )
    '''
    print(f"[{sheet_name}] All plots saved to: {output_dir}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoping review charting and plotting pipeline.")
    parser.add_argument("--excel", default="inputs/Systematic_Review_Workbook.xlsx", help="Path to Excel workbook")
    parser.add_argument("--sheet", default="POOL 8", help="Excel sheet name to examine")
    parser.add_argument("--out", default="outputs", help="Base output directory")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    custom_bins = {
        "META TASK LIST": 3, #3 for both
        "POLICY AREA LIST": 3, #3 for 70, 4 for 121
        "META TECH LIST": 3, #3 for both
        "META DATA LIST": 4, #4 for both
    }

    run_pipeline(
        excel_path=args.excel,
        sheet_name=args.sheet,
        output_root=args.out,
        bins_override=custom_bins,
        show=args.show,
    )
    '''
    python tellingthestory.py --sheet "POOL 9"

    from tellingthestory import run_pipeline
    run_pipeline(
        sheet_name="POOL 9",
        bins_override={"META TASK LIST": 2, "POLICY AREA LIST": 4},
    )

    df = load_sheet("inputs/Systematic_Review_Workbook.xlsx", sheet_name="POOL 8")
    mat = build_count_bin_matrix(df, "META TECH LIST", n_bins=5, overflow=False, singular="Tech", plural="Techs")
    plot_stacked_bar(mat, title="Tech Stack Counts", filename="outputs/custom_tech.png", show=True)
    '''