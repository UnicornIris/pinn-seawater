# Paper scaffold

```
paper/
  main.tex              entry point; swap the preamble for a venue template later
  references.bib         several entries marked TODO -- verify before submitting
  sections/
    00_abstract.tex
    01_introduction.tex   skeleton + TODOs, needs your framing
    02_related_work.tex   drafted from the arXiv lit search (verify citations)
    03_method.tex         mostly filled in from the code, a few TODOs
    04_experiments.tex    pulls in tables/, some TODOs for figures
    05_discussion.tex     skeleton + TODOs, this is where your argument goes
    06_conclusion.tex     skeleton
  tables/
    make_tables.py        regenerates *.tex tables from the result CSVs
    main_results.tex       generated -- do not hand-edit
    ablation1_summary.tex  generated -- do not hand-edit
    ablation2_summary.tex  generated -- do not hand-edit
  figures/                empty; main.tex's \graphicspath also points at
                           qcpinn_results/ and pinn_results/ run directories
                           directly, so plots don't need to be copied here
```

## Workflow

- After any change to the result CSVs (e.g. more seeds), rerun:
  ```bash
  python3 tables/make_tables.py
  ```
  to keep the in-paper tables in sync. Never hand-edit the generated
  `tables/*.tex` files -- edit `make_tables.py` instead.

- No local LaTeX toolchain was found on this machine (`pdflatex`/`latexmk`
  missing). Two options to actually compile:
  1. Upload `paper/` to Overleaf (drag the folder in, or `zip -r paper.zip
     paper/` and import the zip). Easiest, and matches the earlier
     recommendation to use Overleaf for collaboration with your advisor.
  2. Install a local TeX distribution, e.g. `brew install --cask basictex`
     (small, ~100MB) or `brew install --cask mactex` (full, ~4GB), then
     `make pdf` from this directory. Ask before running either -- it's a
     real install, not something to do silently.

- Every `\todo{...}` in the section files (rendered in red once compiled)
  marks something that needs your own judgment, a number to confirm from
  the code, or a citation to verify -- grep for `todo` across `sections/`
  to get the full list.

## Known gaps to close before this is submittable

- `references.bib`: several entries are stub `@article{key}` with no
  fields (`han2026seawater`, `qcpinn2025reservoir`, `hqcpinn2026hydro`,
  `hqpinn2025flow`, `qpinnmac2025`, `qpinn2026fractional`) -- these were
  identified via web search summaries, not full-text reads. Confirm author
  lists and venue before citing them in a real submission.
- Target venue not yet chosen -- `main.tex` uses a generic `article` class.
  Once you and your advisor settle on arXiv / a journal / a workshop,
  swap in that venue's class file and re-check margins, section numbering,
  and citation style (`natbib` may need to become `biblatex` or a
  venue-specific macro set).
- Ablation-2 numbers in `tables/ablation2_summary.tex` come from
  `qcpinn_results/ablation2_20260729_225346/` (the complete run). The
  untracked `qcpinn_results/ablation2_20260730_160554/` directory (3 rows,
  interrupted rerun) is intentionally not used -- confirmed with you
  earlier in the session.
