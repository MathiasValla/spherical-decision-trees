# Pattern Recognition Letters Submission Checklist

Checked on 2026-05-19 against the current ScienceDirect Guide for Authors and
Elsevier LaTeX instructions.

- Manuscript format: `spherical_trees_letter.tex` uses Elsevier `elsarticle`
  in double-column `5p` layout and sets `\journal{Pattern Recognition Letters}`.
- Page limit: `spherical_trees_letter.pdf` compiles to 4 pages, below PRL's
  7-page limit including text, figures, tables, and references.
- Title page: author, corresponding-author marker, corresponding-author e-mail,
  and affiliation are present.
- Abstract: 177 words, below the current 250-word guide limit and the older
  PRL 200-word formatting note.
- Keywords: 5 keywords, within the required 1 to 7 range.
- Highlights: `highlights.txt` contains 5 bullet highlights; each is under 85
  characters, including spaces.
- Graphical abstract: `graphical_abstract.pdf` and `graphical_abstract.tiff`
  are separate submission assets, with a `Graphical_Abstract.png` preview.
  The generated bitmap is 1328 x 531 pixels (width x height), satisfying
  Elsevier's 531 x 1328 pixel minimum (height x width) by matching that aspect
  ratio.
- Artwork/source layout: submission-facing figures are copied to the top level
  of `paper/`, matching Elsevier's instruction to avoid subfolders in LaTeX
  source uploads.
- Data/code: the manuscript states that PMLB data are public and that the
  open-source `treeple` implementation and reproduction scripts are available
  at https://github.com/MathiasValla/spherical-decision-trees, with
  `REPRODUCING_RESULTS.md` documenting how to retrieve or regenerate all
  article results.
- Declarations: data/code availability, competing interest, funding,
  acknowledgments, and generative-AI-use declarations are included before the
  references.
- Authorship confirmation: PRL requires the official confirmation form as a
  separate upload. Complete it in Editorial Manager before submission.

Note: the legacy PRL formatting PDF mentions an additional `prletter`/`prletters`
style file. The current public Guide for Authors points to Elsevier's LaTeX
template, and this repository vendors `elsarticle.cls` and `elsarticle-num.bst`
for reproducibility. If Editorial Manager requests the PRL-specific style file,
use the latest file supplied by Elsevier support rather than old third-party
copies.
