# Final report and presentation sources

These English sources are **DRAFT** material for #85. The
[candidate/evidence ledger](../final-delivery.md) is the delivery status record.
They build on the earlier report's content, team metadata and course structure;
the original report and PDF decks were read/copied without modification. The
report follows Introduction, Methodology and Self-evaluation, with an EDN-status
annex. It does not reproduce pending decision entries or a final team assessment.

The report uses pdfLaTeX. The 16:9 presentation uses Beamer with editable TikZ and
pgfplots figures. This reuses the installed TeX workflow and needs no model,
dataset, browser, service, network access or new Python dependencies. The PDF
presentation is static; its source and [speaker guide](speaker-guide.txt) are
editable. Timing totals 900 seconds and still needs a human rehearsal.

From the repository root, using an installed Python 3.11+, pdfLaTeX, Poppler
(`pdfinfo`, `pdftotext`) and qpdf:

```sh
python3 docs/delivery/build.py --output temporary/issue-85/artifacts
```

The builder refuses output outside ignored `temporary/` and leaves the source,
data, caches and original materials unchanged. It reads versioned JSON for the
coverage, serving and outage figures, checks their scope, writes figure data only
to the output directory and compiles each PDF twice in sequence. No TeX package
installation or shell escape occurs. Build logs and a source/output hash manifest
are local. An existing output directory can be rebuilt; archive the preceding
draft first if its evidence must be retained. Output names are stable:

- `endesa-final-report-DRAFT.pdf`
- `endesa-final-presentation-DRAFT.pdf`
- `endesa-delivery-sources-DRAFT.zip`
- `build-manifest.json` and build logs

The ZIP contains editable delivery sources, versioned figure inputs, the delivery
runbooks and README/cards/requirements. It is a document-source package, not a
complete software/data/model replication bundle. Extract it into an empty folder
and use the same command to rebuild. `git archive <reviewed-sha>` supplies the
separate software snapshot when an authorized final package is prepared. Raw
datasets, models, credentials and proposed EDNs are excluded.

The builder checks PDF structure and extracted content, the report's conservative
total page count (at most 30 including annexes), and exactly 16 slide pages.
Render all pages for visual inspection:

```sh
mkdir -p temporary/issue-85/qa/report temporary/issue-85/qa/slides
pdftoppm -r 110 -png temporary/issue-85/artifacts/endesa-final-report-DRAFT.pdf \
  temporary/issue-85/qa/report/page
pdftoppm -r 120 -png temporary/issue-85/artifacts/endesa-final-presentation-DRAFT.pdf \
  temporary/issue-85/qa/slides/slide
```

Open every rendered page/slide, check diagrams and table labels at readable size,
then rehearse using the guide. The build does not claim a rehearsal, human review,
PDF/UA conformance, fresh DVC retrieval, selected-model inference, release
publication or target deployment. Generated PDFs, bundles and QA stay local and
ignored; the user controls any later distribution or submission.
