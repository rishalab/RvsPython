# ACM SIGSOFT Empirical Standards: Clause-by-Clause Mapping

**Paper:** Measuring Energy Consumption of Machine Learning Ecosystems in R and Python: A Comparative Empirical Study
**Submission:** INFSOF-D-26-00963, *Information and Software Technology*
**Standards applied:** ACM SIGSOFT Empirical Standards — General Standard and Engineering Research Standard (Ralph, P. et al., 2020, *ACM SIGSOFT Empirical Standards*, arXiv:2010.03525), as cited in Section 6.1 of the paper.

This study is a controlled, non-human-subjects comparative measurement study. It is assessed below against the General Standard, which applies to all empirical software-engineering research, and against the attributes of the Engineering Research Standard relevant to a comparative evaluation of two existing ecosystems on a common workload. Each row states the attribute, the paper's compliance status, and the specific section, table, or footnote where it is addressed.

Status key: **Met** — fully addressed; **Partial** — addressed with an acknowledged gap; **Not met** — not attempted, with reasoning given in the manuscript.

---

## Part A — General Standard

| # | Attribute | Status | Where addressed |
|---|---|---|---|
| A1 | The study is motivated by a clear problem statement and gap in prior work | Met | Section 1 (Introduction): prior work (Marini et al., 2025) is limited to small classification-only datasets; the regression dimension and larger deployment scales are identified as unexamined |
| A2 | Explicit, answerable research questions are stated | Met | Section 3 (Research Questions): RQ1/RQ2, each split into training/inference sub-questions, with the rationale for the 2×2 (task type × phase) structure given directly before the RQs |
| A3 | The study design (variables, procedure) is described in enough detail to be understood independent of the code | Met | Section 6.1 (Experimental Design): independent variables (ML task, dataset size), dependent variables (energy, runtime), sub-experiment structure |
| A4 | Subject/dataset selection is described and justified | Met | Section 5 (Datasets): three datasets spanning 49K–1.4M rows, chosen to span realistic deployment scales; Section 4 (ML Tasks) and Table 2: ten algorithms mapped to their standard-library counterpart in each ecosystem |
| A5 | The experimental procedure is controlled and repeatable | Met | Sections 6.2–6.3 (Experimental Setup, Procedure): hardware/OS/kernel specified, warm-up and idle periods, randomised shuffling, effective-energy formula, ten repetitions per configuration |
| A6 | Data collection and measurement instruments are described, including known limitations | Met | Section 6.2: pyJoules/RJoules on Intel RAPL, Package/DRAM domains; Section 8.1 (Internal Validity): RAPL 32-bit counter wraparound disclosed, affected cells enumerated, direction of resulting bias argued |
| A7 | Appropriate statistical analysis is used and reported per comparison, not only in aggregate | Met | Section 7 (Results): Wilcoxon signed-rank test at α = 0.05 reported per cell in Tables 4 and 5, not collapsed to an aggregate count |
| A8 | Effect sizes are reported alongside significance tests | **Not met** | Disclosed directly in the manuscript: Section 6.3 states that effect-size estimation was not computed and that percentage energy differences are reported as a descriptive indication of magnitude instead, with standardised effect-size reporting identified as a direction for future replications; Section 6.1 states this as the one essential General Standard expectation not met |
| A9 | Results are reported even where negative, mixed, or contrary to the study's own prior findings | Met | Sections 7.1D, 7.3 and 9.2 report non-significant cells explicitly (e.g., the n=4 Wilcoxon-floor case, Logistic Regression Dataset1 inference at p=0.1934) and document where an earlier, smaller-scale analysis of the same data reversed direction under the corrected, full-protocol measurement |
| A10 | Threats to validity are discussed by category | Met | Section 8 (Threats to Validity): Internal (8.1), External (8.2), Construct (8.3), Conclusion (8.4), each with concrete, study-specific instances |
| A11 | A replication package is publicly released with sufficient material to reproduce the study | Met | Sections 6.3 and 11 give the package location, `https://github.com/rishalab/RvsPython`, containing the R/Python/scratch scripts, per-trial raw energy CSVs, the SD/CV summary, the log-scale supplementary figures S2–S5, this clause-by-clause mapping, and the full predictive-performance file (precision, recall, accuracy, F1, RMSE, MAE per configuration) underlying Table 14 |
| A12 | Claims of novelty or contribution are not overstated relative to what the results show | Met | Contributions list and Section 9.4's opening state findings at the level of granularity the data support (per-algorithm scaling behaviour rather than a single cross-ecosystem pattern); Section 9.1's runtime claim is qualified consistently with Section 7.1's finding that runtime and energy track "not in exact lockstep" |

## Part B — Engineering Research Standard (attributes applicable to a comparative measurement study)

| # | Attribute | Status | Where addressed |
|---|---|---|---|
| B1 | The artifacts/systems under evaluation are named precisely, including versions | Met | Table 2 (library-version table): scikit-learn and CRAN package versions per algorithm, including `stats` (base R) for Gaussian Regression, with a footnote explaining the fit function used |
| B2 | Evaluation tasks/data are realistic and at a scale relevant to practice | Met | Section 5: dataset scale spans 49K to 1.4M rows, chosen specifically to cover realistic deployment sizes underexamined by prior small-scale work (Section 1) |
| B3 | Where the two systems compared are not doing identical work, this is disclosed rather than presented as a clean like-for-like result | Met | Section 8.3 (Construct Validity) and Table 13 (comparability audit): a three-tier classification (Comparable / Default-divergent / Not-equivalent) is given for all ten algorithm pairings, including the Gaussian Regression algorithm-class mismatch and the Neural Network Regression seeding/convergence issue |
| B4 | Generalizability of findings beyond the specific evaluation environment is discussed | Met | Section 8.2 (External Validity): single-hardware/single-OS scope stated, RAPL's Intel-only availability noted, findings framed as platform-specific pending replication |
| B5 | Known defects or corrections in the evaluation apparatus are disclosed, with their effect on the reported results argued rather than assumed | Met | Section 9.2: the Decision Tree classification measurement-script defect (a support vector classifier was instantiated instead of a decision tree classifier in an earlier analysis) is disclosed by name; Section 7.4 uses the Table 14 predictive-performance values as an independent check that the corrected script now produces algorithm-appropriate output |

---

## Summary

11 of 12 General Standard attributes and all 5 applicable Engineering Research Standard attributes are met. The one attribute not met — standardised effect-size reporting — is disclosed in the manuscript itself (Sections 6.1, 6.3) rather than surfaced externally; the paper reports descriptive percentage differences in its place and identifies standardised effect sizes as a target for future replication work.

---

*Attribute categories follow the General Standard and Engineering Research Standard as defined in Ralph et al. (2020), ACM SIGSOFT Empirical Standards.*
