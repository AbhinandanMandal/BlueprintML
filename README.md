# BlueprintML

**From research papers to reproducible implementations.**

[BlueprintML](https://github.com/AbhinandanMandal/BlueprintML) is an open-source project by **[Advanced Vision Labs](https://detail-primer-91884953.figma.site/)**
dedicated to implementing foundational and state-of-the-art machine
learning papers from scratch. The goal is to make research easier to
understand, reproduce, inspect, and extend through clean code, practical
experiments, and shared learning.

> **Understand. Implement. Experiment. Collaborate.**

[![GitHub](https://img.shields.io/badge/GitHub-BlueprintML-181717?logo=github)](https://github.com/AbhinandanMandal/BlueprintML)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-2ea44f.svg)](CONTRIBUTING.md)
[![Discord](https://img.shields.io/badge/Discord-Advanced%20Vision%20Labs-5865F2?logo=discord&logoColor=white)](https://discord.gg/45aMaqVDk)
[![Maintained by Abhinandan Mandal](https://img.shields.io/badge/GitHub-Abhinandan%20Mandal-181717?logo=github)](https://github.com/AbhinandanMandal)

## About the project

Research papers often communicate powerful ideas, but reproducing them
requires more than reading the abstract or calling a library API.
[BlueprintML](https://github.com/AbhinandanMandal/BlueprintML) focuses on implementing methods in a way that helps
contributors understand the motivation, architecture, objective
functions, training procedure, and experimental choices behind each
paper.

Implementations are designed for ``Google Colab`` where practical, so
students and researchers can explore them without needing to configure a
large local environment.

## Implementations
Please visit [PAPERS.md](https://github.com/AbhinandanMandal/BlueprintML/blob/master/PAPERS.md) for implemented papers list. Also **please make sure to update [PAPERS.md](https://github.com/AbhinandanMandal/BlueprintML/blob/master/PAPERS.md) upon each new implementation according to each domain and field**.

## Project principles

-   **Learn from first principles:** explain the key ideas instead of
    hiding them behind high-level APIs.
-   **Readable implementations:** keep code modular, documented, and
    approachable.
-   **Reproducibility:** record important configurations, datasets,
    seeds, and evaluation details where applicable.
-   **Honest reporting:** distinguish reproduced results from reference
    results and clearly describe deviations.
-   **Collaborative learning:** welcome questions, constructive reviews,
    and contributions from different experience levels.

## Who is this for?

[BlueprintML](https://github.com/AbhinandanMandal/BlueprintML) is for students learning deep learning, researchers studying
or reproducing papers, engineers who want to understand methods under
the hood, and anyone interested in open-source AI research.

You do not need to be an expert to contribute. Documentation
improvements, bug reports, tests, experiment verification, and clear
explanations are valuable contributions too.

## Getting started

1.  **Explore the repository:** browse the method folders and read the
    accompanying paper or notes.
2.  **Open the notebook:** use the Google Colab link provided by the
    relevant implementation, when available.
3.  **Run a small test first:** validate the environment and data
    pipeline before starting a full experiment.
4.  **Compare carefully:** check implementation details and metrics
    against the paper, and document any differences.
5.  **Share what you learn:** report issues, suggest improvements, or
    contribute a pull request.

Because each paper may have different dependencies and datasets, follow
the setup instructions inside its own directory or notebook.

## Contributing

We welcome contributions such as:

-   Implementing a paper or completing a missing component.
-   Improving correctness, readability, tests, and documentation.
-   Reproducing or analysing reported results.
-   Adding small-scale smoke tests suitable for Google Colab.
-   Fixing bugs, clarifying mathematical details, and improving
    experiment reports.

Before opening a pull request, please read
**[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Suggested structure for a new implementation

When practical, include:

``` text
MethodName/
├── README.md              # Paper summary, setup, usage, and results
├── notebook.ipynb         # Google Colab-ready implementation
├── src/                   # Reusable implementation modules (optional)
├── configs/               # Experiment configurations (optional)
├── reports/               # Results, plots, and analysis (optional)
└── requirements.txt       # Method-specific dependencies (if needed)
```

Keep the structure proportionate to the project. A small, self-contained
notebook is fine when it improves accessibility.

## Community and collaboration

[BlueprintML](https://github.com/AbhinandanMandal/BlueprintML) is intended to be more than a collection of notebooks: it is
a place to learn through implementation, exchange ideas, and build
research skills together. Contributors are encouraged to ask questions,
explain trade-offs, share reproducibility notes, and help others get
started.

If you would like to contribute, start with an issue or a small pull
request. For substantial additions, opening an issue first helps align
on scope and avoids duplicated effort.

## Acknowledgements

[BlueprintML](https://github.com/AbhinandanMandal/BlueprintML) builds on the work of researchers who publish papers, code,
datasets, and educational resources. Please cite original papers
appropriately and respect the licences and usage terms of any
third-party code or data used in an implementation.

## Maintained by
[![Maintained by Abhinandan Mandal](https://img.shields.io/badge/GitHub-Abhinandan%20Mandal-181717?logo=github)](https://github.com/AbhinandanMandal)




