# Contributing to BlueprintML

Thank you for your interest in contributing to **[BlueprintML](https://github.com/AbhinandanMandal/BlueprintML) by [Advanced
Vision Labs](https://detail-primer-91884953.figma.site/)**!

[BlueprintML](https://github.com/AbhinandanMandal/BlueprintML) is an open-source effort to implement foundational and
state-of-the-art machine learning papers from scratch, with an emphasis
on understanding, clear engineering, reproducibility, and collaborative
learning. Contributions from students, researchers, engineers, and
curious beginners are welcome.

You do not need to be an expert. A well-explained bug report, a
documentation improvement, a tested fix, or a careful experiment can be
just as valuable as a new implementation.

## Ways to contribute

You can help by:

-   Implementing a paper or completing a missing component.
-   Reviewing code for correctness, clarity, and consistency with the
    paper.
-   Reproducing results and documenting differences from the reference.
-   Improving notebooks so they run reliably in Google Colab.
-   Adding tests, debugging edge cases, or improving performance.
-   Improving READMEs, mathematical explanations, diagrams, and setup
    instructions.
-   Reporting bugs, suggesting papers, or helping review pull requests.

## Before you start

1.  Browse the repository and check whether the method or issue already
    exists.
2.  Read the relevant paper and any implementation-specific README.
3.  Search existing issues and pull requests to avoid duplicating work.
4.  For a large implementation or a significant design change, open an
    issue first to discuss scope and approach.

For small documentation fixes or clearly scoped bugs, you can usually
submit a pull request directly.

## Development workflow

1.  **Fork** the repository to your GitHub account.
2.  **Clone** your fork and create a focused branch.
3.  Make a small, logically grouped change.
4.  Run the relevant checks and a small smoke test.
5.  Update documentation and report any limitations.
6.  Open a pull request against the repository's default branch and
    describe what changed.

Example commands:

``` bash
git clone https://github.com/<your-username>/BlueprintML.git
cd BlueprintML
git checkout -b feat/short-description
```

After making your changes:

``` bash
git add .
git commit -m "Add: short description of change"
git push origin feat/short-description
```

Then open a pull request on GitHub. Replace `<your-username>` with your
GitHub username.

## Standards for paper implementations

BlueprintML aims to make implementations useful for learning and
research. Please follow these guidelines wherever they apply.

### 1. Stay faithful to the paper

-   Identify the paper and link to the original publication or preprint.
-   Explain the method's central idea and key components.
-   Implement the paper's essential method rather than relying on a
    ready-made implementation of the complete method.
-   It is fine to use standard frameworks and primitives (for example,
    PyTorch tensor operations, optimizers, and common layers), but
    explain important design choices.
-   Clearly disclose deviations, simplifications, or details that could
    not be verified.

### 2. Write readable, maintainable code

-   Prefer clear names and small, understandable functions.
-   Separate data preparation, model definition, loss functions,
    training, and evaluation when this makes the implementation easier
    to follow.
-   Add comments for non-obvious decisions, not for every line.
-   Avoid unnecessary abstractions that make educational code harder to
    understand.
-   Keep dependencies minimal and document any method-specific
    requirements.

### 3. Make notebooks accessible

When using Google Colab:

-   Include a brief introduction and links to the paper.
-   Put installation and imports near the beginning.
-   Explain dataset access and any required credentials without exposing
    secrets.
-   Include a small smoke-test path so contributors can verify the code
    without running a full training job.
-   Keep cells organised in execution order and avoid relying on hidden
    notebook state.
-   Use a fixed random seed when appropriate, while noting that exact
    determinism may not be guaranteed across hardware and library
    versions.
-   Explain expected runtime and compute requirements when known.

### 4. Report experiments honestly

Where results are reported, include as much of the following as is
practical:

-   Dataset and split.
-   Model configuration and important hyperparameters.
-   Optimizer, learning rate, batch size, epochs, and relevant training
    details.
-   Random seed and hardware/software environment.
-   Evaluation protocol and metric definitions.
-   Reference result from the paper, your result, and the difference.
-   Known limitations, failed runs, or unresolved discrepancies.

Do not describe a result as a reproduction of the paper unless the setup
and evaluation support that claim. Label preliminary, partial, or
small-subset experiments clearly.

### 5. Respect data and licences

-   Follow the licence and terms for datasets, pretrained weights, and
    third-party code.
-   Do not commit private datasets, credentials, access tokens, or large
    generated files.
-   Credit original authors and include citations.
-   If you adapt existing code, state the source and verify that its
    licence permits the intended use.

## Suggested structure

Use a structure that suits the size of the implementation. For a larger
addition, this layout is a useful starting point:

``` text
MethodName/
├── README.md
├── notebook.ipynb
├── src/                   # Optional reusable modules
├── configs/               # Optional experiment settings
├── reports/               # Optional results and analysis
└── requirements.txt       # If method-specific dependencies are needed
```

The method-level `README.md` should ideally include:

-   Paper title, authors, publication year, and link.
-   A concise explanation of the method.
-   What is implemented and what is not.
-   Setup and run instructions.
-   Dataset requirements and expected compute.
-   Results, evaluation protocol, and known limitations.
-   References and acknowledgements.

## Tracks
Please make sure to update [PAPERS.md](https://github.com/AbhinandanMandal/BlueprintML/blob/master/PAPERS.md) upon each new implementation according to each domain and field.

## Testing checklist

Before submitting, check the items relevant to your contribution:

-   [ ] The notebook or code runs from a clean session, as far as
    practical.
-   [ ] Imports and dependency instructions are complete.
-   [ ] A small smoke test completes successfully.
-   [ ] Tensor shapes, loss values, and outputs are checked where
    appropriate.
-   [ ] Evaluation follows the stated protocol.
-   [ ] Results are labelled accurately and can be interpreted.
-   [ ] Documentation reflects the current implementation.
-   [ ] No credentials, private data, or unnecessary large files are
    included.
-   [ ] Sources and third-party licences are acknowledged.

If you cannot run a full experiment because of compute or data
constraints, say so clearly in the pull request and describe the checks
you were able to perform.

## Issues and pull requests

### Bug reports

Please include:

-   The affected method or notebook.
-   What you expected to happen and what happened instead.
-   Steps to reproduce the issue.
-   Relevant error messages and environment details.
-   A minimal example or screenshot, if helpful.

Remove access tokens, credentials, and private data before sharing logs.

### Feature requests and paper suggestions

Describe the proposed change, why it would benefit the project, and any
relevant paper or reference. For a new paper implementation, briefly
outline the key method and any known compute or dataset requirements.

### Pull requests

A useful pull request should:

-   Have a clear, descriptive title.
-   Explain the motivation and main changes.
-   Link to the related issue, if one exists.
-   Describe testing performed and include results where relevant.
-   Call out deviations from the paper or limitations.
-   Stay focused; separate unrelated changes when practical.

Maintainers may ask for clarification, changes, or additional checks
before merging. Reviews are intended to improve the implementation and
help everyone learn.

## Commit message suggestions

Use short, descriptive messages. These prefixes are suggestions, not
strict requirements:

-   `Add:` new implementation or feature
-   `Fix:` bug fix
-   `Docs:` documentation change
-   `Test:` test or validation addition
-   `Refactor:` code restructuring without intended behaviour changes

Examples:

``` text
Add: implement MoCo training loop
Fix: correct contrastive loss temperature
Docs: clarify Colab setup instructions
Test: add encoder shape smoke test
```

## Community expectations

Please help maintain a constructive, welcoming, and research-focused
environment.

-   Be respectful and specific in technical feedback.
-   Explain disagreements with evidence or reasoning.
-   Welcome beginner questions and different approaches.
-   Give credit to original authors and fellow contributors.
-   Be transparent about results, limitations, and borrowed code.
-   Focus discussion on improving the work, not on personal criticism.

Harassment, discrimination, deliberate misrepresentation, and sharing
private information about others are not acceptable.

## Questions?

Open an issue in the [BlueprintML
repository](https://github.com/AbhinandanMandal/BlueprintML) with a
clear title and enough context for others to help.

Thank you for helping make machine learning research more
understandable, reproducible, and collaborative.

## Maintained by
[![Maintained by Abhinandan Mandal](https://img.shields.io/badge/GitHub-Abhinandan%20Mandal-181717?logo=github)](https://github.com/AbhinandanMandal)
