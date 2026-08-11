# Project instructions

## Python environment

- This project uses the Conda environment `pkb-agent` for Python development and evaluation.
- Run Python, `pkb-agent`, tests, linters, and project scripts with `conda run -n pkb-agent <command>` in non-interactive shells.
- Do not default to the repository `.venv`; it may be stale or lack project dependencies.
- Before diagnosing an import or dependency problem, reproduce it inside the `pkb-agent` Conda environment.

Examples:

```sh
conda run -n pkb-agent pkb-agent eval validate data/evals/tech-multidomain-v1
conda run -n pkb-agent python -m pytest -q
```
