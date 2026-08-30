# Contributing

Thank you for improving YOLO-World Annotator. Contributions should be small enough to review, behavior-focused, and accompanied by tests when they change Python behavior.

## Setup

1. Fork and clone the repository.
2. Create a branch from the default branch.
3. Create a Python 3.10–3.12 virtual environment.
4. Install development dependencies:

   ```bash
   python -m pip install -e ".[dev,build]"
   pre-commit install
   ```

## Development rules

- Keep production imports under the `yolo_world_annotator` namespace.
- Do not commit model weights, images, generated labels, logs, or binary builds.
- Preserve the in-place YOLO label format unless a proposal explicitly changes it.
- Keep model downloads out of unit tests. Use a small fake at the Ultralytics boundary.
- Add a failing regression test before fixing a bug.
- Use `auto` as the default device; explicit CUDA requests must not silently fall back.

## Before opening a pull request

```bash
python -m pytest -q
python -m ruff check .
python -m build
git diff --check
```

Describe the user-visible change, the tests you ran, and any CPU/CUDA hardware used. Screenshots are useful for GUI changes. Pull requests that alter dependencies or model handling must discuss licensing and download size.

By contributing, you agree that your contribution is licensed under AGPL-3.0-or-later and that you have the right to submit it.
