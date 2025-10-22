# Repository Guidelines

## Project Structure & Module Organization
- `autoencoder/`: Core models (`models/`), training (`training/`), evaluation (`evaluation/`), and utils.
- `gui_managers/`: GUI logic split into managers (training, evaluation, visualization, reconstruction).
- `networks/`: Example/reference network definitions.
- `models/`: Pretrained weights and configs (large `.pth` files). Use Git LFS for new large files.
- `tests/` and root `test_*.py`: Script-style tests and demos.
- Top-level scripts: `main.py`, `training.py`, `gui.py`; configuration in `config.json`.
- Generated artifacts: `logs/`, `results/`, `checkpoints/`, `cache/`.

## Build, Test, and Development Commands
- Create env and install: `python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt`
- Run GUI: `python gui.py`
- Train: `python training.py --config config.json`
- Model interface check: `python test_model_interfaces.py`
- Enhanced CNN demo: `python test_enhanced_cnn.py`

## Coding Style & Naming Conventions
- Python 3.13; follow PEP 8. Indent 4 spaces; target line length ≤ 100.
- Type hints required for public APIs; add docstrings (Google style preferred).
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Imports: prefer absolute within package (e.g., `from autoencoder.models import ...`). Avoid circular deps.
- Logging: prefer `logging` to `print`; write long-running outputs to `logs/`.

## Testing Guidelines
- Place tests as `test_*.py` under root or `tests/`. Keep fast and deterministic (set seeds).
- CPU by default; skip GPU-heavy paths unless explicitly needed.
- No formal coverage gate yet; add assertions when converting demos to tests.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:` (see Git history).
- Commits: imperative, concise subject (≤72 chars); descriptive body when needed.
- PRs must include: purpose/summary, linked issues, reproduction steps, expected results, and screenshots for GUI changes. Confirm local run of key scripts and tests above.

## Security & Configuration Tips
- Do not commit datasets or secrets. Use Git LFS for files >50MB (e.g., new `.pth`).
- Keep paths/config in `config.json`; avoid hardcoding. Document new keys in `README.md`.

## Agent-Specific Instructions
- Keep changes focused and incremental. Update docs when altering CLI, config keys, or public APIs. Maintain compatibility across `autoencoder/*` modules.

