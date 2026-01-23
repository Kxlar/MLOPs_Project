# Style & Linting
## Tools
- **Ruff:** Fast Python linter and formatter.
- **pre-commit:** Runs checks automatically at commit/push.
- **Pytest + Coverage:** Executed by a pre-commit hook to keep tests green.

Configuration lives in `pyproject.toml` and `.pre-commit-config.yaml`.


### Format
- Check formatting (no changes):
```powershell
uv run ruff format --check .e
```
- Auto-format (apply changes):
```powershell
uv run ruff format .
```

### Lint
- Lint entire repo:
```powershell
uv run ruff check .
```
- Apply autofixes for fixable lint rules:
```powershell
uv run ruff check . --fix
```

## pre-commit Hooks (on commit/push)
Hooks are defined in [.pre-commit-config.yaml](../../.pre-commit-config.yaml). They include:
- General hygiene: trailing whitespace, end-of-file, YAML check, large files
- Ruff lint with `--fix` and `ruff-format`
- Unit tests via `pytest`

Install hooks so they run automatically on `git commit`:
```powershell
uv run pre-commit install
```