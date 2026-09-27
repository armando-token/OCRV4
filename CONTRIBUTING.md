# Contributing to OCRV4

Thank you for your interest in improving OCRV4! We welcome community contributions, bug reports, and enhancements.

---

## Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for all contributors. Please be respectful and constructive in all discussions and code reviews.

---

## How to Report Bugs

Before opening a new issue:
1. **Search existing issues** to see if the bug has already been reported.
2. If new, open an issue with:
   - A clear, descriptive title.
   - Operating system and Python version (`python --version`).
   - Installed Poppler version (`pdftoppm -v`).
   - Minimal reproduction steps, including sample input characteristics (avoid uploading proprietary or sensitive PDFs).
   - Full stack trace or error log output.

---

## Development Workflow

1. **Fork the Repository** to your GitHub account.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/ocrv4.git
   cd ocrv4
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install ruff flake8
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Implement your changes**:
   - Write clean, well-documented code.
   - Preserve backward compatibility wherever practical.
   - Verify syntax with `python -m py_compile pdf_to_markdown_seq.py`.
   - Run linters: `ruff check .` and `flake8 .`.
6. **Commit with conventional messages**:
   - `feat: add retry logic for rate limits`
   - `fix: handle edge case in single-page PDFs`
   - `docs: update CLI options table`
7. **Push and open a Pull Request**:
   - Provide a concise summary of the changes and motivation.
   - Reference any related issues (e.g., `Fixes #12`).

---

## Coding Standards

- **Python Version**: Target Python 3.10+.
- **Style Guide**: Conform to PEP 8 standards. Linting is enforced via Ruff and Flake8 in CI.
- **Typing**: Use standard Python type hints (`typing` module / built-in generics) for function signatures.
- **Error Handling**: Gracefully handle network timeouts and external Poppler binary errors.
- **Zero Artifact Leaks**: Ensure temporary rasters and cache files are cleaned up in `finally` blocks.

---

## Pull Request Checklist

- [ ] Code compiles without errors (`python -m py_compile`).
- [ ] Linters pass cleanly (`ruff check .`).
- [ ] Documentation / README updated if CLI flags or requirements changed.
- [ ] Commit history is clean and descriptive.
