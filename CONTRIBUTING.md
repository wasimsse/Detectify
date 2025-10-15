# Contributing to Detectify

Thank you for your interest in contributing to Detectify! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/wasimsse/Detectify/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, GPU info)
   - Screenshots if applicable

### Suggesting Features

1. Check existing [Issues](https://github.com/wasimsse/Detectify/issues) for similar suggestions
2. Create a new issue with:
   - Clear feature description
   - Use case and benefits
   - Possible implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit with clear messages
7. Push to your fork
8. Submit a Pull Request

## 📋 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Detectify.git
cd Detectify

# Add upstream remote
git remote add upstream https://github.com/wasimsse/Detectify.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies including dev tools
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Create .env file
cp .env.example .env
```

## 🎨 Code Style

- Follow PEP 8 style guide
- Use type hints
- Write docstrings for functions and classes
- Keep functions focused and modular
- Maximum line length: 100 characters

### Formatting

```bash
# Format code with black
black src/

# Check with flake8
flake8 src/

# Type checking with mypy
mypy src/
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📝 Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example: `feat: add support for YOLOv11 models`

## 🔍 Code Review Process

1. All PRs require at least one review
2. Address review comments
3. Keep PRs focused and reasonably sized
4. Update PR description if scope changes

## 📚 Documentation

- Update README.md for new features
- Add docstrings to new functions/classes
- Update inline comments as needed
- Create/update docs in `docs/` folder

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙋 Questions?

Feel free to ask questions by opening an issue or discussion!

Thank you for contributing! 🎉

