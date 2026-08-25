# Contributing to PAIMANA Intelligence Platform

Thank you for your interest in contributing to infrastructure transparency in India!

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/[username]/paimana-intelligence-platform/issues)
2. If not, open a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs. actual behavior
   - System information (OS, Python version)

### Suggesting Features

1. Open an issue with the "enhancement" label
2. Describe the feature and its use case
3. Explain why it would be valuable for infrastructure transparency

### Code Contributions

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes following our coding standards:
   - Follow PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Include unit tests for new features
   - Update documentation as needed
4. Test your changes: `python -m pytest tests/`
5. Commit with clear messages: `git commit -m "Add feature: description"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Open a Pull Request with:
   - Description of changes
   - Reference to related issues
   - Screenshots if UI changes

### Code Standards

- Use type hints where applicable
- Write self-documenting code with meaningful variable names
- Add error handling and logging
- Keep functions focused (Single Responsibility Principle)
- Avoid code duplication (DRY principle)

### Data Contributions

- If adding new data sources, document:
  - Source URL and access method
  - Data licensing and terms of use
  - Schema and field descriptions
  - Update frequency

### Testing

- All new features must include tests
- Maintain >80% code coverage
- Test edge cases and error conditions

## Development Setupnpm

```bash
# Clone your fork
git clone https://github.com/yourusername/paimana-intelligence-platform.git
cd paimana-intelligence-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## Community Guidelines

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers
- Prioritize accessibility and user impact

## Questions?

Open a [Discussion](https://github.com/[username]/paimana-intelligence-platform/discussions) or reach out to maintainers.

Thank you for helping make infrastructure data accessible to all!
