# Contributing to Predictive Analytics for Social Cause

Thank you for your interest in contributing to this project! This guide will help you get started.

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. Please read and follow our community guidelines.

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs or suggest features
- Provide detailed information including steps to reproduce
- Include relevant system information and error messages

### Development Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write unit tests for new functionality

### Testing
- Run the full test suite before submitting PRs
- Add tests for any new features or bug fixes
- Ensure code coverage remains above 80%

### Documentation
- Update README.md if needed
- Add docstrings to new functions
- Update methodology.md for algorithmic changes
- Include examples in code comments

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/your-username/predictive-social-cause.git
   cd predictive-social-cause
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. Run tests to ensure everything works:
   ```bash
   pytest
   ```

## Pull Request Guidelines

- Keep PRs focused and atomic
- Write clear commit messages
- Include tests for new features
- Update documentation as needed
- Ensure CI/CD pipeline passes

## Questions or Need Help?

- Open an issue for questions
- Join our community discussions
- Check existing documentation first

Thank you for contributing to educational equity through data science!
