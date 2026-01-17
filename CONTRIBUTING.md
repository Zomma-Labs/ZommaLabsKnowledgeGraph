# Contributing to ZommaLabsKG

Thank you for your interest in contributing to ZommaLabsKG! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Neo4j database (local or Aura)
- API keys for OpenAI and Google (see `.env.example`)

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Zomma-Labs/ZommaLabsKnowledgeGraph.git
   cd ZommaLabsKnowledgeGraph
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Set up environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Initialize the database:
   ```bash
   uv run src/scripts/setup_graph_index.py
   ```

5. Run tests:
   ```bash
   uv run pytest tests/
   ```

## How to Contribute

### Reporting Bugs

- Check existing [issues](https://github.com/Zomma-Labs/ZommaLabsKnowledgeGraph/issues) first
- Use the bug report template when creating a new issue
- Include reproduction steps, expected vs actual behavior
- Include relevant logs and environment details

### Suggesting Features

- Open an issue with the feature request template
- Describe the use case and proposed solution
- Be open to discussion about alternatives

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style guidelines below
4. **Write/update tests** for your changes
5. **Run tests** to ensure they pass:
   ```bash
   uv run pytest tests/
   ```
6. **Commit** with a clear message:
   ```bash
   git commit -m "Add: brief description of change"
   ```
7. **Push** and create a Pull Request

### Commit Message Format

Use clear, descriptive commit messages:
- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for changes to existing features
- `Remove:` for removed features
- `Docs:` for documentation changes
- `Refactor:` for code refactoring

Example: `Add: entity deduplication with vector similarity`

## Code Style

### Python

- Follow PEP 8 conventions
- Use type hints for function signatures
- Use docstrings for public functions and classes
- Keep functions focused and reasonably sized

### Project Structure

```
src/
├── agents/          # LLM-based extraction agents
├── chunker/         # Document chunking logic
├── config/          # Configuration files
├── schemas/         # Pydantic models and schemas
├── util/            # Shared utilities
├── scripts/         # Setup and maintenance scripts
├── querying_system/ # Query pipelines
│   ├── v6/          # Production pipeline
│   └── deep_research/
└── pipeline.py      # Main ingestion pipeline
```

### Key Conventions

- Use `async/await` for I/O operations
- Use Pydantic models for structured data
- Use the services singleton (`src/util/services.py`) for shared clients
- Add new LLM prompts to appropriate prompt files

## Testing

- Write tests for new functionality in `tests/`
- Use pytest with async support (`pytest-asyncio`)
- Mock external API calls where appropriate
- Test both success and failure cases

Run tests:
```bash
uv run pytest tests/ -v
```

## Documentation

- Update `CLAUDE.md` for architectural changes
- Update `README.md` for user-facing changes
- Add docstrings to new public APIs
- Document environment variables in `.env.example`

## Questions?

Feel free to open an issue for questions or join discussions on existing issues.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
