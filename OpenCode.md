OpenCode Guidelines

Setup: pip install -r requirements.txt && export OPENAI_API_KEY=YOUR_KEY
Run: uvicorn main:app --reload

Lint & Format: black . && isort . && flake8 .

Test: pytest
Single test: pytest test_search.py::TestSearch::test_search

Style Guidelines:
• Imports: stdlib, 3rd-party, local
• Formatting: Black (88 cols), isort
• Naming: snake_case funcs/vars, CamelCase classes
• Typing: PEP484 type hints on public APIs
• Errors: raise HTTPException in controllers
• Logging: Python logging module
• Documentation: update README.md

No Cursor or Copilot rules detected.