"""Project scaffolding and code templates for LEGIONHERCULES."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from datetime import datetime

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TemplateVariable:
    """A variable in a template."""
    name: str
    description: str
    default: Any = None
    required: bool = True


@dataclass
class CodeTemplate:
    """A code template."""
    name: str
    description: str
    category: str
    file_pattern: str
    content: str
    variables: List[TemplateVariable] = field(default_factory=list)


class TemplateLibrary:
    """Library of code templates."""

    PYTHON_CLASS = CodeTemplate(
        name="python_class",
        description="Python class with docstring",
        category="python",
        file_pattern="{name}.py",
        content='''"""{description}"""

from __future__ import annotations

from typing import Any, Optional
from dataclasses import dataclass, field

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class {class_name}:
    """{description}"""
    
    def __init__(self{init_params}):
        """Initialize {class_name}."""
{init_assignments}
    
    def __repr__(self) -> str:
        return f"{class_name}({repr_params})"
''',
        variables=[
            TemplateVariable("class_name", "Name of the class"),
            TemplateVariable("description", "Class description"),
            TemplateVariable("init_params", "__init__ parameters", ""),
            TemplateVariable("init_assignments", "Attribute assignments", "        pass"),
            TemplateVariable("repr_params", "Parameters for __repr__", ""),
        ],
    )

    PYTHON_DATACLASS = CodeTemplate(
        name="python_dataclass",
        description="Python dataclass",
        category="python",
        file_pattern="{name}.py",
        content='''"""{description}"""

from __future__ import annotations

from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class {class_name}:
    """{description}"""
    
{fields}
    
    def to_dict(self) -> Dict[str, Any]:
        return {{
{to_dict_fields}
        }}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> {class_name}:
        return cls(
{from_dict_fields}
        )
''',
        variables=[
            TemplateVariable("class_name", "Name of the dataclass"),
            TemplateVariable("description", "Class description"),
            TemplateVariable("fields", "Dataclass fields"),
            TemplateVariable("to_dict_fields", "Fields for to_dict method"),
            TemplateVariable("from_dict_fields", "Fields for from_dict method"),
        ],
    )

    PYTHON_TEST = CodeTemplate(
        name="python_test",
        description="Python test file with pytest",
        category="python",
        file_pattern="test_{name}.py",
        content='''"""Tests for {module_name}."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from legionhercules.{module_path} import {class_name}


class Test{class_name}:
    """Test cases for {class_name}."""
    
    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return {class_name}()
    
    def test_initialization(self, instance):
        """Test instance initialization."""
        assert instance is not None
    
    def test_basic_functionality(self, instance):
        """Test basic functionality."""
        # TODO: Add test implementation
        pass
''',
        variables=[
            TemplateVariable("module_name", "Name of module being tested"),
            TemplateVariable("module_path", "Import path to module"),
            TemplateVariable("class_name", "Name of class being tested"),
        ],
    )

    FASTAPI_ENDPOINT = CodeTemplate(
        name="fastapi_endpoint",
        description="FastAPI endpoint with CRUD operations",
        category="web",
        file_pattern="{name}.py",
        content='''"""{resource_name} API endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional

router = APIRouter(prefix="/{resource_path}", tags=["{resource_name}"])


@router.get("/")
async def list_{resource_path}(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """List all {resource_name_plural}."""
    return {{"items": [], "total": 0}}


@router.get("/{{item_id}}")
async def get_{resource_path}(item_id: str):
    """Get a specific {resource_name} by ID."""
    return {{"id": item_id}}


@router.post("/")
async def create_{resource_path}(data: dict):
    """Create a new {resource_name}."""
    return {{"id": "new-id", **data}}


@router.put("/{{item_id}}")
async def update_{resource_path}(item_id: str, data: dict):
    """Update a {resource_name}."""
    return {{"id": item_id, **data}}


@router.delete("/{{item_id}}")
async def delete_{resource_path}(item_id: str):
    """Delete a {resource_name}."""
    return {{"deleted": True, "id": item_id}}
''',
        variables=[
            TemplateVariable("resource_name", "Name of the resource (singular)"),
            TemplateVariable("resource_name_plural", "Plural name of the resource"),
            TemplateVariable("resource_path", "URL path for the resource"),
        ],
    )

    REACT_COMPONENT = CodeTemplate(
        name="react_component",
        description="React functional component with hooks",
        category="web",
        file_pattern="{name}.jsx",
        content='''import React, {{ useState, useEffect }} from 'react';
import PropTypes from 'prop-types';

/**
 * {component_name} component
 * {description}
 */
const {component_name} = ({{ {props} }}) => {{
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);

  useEffect(() => {{
    // Component mount logic
  }}, []);

  return (
    <div className="{class_name}">
      <h2>{component_name}</h2>
      {{/* Component content */}}
    </div>
  );
}};

{component_name}.propTypes = {{
{prop_types}
}};

{component_name}.defaultProps = {{
{default_props}
}};

export default {component_name};
''',
        variables=[
            TemplateVariable("component_name", "Name of the React component"),
            TemplateVariable("description", "Component description"),
            TemplateVariable("props", "Component props"),
            TemplateVariable("class_name", "CSS class name"),
            TemplateVariable("prop_types", "PropTypes definitions"),
            TemplateVariable("default_props", "Default prop values"),
        ],
    )

    DOCKERFILE = CodeTemplate(
        name="dockerfile",
        description="Multi-stage Dockerfile for Python",
        category="devops",
        file_pattern="Dockerfile",
        content='''# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Make sure scripts are executable
ENV PATH=/root/.local/bin:$PATH

# Run as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "-m", "{module_name}"]
''',
        variables=[
            TemplateVariable("module_name", "Python module to run"),
        ],
    )

    GITHUB_ACTIONS = CodeTemplate(
        name="github_actions",
        description="GitHub Actions CI/CD workflow",
        category="devops",
        file_pattern=".github/workflows/{name}.yml",
        content='''name: {workflow_name}

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov={package_name} --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: false
''',
        variables=[
            TemplateVariable("workflow_name", "Name of the workflow"),
            TemplateVariable("package_name", "Name of the package"),
        ],
    )

    @classmethod
    def get_all_templates(cls) -> List[CodeTemplate]:
        """Get all available templates."""
        return [
            cls.PYTHON_CLASS,
            cls.PYTHON_DATACLASS,
            cls.PYTHON_TEST,
            cls.FASTAPI_ENDPOINT,
            cls.REACT_COMPONENT,
            cls.DOCKERFILE,
            cls.GITHUB_ACTIONS,
        ]

    @classmethod
    def get_template(cls, name: str) -> Optional[CodeTemplate]:
        """Get a template by name."""
        for template in cls.get_all_templates():
            if template.name == name:
                return template
        return None

    @classmethod
    def get_templates_by_category(cls, category: str) -> List[CodeTemplate]:
        """Get templates by category."""
        return [t for t in cls.get_all_templates() if t.category == category]


class ProjectScaffolder:
    """Scaffolds new projects from templates."""

    PROJECT_TEMPLATES = {
        "python_package": {
            "name": "Python Package",
            "description": "Standard Python package structure",
            "directories": [
                "src/{name}",
                "src/{name}/core",
                "tests",
                "docs",
                ".github/workflows",
            ],
            "files": {
                "pyproject.toml": '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.1.0"
description = "{description}"
readme = "README.md"
requires-python = ">=3.9"
license = {{text = "MIT"}}
authors = [
    {{name = "{author}", email = "{email}"}},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "black>=23.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]

[project.scripts]
{name} = "{name}.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100
target-version = ['py39']

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W", "UP"]
''',
                "README.md": '''# {name}

{description}

## Installation

```bash
pip install {name}
```

## Usage

```python
import {name}
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
''',
                "src/{name}/__init__.py": '''"""{description}"""

__version__ = "0.1.0"
''',
                "src/{name}/core/__init__.py": '''"""Core module."""''',
                "tests/__init__.py": '''"""Tests package.""''',
                "tests/test_basic.py": '''"""Basic tests."""

import pytest


def test_import():
    """Test package can be imported."""
    import {name}
    assert {name}.__version__ == "0.1.0"
''',
            },
        },
        "fastapi_app": {
            "name": "FastAPI Application",
            "description": "FastAPI web application with structure",
            "directories": [
                "app",
                "app/api",
                "app/core",
                "app/models",
                "app/services",
                "tests",
            ],
            "files": {
                "pyproject.toml": '''[project]
name = "{name}"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.9"
dependencies = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.23",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "httpx>=0.24",
]
''',
                "app/__init__.py": '''"""{description}"""''',
                "app/main.py": '''"""Main application."""

from fastapi import FastAPI
from app.api import router

app = FastAPI(title="{name}", version="0.1.0")
app.include_router(router)


@app.get("/health")
async def health_check():
    return {{"status": "healthy"}}
''',
                "app/api/__init__.py": '''"""API routes."""

from fastapi import APIRouter

router = APIRouter()

from app.api import items

router.include_router(items.router, prefix="/items")
''',
                "app/api/items.py": '''"""Items API."""

from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter()


@router.get("/")
async def list_items():
    return {{"items": []}}


@router.get("/{{item_id}}")
async def get_item(item_id: str):
    return {{"id": item_id, "name": "Item"}}
''',
                "app/core/__init__.py": '''"""Core module."""''',
                "app/core/config.py": '''"""Configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "{name}"
    debug: bool = False
    
    class Config:
        env_file = ".env"


settings = Settings()
''',
            },
        },
    }

    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.template_library = TemplateLibrary()

    def scaffold_project(
        self,
        template_name: str,
        name: str,
        description: str = "",
        author: str = "",
        email: str = "",
        variables: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Scaffold a new project."""
        if template_name not in self.PROJECT_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")

        template = self.PROJECT_TEMPLATES[template_name]
        project_dir = self.output_dir / name
        
        logger.info(f"Scaffolding {template['name']} at {project_dir}")

        # Create directories
        for dir_template in template["directories"]:
            dir_path = project_dir / dir_template.format(name=name)
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")

        # Create files
        vars_dict = {
            "name": name,
            "description": description or f"{name} project",
            "author": author,
            "email": email,
            **(variables or {}),
        }

        for file_template, content_template in template["files"].items():
            file_path = project_dir / file_template.format(name=name)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            content = content_template.format(**vars_dict)
            file_path.write_text(content)
            logger.debug(f"Created file: {file_path}")

        logger.info(f"Project scaffolded successfully at {project_dir}")
        return project_dir

    def generate_from_template(
        self,
        template_name: str,
        output_path: str,
        variables: Dict[str, str],
    ) -> Path:
        """Generate a file from a template."""
        template = self.template_library.get_template(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        # Format file path
        file_name = template.file_pattern.format(**variables)
        output_file = Path(output_path) / file_name
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Format content
        content = template.content.format(**variables)
        output_file.write_text(content)

        logger.info(f"Generated {template.name} at {output_file}")
        return output_file

    def list_templates(self) -> List[Dict[str, str]]:
        """List available project templates."""
        return [
            {
                "name": name,
                "description": template["description"],
            }
            for name, template in self.PROJECT_TEMPLATES.items()
        ]

    def list_code_templates(self) -> List[Dict[str, str]]:
        """List available code templates."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
            }
            for t in self.template_library.get_all_templates()
        ]
