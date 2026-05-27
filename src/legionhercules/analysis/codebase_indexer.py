"""Codebase indexing for project understanding."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Symbol:
    """Represents a code symbol (function, class, variable)."""
    name: str
    type: str  # function, class, method, variable
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)


@dataclass
class FileInfo:
    """Information about a source file."""
    path: str
    language: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    size_bytes: int = 0
    line_count: int = 0


@dataclass
class ProjectIndex:
    """Complete index of a codebase."""
    root_path: str
    files: list[FileInfo] = field(default_factory=list)
    symbols: dict[str, list[Symbol]] = field(default_factory=dict)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> ProjectIndex:
        files = [FileInfo(**f) for f in data.get("files", [])]
        symbols = {
            k: [Symbol(**s) for s in v]
            for k, v in data.get("symbols", {}).items()
        }
        return cls(
            root_path=data["root_path"],
            files=files,
            symbols=symbols,
            dependencies=data.get("dependencies", {}),
            summary=data.get("summary", {}),
        )


class CodebaseIndexer:
    """Indexes a codebase for project understanding."""
    
    SUPPORTED_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "jsx",
        ".tsx": "tsx",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the indexer.
        
        Args:
            cache_dir: Directory to cache indexes (default: ~/.legionhercules/indexes)
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".legionhercules" / "indexes"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._current_index: Optional[ProjectIndex] = None
    
    def index_project(
        self,
        project_path: str,
        exclude_patterns: Optional[list[str]] = None,
    ) -> ProjectIndex:
        """Index a project directory.
        
        Args:
            project_path: Path to project root
            exclude_patterns: Patterns to exclude (e.g., ["*.pyc", "node_modules"])
        
        Returns:
            ProjectIndex with complete codebase information
        """
        root = Path(project_path).resolve()
        exclude = exclude_patterns or ["node_modules", ".git", "__pycache__", ".venv", "venv", ".env", "dist", "build", ".pytest_cache", ".mypy_cache", "*.pyc", ".egg-info"]
        
        logger.info(f"Indexing project: {root}")
        
        files = []
        symbols_by_type: dict[str, list[Symbol]] = {}
        all_dependencies: dict[str, list[str]] = {}
        
        for file_path in self._walk_files(root, exclude):
            try:
                file_info = self._analyze_file(file_path, root)
                if file_info:
                    files.append(file_info)
                    
                    # Collect symbols
                    for sym in file_info.symbols:
                        if sym.type not in symbols_by_type:
                            symbols_by_type[sym.type] = []
                        symbols_by_type[sym.type].append(sym)
                    
                    # Collect dependencies
                    if file_info.dependencies:
                        all_dependencies[file_info.path] = file_info.dependencies
                        
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # Generate summary
        summary = self._generate_summary(files, symbols_by_type)
        
        index = ProjectIndex(
            root_path=str(root),
            files=files,
            symbols=symbols_by_type,
            dependencies=all_dependencies,
            summary=summary,
        )
        
        self._current_index = index
        logger.info(f"Indexed {len(files)} files with {sum(len(s) for s in symbols_by_type.values())} symbols")
        
        return index
    
    def _walk_files(self, root: Path, exclude: list[str]) -> list[Path]:
        """Walk directory and return files to analyze."""
        files = []
        
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            
            # Check exclusion
            rel_path = str(path.relative_to(root))
            if any(self._matches_pattern(rel_path, pattern) for pattern in exclude):
                continue
            
            # Check extension
            if path.suffix in self.SUPPORTED_EXTENSIONS:
                files.append(path)
        
        return files
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches exclusion pattern."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern)
    
    def _analyze_file(self, file_path: Path, root: Path) -> Optional[FileInfo]:
        """Analyze a single file."""
        language = self.SUPPORTED_EXTENSIONS.get(file_path.suffix)
        if not language:
            return None
        
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        
        file_info = FileInfo(
            path=str(file_path.relative_to(root)),
            language=language,
            size_bytes=file_path.stat().st_size,
            line_count=len(lines),
        )
        
        if language == "python":
            self._analyze_python_file(file_path, content, file_info)
        elif language in ("javascript", "typescript", "jsx", "tsx"):
            self._analyze_js_file(file_path, content, file_info)
        
        return file_info
    
    def _analyze_python_file(self, file_path: Path, content: str, file_info: FileInfo) -> None:
        """Analyze Python file using AST."""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        file_info.imports.append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    file_info.imports.append(module)
                
                elif isinstance(node, ast.FunctionDef):
                    symbol = Symbol(
                        name=node.name,
                        type="function",
                        file_path=file_info.path,
                        line_number=node.lineno,
                        docstring=ast.get_docstring(node),
                        signature=self._get_function_signature(node),
                    )
                    file_info.symbols.append(symbol)
                
                elif isinstance(node, ast.ClassDef):
                    symbol = Symbol(
                        name=node.name,
                        type="class",
                        file_path=file_info.path,
                        line_number=node.lineno,
                        docstring=ast.get_docstring(node),
                    )
                    file_info.symbols.append(symbol)
                    
                    # Add methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method = Symbol(
                                name=f"{node.name}.{item.name}",
                                type="method",
                                file_path=file_info.path,
                                line_number=item.lineno,
                                docstring=ast.get_docstring(item),
                                signature=self._get_function_signature(item),
                            )
                            file_info.symbols.append(method)
        
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature from AST node."""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"
        
        return f"({', '.join(args)}){returns}"
    
    def _analyze_js_file(self, file_path: Path, content: str, file_info: FileInfo) -> None:
        """Analyze JavaScript/TypeScript file using regex patterns."""
        import re
        
        # Find imports
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'require\([\'"]([^\'"]+)[\'"]\)',
            r'import\s+[\'"]([^\'"]+)[\'"]',
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            file_info.imports.extend(matches)
        
        # Find functions
        func_pattern = r'(?:async\s+)?function\s+(\w+)\s*\('
        for match in re.finditer(func_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            symbol = Symbol(
                name=match.group(1),
                type="function",
                file_path=file_info.path,
                line_number=line_num,
            )
            file_info.symbols.append(symbol)
        
        # Find classes
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            symbol = Symbol(
                name=match.group(1),
                type="class",
                file_path=file_info.path,
                line_number=line_num,
            )
            file_info.symbols.append(symbol)
    
    def _generate_summary(
        self,
        files: list[FileInfo],
        symbols: dict[str, list[Symbol]],
    ) -> dict[str, Any]:
        """Generate project summary."""
        languages = {}
        for f in files:
            languages[f.language] = languages.get(f.language, 0) + 1
        
        total_lines = sum(f.line_count for f in files)
        total_size = sum(f.size_bytes for f in files)
        
        return {
            "total_files": len(files),
            "total_lines": total_lines,
            "total_size_bytes": total_size,
            "languages": languages,
            "symbol_counts": {k: len(v) for k, v in symbols.items()},
        }
    
    def save_index(self, index: ProjectIndex, name: Optional[str] = None) -> str:
        """Save index to cache."""
        index_name = name or f"index_{Path(index.root_path).name}_{hash(index.root_path) % 10000}"
        cache_file = self.cache_dir / f"{index_name}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(index.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved index to {cache_file}")
        return str(cache_file)
    
    def load_index(self, name: str) -> Optional[ProjectIndex]:
        """Load index from cache."""
        cache_file = self.cache_dir / f"{name}.json"
        
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        return ProjectIndex.from_dict(data)
    
    def query_symbols(
        self,
        query: str,
        symbol_type: Optional[str] = None,
    ) -> list[Symbol]:
        """Query symbols by name pattern."""
        if not self._current_index:
            return []
        
        results = []
        query_lower = query.lower()
        
        types_to_search = [symbol_type] if symbol_type else self._current_index.symbols.keys()
        
        for sym_type in types_to_search:
            for symbol in self._current_index.symbols.get(sym_type, []):
                if query_lower in symbol.name.lower():
                    results.append(symbol)
        
        return results
    
    def get_file_dependencies(self, file_path: str) -> list[str]:
        """Get dependencies for a file."""
        if not self._current_index:
            return []
        
        return self._current_index.dependencies.get(file_path, [])
    
    def find_symbol_references(self, symbol_name: str) -> list[str]:
        """Find all files that reference a symbol."""
        if not self._current_index:
            return []
        
        references = []
        for file_info in self._current_index.files:
            if any(symbol_name in imp for imp in file_info.imports):
                references.append(file_info.path)
        
        return references
