import ast
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_docs_modules_have_valid_python_syntax():
    for relative_path in (
        "src/rxn_location/plotly_utils.py",
        "src/rxn_location/master_jet_list.py",
        "src/rxn_location/rc_stats_fncs.py",
    ):
        source = (REPO_ROOT / relative_path).read_text()
        ast.parse(source)


def test_pytz_is_declared_as_a_runtime_dependency():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    dependencies = pyproject["tool"]["poetry"]["dependencies"]

    assert "pytz" in dependencies


def test_pytz_is_locked_in_the_main_dependency_group():
    poetry_lock = tomllib.loads((REPO_ROOT / "poetry.lock").read_text())
    pytz_package = next(
        package for package in poetry_lock["package"] if package["name"] == "pytz"
    )

    assert "main" in pytz_package.get("groups", [])
