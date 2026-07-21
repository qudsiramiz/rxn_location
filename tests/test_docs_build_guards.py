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


def test_docs_workflow_deploys_versioned_documentation():
    workflow = (REPO_ROOT / ".github/workflows/docs.yml").read_text()

    assert "fetch-depth: 0" in workflow
    assert "poetry version --short" in workflow
    assert "mike deploy --push --update-aliases" in workflow
    assert "mike set-default --push latest" in workflow


def test_docs_config_enables_mike_version_selector():
    config = (REPO_ROOT / "mkdocs.yml").read_text()

    assert "  - mike\n" in config
