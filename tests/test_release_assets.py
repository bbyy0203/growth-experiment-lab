from pathlib import Path


def test_readme_commands_match_cli():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "python -m growth_experiment_lab run-all --config configs/demo.toml" in readme
    assert "python -m growth_experiment_lab export-examples --config configs/demo.toml" in readme


def test_github_actions_is_cross_platform_and_uses_ci_config():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "runs-on: ubuntu-latest" in workflow
    assert 'python-version: "3.11"' in workflow
    assert 'python -m pip install -e ".[dev]"' in workflow
    assert "run-all --config configs/ci.toml --overwrite" in workflow
    assert "D:\\" not in workflow and "Anaconda" not in workflow


def test_public_source_docs_do_not_contain_local_identity_or_secrets():
    paths = [Path("README.md"), Path("PROJECT_SPEC.md"), *Path("docs").glob("*.md")]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "D:\\Anaconda" not in combined
    assert "谭冰月" not in combined
    assert "api_key" not in combined.lower()
