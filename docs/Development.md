# Development

## Linting

Install the development dependencies and set up the pre-commit hooks:

```
pip install .[dev]
pre-commit install
pre-commit run --all-files
```

## Running Tests

Run the unit and integration tests with:

```
pytest tests/unit tests/integration
```

Online tests exercise real data sources. You can store credentials in a `.env` file and
they will be loaded automatically by pytest-dotenv:

```
pytest tests/online
```

Several data source tests verify that the test runs correctly when certain
configuration options are non-local UPaths rather than local ones. These tests use
UPaths on Google Cloud Storage and depend on GCS access. To run them, the `TEST_BUCKET`
and `TEST_PREFIX` environment variables must be set, e.g.:

```
TEST_BUCKET=test-bucket-rslearn
TEST_PREFIX=tests/
```

## Building the Documentation Locally

The documentation site is built with [MkDocs](https://www.mkdocs.org/) and the Material
theme. Install the docs dependencies and serve the site locally with live reload, from
the repository root:

```
pip install .[docs]
mkdocs serve
```

Then open the printed URL (by default http://127.0.0.1:8000/rslearn/) in your browser.
The site rebuilds automatically as you edit files under `docs/`.

## Releasing

### Automated Release Machinery

This project uses GitHub Actions ([`publish.yml`](https://github.com/allenai/rslearn/blob/master/.github/workflows/publish.yml)) to automatically build and publish to PyPI. The workflow triggers when a GitHub release is created.

The workflow:
- validates that the release tag version (`v*`) matches the version in [`pyproject.toml`](https://github.com/allenai/rslearn/blob/master/pyproject.toml)
- builds the distribution packages using `uv build`
- publishes to PyPI using trusted publishing

### Releasing Workflow

1. Confirm that the version in [`pyproject.toml`](https://github.com/allenai/rslearn/blob/master/pyproject.toml) matches the [semantic version](https://semver.org/) you plan to release. Update this if necessary.
    - Consider using [uv's version bump command](https://docs.astral.sh/uv/reference/cli/#uv-version--bump) to do this, since the project uses `uv` and this command may flag any problems.
1. If there are any changes, including a change to [`pyproject.toml`](https://github.com/allenai/rslearn/blob/master/pyproject.toml), open a pull request to merge those changes into `master`.
1. When that PR is merged, push a tag named `vX.Y.Z` to the merge commit from your merged PR.
1. Use GitHub's web UI to [create a release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release), selecting the tag you just pushed.
1. The GitHub release you created should automatically trigger the `publish` GitHub Action. Monitor that workflow to confirm success.
    - The workflow requires approval from a team member; if you cannot approve it yourself, contact someone who can.
1. Verify the new version is available on [PyPI](https://pypi.org/project/rslearn/).
