# Releasing

## Automated Release Machinery

This project uses GitHub Actions ([`publish.yml`](../.github/workflows/publish.yml)) to automatically build and publish to PyPI. The workflow triggers when a GitHub release is created.

The workflow:
- validates that the release tag version (`v*`) matches the version in [`pyproject.toml`](../pyproject.toml)
- builds the distribution packages using `uv build`
- publishes to PyPI using trusted publishing

## Releasing Workflow

1. Confirm that the version in [`pyproject.toml`](../pyproject.toml) matches the [semantic version](https://semver.org/) you plan to release. Update this if necessary.
    - Consider using [uv's version bump command](https://docs.astral.sh/uv/reference/cli/#uv-version--bump) to do this, since the project uses `uv` and this command may flag any problems.
1. If there are any changes, including a change to [`pyproject.toml`](../pyproject.toml), open a pull request to merge those changes into `master`.
1. When that PR is merged, push a tag named `vX.Y.Z` to the merge commit from your merged PR.
1. Use GitHub's web UI to [create a release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release), selecting the tag you just pushed.
1. The GitHub release you created should automatically trigger the `publish` GitHub Action. Monitor that workflow to confirm success.
    - The workflow requires approval from a team member; if you cannot approve it yourself, contact someone who can.
1. Verify the new version is available on [PyPI](https://pypi.org/project/rslearn/).
