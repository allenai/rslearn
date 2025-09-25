Tests
-----

Several data source tests verify that the test runs correctly when certain
configuration options are non-local UPaths rather than local ones. These tests use
UPaths on Google Cloud Storage and depend on GCS access. To run them, the TEST_BUCKET
and TEST_PREFIX environment variables must be set, e.g.:

```
TEST_BUCKET=test-bucket-rslearn
TEST_PREFIX=tests/
```

Secrets and External Services
-----------------------------

Some integrations require API credentials at test time. The pytest configuration (see
`pyproject.toml`) together with an early hook in `tests/conftest.py` uses
`python-dotenv` to load any key-value pairs defined in the repository-level `.env`
file before the test session starts. Populate that file with the required environment
variables, e.g.:

```
EDS_CLIENT_ID=...
EDS_SECRET=...
EDS_AUTH_URL=...
EDS_API_URL=...
```

Note that the `.env` file is ignored by git.
