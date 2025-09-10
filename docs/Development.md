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
