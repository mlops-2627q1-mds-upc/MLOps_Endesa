"""Temporary failing test to verify the pull request check rejects a failure."""


def test_ci_rejects_failure():
    assert False, "intentional CI failure probe"
