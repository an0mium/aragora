## Summary

<!-- Describe what this PR does in 1-3 bullet points -->

-

## Type of Change

<!-- Check all that apply -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Test improvement
- [ ] Performance improvement

## Related Issues

<!-- Link any related issues using "Fixes #123" or "Related to #123" -->

## Checklist

### Before Submitting

- [ ] I have read the [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines
- [ ] My code follows the project's code style (ran `make lint` and `make format`)
- [ ] I have added tests that prove my fix/feature works
- [ ] All new and existing tests pass (`make test-fast`)
- [ ] I have updated documentation if needed
- [ ] My commit messages follow conventional commit format

### Code Quality

- [ ] No new `# type: ignore` comments without justification
- [ ] No hardcoded secrets or credentials
- [ ] Error handling is appropriate (not bare `except:`)
- [ ] New code has type hints

### For New Features

- [ ] Feature is documented in relevant `docs/*.md` file
- [ ] Feature is tested with unit tests
- [ ] Feature integrates with existing systems appropriately

### For Bug Fixes

- [ ] Root cause is identified and explained
- [ ] Test case added to prevent regression
- [ ] Related edge cases considered

## Test Plan

<!-- How did you test these changes? Include commands, screenshots, or test output -->

```bash
# Example: Run specific tests
pytest tests/path/to/test.py -v
```

## Screenshots / Demo

<!-- If applicable, add screenshots or GIFs showing the change -->

## Additional Notes

<!-- Any additional context, considerations, or trade-offs -->

---
<!--
Reviewer Guidelines:
- Check that tests are comprehensive
- Verify no breaking changes to public APIs
- Ensure documentation is updated
- Look for security implications
-->
