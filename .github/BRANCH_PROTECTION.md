# Branch Protection Setup Guide

This document provides instructions for setting up branch protection for the main branch of the MEssE repository.

## Overview

Branch protection ensures that all changes to the main branch go through proper review and approval processes. This helps maintain code quality and prevents accidental or unauthorized changes.

## Required Protection Rules

The main branch should have the following protection rules configured:

### 1. Require Pull Request Reviews Before Merging
- **Required approving reviews**: 1
- **Require review from Code Owners**: ✓ (enabled)
- **Dismiss stale pull request approvals when new commits are pushed**: ✓ (enabled)

### 2. Code Owners
All pull requests automatically request review from `@liuquan18` (defined in `.github/CODEOWNERS`)

### 3. Additional Recommendations
- **Require conversation resolution before merging**: ✓ (enabled)
- **Require branches to be up to date before merging**: ✓ (enabled)
- **Do not allow bypassing the above settings**: ✓ (enabled for non-admins)

## How to Configure Branch Protection

### Method 1: Via GitHub Web Interface

1. Navigate to the repository: https://github.com/liuquan18/MEssE
2. Click on **Settings** tab
3. Click on **Branches** in the left sidebar
4. Click **Add branch protection rule** (or edit existing rule)
5. In "Branch name pattern", enter: `main`
6. Enable the following settings:
   - ✅ **Require a pull request before merging**
     - ✅ Require approvals: **1**
     - ✅ Dismiss stale pull request approvals when new commits are pushed
     - ✅ Require review from Code Owners
   - ✅ **Require conversation resolution before merging**
   - ✅ **Require linear history** (optional, but recommended)
   - ✅ **Do not allow bypassing the above settings** (optional)
7. Click **Create** or **Save changes**

### Method 2: Via GitHub API

You can use the GitHub API to configure branch protection programmatically. Here's an example using `curl`:

```bash
curl -X PUT \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/liuquan18/MEssE/branches/main/protection \
  -d '{
    "required_status_checks": null,
    "enforce_admins": false,
    "required_pull_request_reviews": {
      "required_approving_review_count": 1,
      "dismiss_stale_reviews": true,
      "require_code_owner_reviews": true
    },
    "restrictions": null,
    "required_linear_history": false,
    "allow_force_pushes": false,
    "allow_deletions": false,
    "required_conversation_resolution": true
  }'
```

### Method 3: Using GitHub CLI

If you have the GitHub CLI installed:

```bash
# Enable branch protection for main branch
gh api repos/liuquan18/MEssE/branches/main/protection \
  --method PUT \
  --field required_pull_request_reviews[required_approving_review_count]=1 \
  --field required_pull_request_reviews[dismiss_stale_reviews]=true \
  --field required_pull_request_reviews[require_code_owner_reviews]=true \
  --field required_conversation_resolution=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false
```

## Files in This Repository

- **`.github/CODEOWNERS`**: Defines code owners who must review PRs
- **`.github/settings.yml`**: Configuration file documenting branch protection settings
- **`.github/BRANCH_PROTECTION.md`**: This documentation file

## Verification

After setting up branch protection, you can verify it's working by:

1. Try to push directly to the main branch (should be blocked)
2. Create a test branch and open a PR
3. Verify that @liuquan18 is automatically requested as a reviewer
4. Verify that the PR cannot be merged without approval

## Troubleshooting

### Issue: Cannot push to main branch
**Solution**: This is expected behavior. Create a feature branch and open a pull request instead.

```bash
git checkout -b feature/your-feature-name
git add .
git commit -m "Your changes"
git push origin feature/your-feature-name
```

### Issue: Code owner not automatically requested for review
**Solution**: Ensure the `.github/CODEOWNERS` file is in the main branch and properly formatted.

### Issue: Can still merge PR without approval
**Solution**: Check that branch protection rules are properly configured and that "Require review from Code Owners" is enabled.

## References

- [About protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [About code owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [Managing a branch protection rule](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule)
