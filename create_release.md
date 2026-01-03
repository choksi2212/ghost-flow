# Create GitHub Release v0.1.0

Since GitHub CLI (`gh`) is not installed, here's how to create the release manually:

## Option 1: Via GitHub Web Interface (Easiest)

1. Go to: https://github.com/choksi2212/ghost-flow/releases/new

2. Fill in the form:
   - **Tag**: `v0.1.0` (select from dropdown or create new)
   - **Target**: `main`
   - **Title**: `GhostFlow v0.1.0 - Initial Release`
   - **Description**: Copy the contents from `RELEASE_NOTES_v0.1.0.md`

3. Click "Publish release"

## Option 2: Install GitHub CLI and Use Command

### Install GitHub CLI:
```powershell
# Using winget (Windows)
winget install --id GitHub.cli

# Or download from: https://cli.github.com/
```

### After installation, run:
```powershell
cd GHOSTFLOW

# Login to GitHub
gh auth login

# Create release
gh release create v0.1.0 `
  --title "GhostFlow v0.1.0 - Initial Release" `
  --notes-file RELEASE_NOTES_v0.1.0.md `
  --target main
```

## What's Already Done

✅ Git tag `v0.1.0` created locally  
✅ Release notes prepared in `RELEASE_NOTES_v0.1.0.md`  
✅ All code committed and pushed  
✅ Zero warnings, zero errors in CI  

## After Creating Release

The release will be visible at:
https://github.com/choksi2212/ghost-flow/releases

Users can then:
- Download source code
- View release notes
- Install via cargo (after publishing to crates.io)
