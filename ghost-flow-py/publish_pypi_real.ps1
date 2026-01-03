# Publish to REAL PyPI - This is the final step!
Write-Host "🚀 Publishing GhostFlow to REAL PyPI" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "⚠️  This will publish to the REAL PyPI!" -ForegroundColor Yellow
Write-Host "⚠️  Make sure you're using your REAL PyPI token (not TestPyPI)!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Get your token from: https://pypi.org/manage/account/" -ForegroundColor Cyan
Write-Host ""

# Confirm
$confirm = Read-Host "Are you ready to publish to REAL PyPI? (yes/no)"

if ($confirm -ne "yes") {
    Write-Host ""
    Write-Host "❌ Publishing cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
# Prompt for token securely
$token = Read-Host "Paste your REAL PyPI token (starts with pypi-)"

Write-Host ""
Write-Host "Publishing to PyPI..." -ForegroundColor Yellow
Write-Host ""

# Publish with token
python -m maturin publish --username __token__ --password $token

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host "✅ Successfully published to PyPI!" -ForegroundColor Green
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "🎉 GhostFlow is now available worldwide!" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "View your package at:" -ForegroundColor Yellow
    Write-Host "  https://pypi.org/project/ghost-flow/" -ForegroundColor White
    Write-Host ""
    Write-Host "Anyone can now install with:" -ForegroundColor Yellow
    Write-Host "  pip install ghost-flow" -ForegroundColor White
    Write-Host ""
    Write-Host "Test it yourself:" -ForegroundColor Yellow
    Write-Host "  pip install ghost-flow --upgrade" -ForegroundColor White
    Write-Host "  python -c `"import ghost_flow as gf; print('Success!')`"" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Publishing failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  - Invalid token (make sure it's from pypi.org, not test.pypi.org)" -ForegroundColor White
    Write-Host "  - Package name already taken" -ForegroundColor White
    Write-Host "  - Version already exists (update version in pyproject.toml)" -ForegroundColor White
    Write-Host ""
}
