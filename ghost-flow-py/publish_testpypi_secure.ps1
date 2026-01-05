# Secure TestPyPI Publishing Script
Write-Host "🧪 Publishing to TestPyPI..." -ForegroundColor Cyan
Write-Host ""

# Prompt for token securely
$token = Read-Host "Paste your TestPyPI token (starts with pypi-)"

Write-Host ""
Write-Host "Publishing..." -ForegroundColor Yellow

# Publish with token
python -m maturin publish --repository testpypi --username __token__ --password $token

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Successfully published to TestPyPI!" -ForegroundColor Green
    Write-Host ""
    Write-Host "View your package at:" -ForegroundColor Cyan
    Write-Host "  https://test.pypi.org/project/ghost-flow/" -ForegroundColor White
    Write-Host ""
    Write-Host "Test installation:" -ForegroundColor Yellow
    Write-Host "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ghost-flow" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ Publishing failed!" -ForegroundColor Red
}
