# Quick PyPI Publish Script
# This will publish the already-built wheel to PyPI

Write-Host "🚀 Publishing GhostFlow to PyPI..." -ForegroundColor Cyan
Write-Host ""
Write-Host "When prompted, enter your PyPI API token" -ForegroundColor Yellow
Write-Host "(It starts with 'pypi-')" -ForegroundColor Yellow
Write-Host ""

# Navigate to the Python bindings directory
Set-Location ghost-flow-py

# Publish using maturin
python -m maturin publish

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "If successful, users can now install with:" -ForegroundColor Green
Write-Host "  pip install ghost-flow" -ForegroundColor White
Write-Host ""
