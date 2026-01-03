# Publish to TestPyPI - Safe Testing Before Real PyPI
# Run this first to test everything works!

Write-Host "🧪 Publishing GhostFlow to TestPyPI (Test Server)" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This is a SAFE test - it won't affect the real PyPI!" -ForegroundColor Green
Write-Host ""
Write-Host "You'll need your TestPyPI API token from:" -ForegroundColor Yellow
Write-Host "  https://test.pypi.org/manage/account/" -ForegroundColor White
Write-Host ""

# Publish to TestPyPI
Write-Host "Publishing to TestPyPI..." -ForegroundColor Yellow
Write-Host ""
Write-Host "When prompted for username, enter: __token__" -ForegroundColor Cyan
Write-Host "When prompted for password, paste your TestPyPI token (starts with pypi-)" -ForegroundColor Cyan
Write-Host ""

python -m maturin publish --repository testpypi

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=================================================" -ForegroundColor Cyan
    Write-Host "✅ Successfully published to TestPyPI!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Now test the installation:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Create a test environment:" -ForegroundColor Cyan
    Write-Host "   python -m venv test_env" -ForegroundColor White
    Write-Host "   test_env\Scripts\activate" -ForegroundColor White
    Write-Host ""
    Write-Host "2. Install from TestPyPI:" -ForegroundColor Cyan
    Write-Host "   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ghost-flow" -ForegroundColor White
    Write-Host ""
    Write-Host "3. Test it works:" -ForegroundColor Cyan
    Write-Host "   python -c `"import ghost_flow as gf; print('Success!'); x = gf.Tensor.randn([10,10]); print(x.shape)`"" -ForegroundColor White
    Write-Host ""
    Write-Host "4. If everything works, publish to real PyPI:" -ForegroundColor Cyan
    Write-Host "   python -m maturin publish" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Publishing failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  - Invalid token (make sure it's from test.pypi.org, not pypi.org)" -ForegroundColor White
    Write-Host "  - Package name already taken on TestPyPI" -ForegroundColor White
    Write-Host "  - Network issues" -ForegroundColor White
    Write-Host ""
}
