# GhostFlow PyPI Publishing Script for Windows
# Run this to publish to PyPI

Write-Host "🚀 GhostFlow PyPI Publisher" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if maturin is installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$maturinInstalled = Get-Command maturin -ErrorAction SilentlyContinue
if (-not $maturinInstalled) {
    Write-Host "❌ Maturin not found. Installing..." -ForegroundColor Red
    pip install maturin
}

# Clean previous builds
Write-Host ""
Write-Host "🧹 Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "target\wheels") {
    Remove-Item -Recurse -Force target\wheels\*
}

# Build wheel
Write-Host ""
Write-Host "📦 Building wheel for Windows..." -ForegroundColor Yellow
maturin build --release

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Build successful!" -ForegroundColor Green

# Test locally
Write-Host ""
Write-Host "🧪 Testing local installation..." -ForegroundColor Yellow
$wheel = Get-Item target\wheels\*.whl | Select-Object -First 1
pip install --force-reinstall $wheel.FullName

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Installation failed!" -ForegroundColor Red
    exit 1
}

# Test import
Write-Host ""
Write-Host "Testing import..." -ForegroundColor Yellow
python -c "import ghost_flow as gf; print(f'✅ GhostFlow v{gf.__version__} works!'); x = gf.Tensor.randn([10, 10]); print(f'✅ Tensor operations work! Shape: {x.shape}')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Import test failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ All tests passed!" -ForegroundColor Green

# Ask for confirmation
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Ready to publish to PyPI!" -ForegroundColor Green
Write-Host ""
Write-Host "You will need your PyPI API token." -ForegroundColor Yellow
Write-Host "Get it from: https://pypi.org/manage/account/" -ForegroundColor Yellow
Write-Host ""
$confirm = Read-Host "Do you want to publish now? (yes/no)"

if ($confirm -ne "yes") {
    Write-Host ""
    Write-Host "❌ Publishing cancelled." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To publish later, run:" -ForegroundColor Cyan
    Write-Host "  maturin publish --username __token__ --password YOUR_TOKEN" -ForegroundColor White
    exit 0
}

# Get PyPI token
Write-Host ""
$token = Read-Host "Enter your PyPI token (starts with pypi-)" -AsSecureString
$tokenPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($token)
)

# Publish
Write-Host ""
Write-Host "📤 Publishing to PyPI..." -ForegroundColor Yellow
maturin publish --username __token__ --password $tokenPlain

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Publishing failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  - Invalid token" -ForegroundColor White
    Write-Host "  - Version already exists (update version in pyproject.toml)" -ForegroundColor White
    Write-Host "  - Network issues" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "✅ Successfully published to PyPI!" -ForegroundColor Green
Write-Host ""
Write-Host "Users can now install with:" -ForegroundColor Cyan
Write-Host "  pip install ghost-flow" -ForegroundColor White
Write-Host ""
Write-Host "Test it yourself:" -ForegroundColor Cyan
Write-Host "  pip install ghost-flow --upgrade" -ForegroundColor White
Write-Host ""
Write-Host "🎉 Congratulations! GhostFlow is now available worldwide!" -ForegroundColor Green
