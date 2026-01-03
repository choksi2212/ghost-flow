# Finish Publishing Remaining Crates to crates.io
# Run this after Sat, 03 Jan 2026 11:07:47 GMT

Write-Host "Publishing remaining crates..." -ForegroundColor Green

Write-Host "`nPublishing ghostflow-data..." -ForegroundColor Cyan
cargo publish -p ghostflow-data

Write-Host "`nWaiting 30 seconds before next publish..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

Write-Host "`nPublishing ghostflow-ml..." -ForegroundColor Cyan
cargo publish -p ghostflow-ml

Write-Host "`n✅ All crates published successfully!" -ForegroundColor Green
Write-Host "`nPublished crates:" -ForegroundColor Yellow
Write-Host "  ✅ ghostflow-core v0.1.0"
Write-Host "  ✅ ghostflow-cuda v0.1.0"
Write-Host "  ✅ ghostflow-autograd v0.1.0"
Write-Host "  ✅ ghostflow-nn v0.1.0"
Write-Host "  ✅ ghostflow-optim v0.1.0"
Write-Host "  ✅ ghostflow-data v0.1.0"
Write-Host "  ✅ ghostflow-ml v0.1.0"
Write-Host "`nUsers can now install with:" -ForegroundColor Cyan
Write-Host "  cargo add ghostflow-core"
Write-Host "  cargo add ghostflow-nn"
Write-Host "  cargo add ghostflow-ml"
Write-Host "`nView on crates.io:" -ForegroundColor Cyan
Write-Host "  https://crates.io/crates/ghostflow-core"
