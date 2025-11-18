# Repair-DockerWSL.ps1
# Safe repair for Docker Desktop + WSL2 integration
# Fixes "Wsl/Service/E_UNEXPECTED" and config write errors.

Write-Host "Stopping WSL and Docker Desktop..." -ForegroundColor Cyan
wsl --shutdown 2>$null
Stop-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

Write-Host "Restarting Docker Desktop..." -ForegroundColor Cyan
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
Start-Sleep -Seconds 15

Write-Host "Checking available WSL distributions..." -ForegroundColor Yellow
wsl -l -v

# Auto-detect Ubuntu
$distro = "Ubuntu"
$exists = wsl -l -q | Select-String -Pattern $distro

if ($exists) {
    Write-Host "Ubuntu detected. Opening a repair session..." -ForegroundColor Green

    # Create ~/.docker if missing and reset config.json
    wsl -d $distro -- bash -c "mkdir -p ~/.docker && rm -f ~/.docker/config.json && echo '{}' > ~/.docker/config.json && chmod 600 ~/.docker/config.json"

    # Verify write permission
    Write-Host "Testing write permissions in ~/.docker ..." -ForegroundColor Yellow
    wsl -d $distro -- bash -c "echo test > ~/.docker/test_write && cat ~/.docker/test_write && rm ~/.docker/test_write"

    Write-Host "Docker config directory repaired inside Ubuntu." -ForegroundColor Green
}
else {
    Write-Host "Ubuntu WSL distribution not found. Please ensure it's installed (wsl --install -d Ubuntu)." -ForegroundColor Red
}

Write-Host "Restarting WSL services..." -ForegroundColor Cyan
wsl --shutdown
Start-Sleep -Seconds 5

Write-Host "Attempting Docker CLI check..." -ForegroundColor Cyan
try {
    docker version
}
catch {
    Write-Host "Docker CLI not responding yet. Please open Docker Desktop manually and wait for it to start." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Repair script completed. If Docker still fails, try reopening Docker Desktop or rebooting Windows." -ForegroundColor Green
