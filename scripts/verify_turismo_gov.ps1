$url = "https://dados.turismo.gov.br/dataset/"

try {
    $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
    
    if ($response.StatusCode -eq 200) {
        
        $wshell = New-Object -ComObject Wscript.Shell
        $wshell.Popup("O portal de Dados do Turismo voltou a funcionar!", 0, "Monitor de Status", 64)
        
        Disable-ScheduledTask -TaskName "MonitorTurismo" -ErrorAction SilentlyContinue
    }
} catch {
    $_ | Out-File "C:\git_reps\brazil_tourism_forecasting_benchmark\logs\monitor.log" -Append
}