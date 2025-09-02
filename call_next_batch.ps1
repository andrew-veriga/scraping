# This script sends a POST request to the /process-next-batch endpoint
# of the running FastAPI application.

$uri = "http://127.0.0.1:8000/process-next-batch"

Write-Host "Sending POST request to $uri"

try {
    # Use Invoke-RestMethod to make the web request
    $response = Invoke-RestMethod -Uri $uri -Method POST

    # Print the server's response
    Write-Host "Request successful. Server response:"
    $response | ConvertTo-Json | Write-Host
} catch {
    Write-Host "An error occurred:" -ForegroundColor Red
    Write-Host $_.Exception.Message
}