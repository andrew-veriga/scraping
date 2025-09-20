# Discord SUI Analyzer Visual Editor - Development Startup Script for Windows

Write-Host "🚀 Starting Discord SUI Analyzer Visual Editor..." -ForegroundColor Green

# Check if Node.js is installed
try {
    $nodeVersion = node -v
    Write-Host "✅ Node.js version: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js is not installed. Please install Node.js 18+ first." -ForegroundColor Red
    exit 1
}

# Check if npm is installed
try {
    $npmVersion = npm -v
    Write-Host "✅ npm version: $npmVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ npm is not installed. Please install npm first." -ForegroundColor Red
    exit 1
}

# Check if dependencies are installed
if (-not (Test-Path "node_modules")) {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "✅ Dependencies already installed" -ForegroundColor Green
}

# Check if .env.local exists
if (-not (Test-Path ".env.local")) {
    Write-Host "⚙️  Creating .env.local file..." -ForegroundColor Yellow
    @"
API_BASE_URL=http://localhost:8001
NEXT_PUBLIC_API_URL=http://localhost:8001
"@ | Out-File -FilePath ".env.local" -Encoding UTF8
    Write-Host "✅ .env.local created" -ForegroundColor Green
} else {
    Write-Host "✅ .env.local already exists" -ForegroundColor Green
}

# Check if main API server is running
Write-Host "🔍 Checking if main API server is running..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8001/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Main API server is running on port 8001" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Main API server is not running on port 8001" -ForegroundColor Yellow
    Write-Host "   Please start the main Discord SUI Analyzer server first:" -ForegroundColor Yellow
    Write-Host "   cd .. && python -m uvicorn app.main:app --host 0.0.0.0 --port 8001" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   Or continue anyway (API calls will fail until server is started)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎯 Starting Next.js development server..." -ForegroundColor Green
Write-Host "   Visual Editor will be available at: http://localhost:3000" -ForegroundColor Cyan
Write-Host "   Main API should be running at: http://localhost:8001" -ForegroundColor Cyan
Write-Host ""
Write-Host "   Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the development server
npm run dev
