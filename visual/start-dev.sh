#!/bin/bash

# Discord SUI Analyzer Visual Editor - Development Startup Script

echo "🚀 Starting Discord SUI Analyzer Visual Editor..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18+ is required. Current version: $(node -v)"
    exit 1
fi

echo "✅ Node.js version: $(node -v)"

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "✅ Dependencies already installed"
fi

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "⚙️  Creating .env.local file..."
    cat > .env.local << EOF
API_BASE_URL=http://localhost:8001
NEXT_PUBLIC_API_URL=http://localhost:8001
EOF
    echo "✅ .env.local created"
else
    echo "✅ .env.local already exists"
fi

# Check if main API server is running
echo "🔍 Checking if main API server is running..."
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Main API server is running on port 8001"
else
    echo "⚠️  Main API server is not running on port 8001"
    echo "   Please start the main Discord SUI Analyzer server first:"
    echo "   cd .. && python -m uvicorn app.main:app --host 0.0.0.0 --port 8001"
    echo ""
    echo "   Or continue anyway (API calls will fail until server is started)"
fi

echo ""
echo "🎯 Starting Next.js development server..."
echo "   Visual Editor will be available at: http://localhost:3000"
echo "   Main API should be running at: http://localhost:8001"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the development server
npm run dev
