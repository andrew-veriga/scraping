import { NextRequest, NextResponse } from 'next/server';
import { Thread, ThreadList, ApiResponse } from '@/types';

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8001';

// GET /api/threads - получить все threads
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get('limit');
    const offset = searchParams.get('offset');
    const status = searchParams.get('status');
    const technical = searchParams.get('technical');

    let url = `${API_BASE_URL}/threads`;
    const params = new URLSearchParams();
    
    // Always use file source for now
    params.append('source', 'file');
    if (limit) params.append('limit', limit);
    if (offset) params.append('offset', offset);
    if (status) params.append('status', status);
    if (technical) params.append('technical', technical);
    
    if (params.toString()) {
      url += `?${params.toString()}`;
    }

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json({ data } as ApiResponse<Thread[]>);
  } catch (error) {
    console.error('Error fetching threads:', error);
    return NextResponse.json(
      { error: 'Failed to fetch threads' } as ApiResponse<never>,
      { status: 500 }
    );
  }
}

// POST /api/threads - создать новый thread
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    const response = await fetch(`${API_BASE_URL}/threads`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json({ data } as ApiResponse<Thread>);
  } catch (error) {
    console.error('Error creating thread:', error);
    return NextResponse.json(
      { error: 'Failed to create thread' } as ApiResponse<never>,
      { status: 500 }
    );
  }
}
