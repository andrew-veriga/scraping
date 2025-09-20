import { NextRequest, NextResponse } from 'next/server';
import { HierarchyOperationRequest, ApiResponse } from '@/types';

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8001';

// POST /api/threads/hierarchy - выполнить операцию с иерархией
export async function POST(request: NextRequest) {
  try {
    const body: HierarchyOperationRequest = await request.json();
    
    const response = await fetch(`${API_BASE_URL}/threads/hierarchy`, {
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
    return NextResponse.json({ data } as ApiResponse<any>);
  } catch (error) {
    console.error('Error performing hierarchy operation:', error);
    return NextResponse.json(
      { error: 'Failed to perform hierarchy operation' } as ApiResponse<never>,
      { status: 500 }
    );
  }
}
