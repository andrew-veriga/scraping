import { NextRequest, NextResponse } from 'next/server';
import { MessageDetails, ApiResponse } from '@/types';

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8001';

// GET /api/messages/[id] - получить детали сообщения
export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const response = await fetch(`${API_BASE_URL}/messages/${params.id}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      if (response.status === 404) {
        return NextResponse.json(
          { error: 'Message not found' } as ApiResponse<never>,
          { status: 404 }
        );
      }
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json({ data } as ApiResponse<MessageDetails>);
  } catch (error) {
    console.error('Error fetching message:', error);
    return NextResponse.json(
      { error: 'Failed to fetch message' } as ApiResponse<never>,
      { status: 500 }
    );
  }
}
