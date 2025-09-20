import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { Thread, MessageDetails, Conversation, HierarchyNode } from '@/types';

interface ThreadsState {
  threads: Thread[];
  messages: Record<string, MessageDetails>;
  conversations: Record<string, Conversation>;
  hierarchy: HierarchyNode[];
  loading: boolean;
  error: string | null;
  selectedThreadId: string | null;
  expandedNodes: string[];
}

const initialState: ThreadsState = {
  threads: [],
  messages: {},
  conversations: {},
  hierarchy: [],
  loading: false,
  error: null,
  selectedThreadId: null,
  expandedNodes: [],
};

// Async thunks
export const fetchThreads = createAsyncThunk(
  'threads/fetchThreads',
  async (params?: { limit?: number; offset?: number; status?: string; technical?: boolean }) => {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.offset) searchParams.append('offset', params.offset.toString());
    if (params?.status) searchParams.append('status', params.status);
    if (params?.technical) searchParams.append('technical', params.technical.toString());

    const response = await fetch(`/api/threads?${searchParams.toString()}`);
    if (!response.ok) {
      throw new Error('Failed to fetch threads');
    }
    const result = await response.json();
    return result.data;
  }
);

export const fetchThread = createAsyncThunk(
  'threads/fetchThread',
  async (threadId: string) => {
    const response = await fetch(`/api/threads/${threadId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch thread');
    }
    const result = await response.json();
    return result.data;
  }
);

export const updateThread = createAsyncThunk(
  'threads/updateThread',
  async ({ threadId, updates }: { threadId: string; updates: Partial<Thread> }) => {
    const response = await fetch(`/api/threads/${threadId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updates),
    });
    if (!response.ok) {
      throw new Error('Failed to update thread');
    }
    const result = await response.json();
    return result.data;
  }
);

export const performHierarchyOperation = createAsyncThunk(
  'threads/performHierarchyOperation',
  async (operation: any) => {
    const response = await fetch('/api/threads/hierarchy', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(operation),
    });
    if (!response.ok) {
      throw new Error('Failed to perform hierarchy operation');
    }
    const result = await response.json();
    return result.data;
  }
);

const threadsSlice = createSlice({
  name: 'threads',
  initialState,
  reducers: {
    setSelectedThread: (state, action: PayloadAction<string | null>) => {
      state.selectedThreadId = action.payload;
    },
    toggleNodeExpansion: (state, action: PayloadAction<string>) => {
      const nodeId = action.payload;
      const index = state.expandedNodes.indexOf(nodeId);
      if (index > -1) {
        state.expandedNodes.splice(index, 1);
        console.log(`Collapsed node: ${nodeId}`);
      } else {
        state.expandedNodes.push(nodeId);
        console.log(`Expanded node: ${nodeId}`);
      }
    },
    expandAllNodes: (state) => {
      state.expandedNodes = state.hierarchy.map(node => node.id);
    },
    collapseAllNodes: (state) => {
      state.expandedNodes = [];
    },
    addMessage: (state, action: PayloadAction<MessageDetails>) => {
      const message = action.payload;
      state.messages[message.message_id] = message;
    },
    updateMessage: (state, action: PayloadAction<{ id: string; updates: Partial<MessageDetails> }>) => {
      const { id, updates } = action.payload;
      if (state.messages[id]) {
        state.messages[id] = { ...state.messages[id], ...updates };
      }
    },
    removeMessage: (state, action: PayloadAction<string>) => {
      delete state.messages[action.payload];
    },
    addConversation: (state, action: PayloadAction<Conversation>) => {
      const conversation = action.payload;
      state.conversations[conversation.id] = conversation;
    },
    updateConversation: (state, action: PayloadAction<{ id: string; updates: Partial<Conversation> }>) => {
      const { id, updates } = action.payload;
      if (state.conversations[id]) {
        state.conversations[id] = { ...state.conversations[id], ...updates };
      }
    },
    removeConversation: (state, action: PayloadAction<string>) => {
      delete state.conversations[action.payload];
    },
    buildHierarchy: (state) => {
      // Build hierarchy from threads, conversations, and messages
      state.hierarchy = state.threads.map(thread => ({
        id: thread.topic_id,
        type: 'thread' as const,
        data: thread,
        children: [],
        expanded: state.expandedNodes.has(thread.topic_id),
      }));
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch threads
      .addCase(fetchThreads.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchThreads.fulfilled, (state, action) => {
        state.loading = false;
        state.threads = action.payload;
        
        // Rebuild hierarchy when threads are loaded
        state.hierarchy = action.payload.map((thread: any) => ({
          id: thread.topic_id,
          type: 'thread' as const,
          data: thread,
          children: thread.messages_hierarchy || [], // Use the hierarchical structure
          expanded: state.expandedNodes.includes(thread.topic_id),
        }));
      })
      .addCase(fetchThreads.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch threads';
      })
      // Fetch single thread
      .addCase(fetchThread.fulfilled, (state, action) => {
        const thread = action.payload;
        const existingIndex = state.threads.findIndex(t => t.topic_id === thread.topic_id);
        if (existingIndex >= 0) {
          state.threads[existingIndex] = thread;
        } else {
          state.threads.push(thread);
        }
      })
      // Update thread
      .addCase(updateThread.fulfilled, (state, action) => {
        const updatedThread = action.payload;
        const index = state.threads.findIndex(t => t.topic_id === updatedThread.topic_id);
        if (index >= 0) {
          state.threads[index] = updatedThread;
        }
      })
      // Hierarchy operations
      .addCase(performHierarchyOperation.fulfilled, (state, action) => {
        // Refresh threads after hierarchy operation
        // This will trigger a rebuild of the hierarchy
      });
  },
});

export const {
  setSelectedThread,
  toggleNodeExpansion,
  expandAllNodes,
  collapseAllNodes,
  addMessage,
  updateMessage,
  removeMessage,
  addConversation,
  updateConversation,
  removeConversation,
  buildHierarchy,
  clearError,
} = threadsSlice.actions;

export default threadsSlice.reducer;
