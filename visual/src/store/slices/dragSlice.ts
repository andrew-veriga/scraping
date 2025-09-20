import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { DragState, HierarchyNode } from '@/types';

const initialState: DragState = {
  isDragging: false,
  draggedItem: undefined,
  dragOverItem: undefined,
  dragType: undefined,
};

const dragSlice = createSlice({
  name: 'drag',
  initialState,
  reducers: {
    startDrag: (state, action: PayloadAction<{
      item: HierarchyNode;
      type: 'thread' | 'conversation' | 'message';
    }>) => {
      state.isDragging = true;
      state.draggedItem = action.payload.item;
      state.dragType = action.payload.type;
    },
    setDragOver: (state, action: PayloadAction<HierarchyNode | undefined>) => {
      state.dragOverItem = action.payload;
    },
    endDrag: (state) => {
      state.isDragging = false;
      state.draggedItem = undefined;
      state.dragOverItem = undefined;
      state.dragType = undefined;
    },
    clearDrag: (state) => {
      state.isDragging = false;
      state.draggedItem = undefined;
      state.dragOverItem = undefined;
      state.dragType = undefined;
    },
  },
});

export const {
  startDrag,
  setDragOver,
  endDrag,
  clearDrag,
} = dragSlice.actions;

export default dragSlice.reducer;
