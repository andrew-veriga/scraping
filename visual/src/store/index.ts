import { configureStore } from '@reduxjs/toolkit';
import { enableMapSet } from 'immer';
import threadsReducer from './slices/threadsSlice';
import uiReducer from './slices/uiSlice';
import dragReducer from './slices/dragSlice';

// Enable Map and Set support in Immer
enableMapSet();

export const store = configureStore({
  reducer: {
    threads: threadsReducer,
    ui: uiReducer,
    drag: dragReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['drag/setDragState'],
        ignoredPaths: ['drag.dragState'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
