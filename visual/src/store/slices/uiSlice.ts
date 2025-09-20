import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { SolutionStatus } from '@/types';

interface UIState {
  leftPanel: {
    searchQuery: string;
    filterStatus?: SolutionStatus;
    filterTechnical?: boolean;
    collapsed: boolean;
  };
  rightPanel: {
    searchQuery: string;
    filterStatus?: SolutionStatus;
    filterTechnical?: boolean;
    collapsed: boolean;
  };
  sidebar: {
    collapsed: boolean;
    width: number;
  };
  theme: 'light' | 'dark';
  loading: boolean;
  notifications: Array<{
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    message: string;
    timestamp: number;
  }>;
}

const initialState: UIState = {
  leftPanel: {
    searchQuery: '',
    collapsed: false,
  },
  rightPanel: {
    searchQuery: '',
    collapsed: false,
  },
  sidebar: {
    collapsed: false,
    width: 300,
  },
  theme: 'light',
  loading: false,
  notifications: [],
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setLeftPanelSearch: (state, action: PayloadAction<string>) => {
      state.leftPanel.searchQuery = action.payload;
    },
    setRightPanelSearch: (state, action: PayloadAction<string>) => {
      state.rightPanel.searchQuery = action.payload;
    },
    setLeftPanelFilter: (state, action: PayloadAction<{ status?: SolutionStatus; technical?: boolean }>) => {
      state.leftPanel.filterStatus = action.payload.status;
      state.leftPanel.filterTechnical = action.payload.technical;
    },
    setRightPanelFilter: (state, action: PayloadAction<{ status?: SolutionStatus; technical?: boolean }>) => {
      state.rightPanel.filterStatus = action.payload.status;
      state.rightPanel.filterTechnical = action.payload.technical;
    },
    toggleLeftPanel: (state) => {
      state.leftPanel.collapsed = !state.leftPanel.collapsed;
    },
    toggleRightPanel: (state) => {
      state.rightPanel.collapsed = !state.rightPanel.collapsed;
    },
    toggleSidebar: (state) => {
      state.sidebar.collapsed = !state.sidebar.collapsed;
    },
    setSidebarWidth: (state, action: PayloadAction<number>) => {
      state.sidebar.width = action.payload;
    },
    setTheme: (state, action: PayloadAction<'light' | 'dark'>) => {
      state.theme = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    addNotification: (state, action: PayloadAction<{
      type: 'success' | 'error' | 'warning' | 'info';
      message: string;
    }>) => {
      const notification = {
        id: Date.now().toString(),
        timestamp: Date.now(),
        ...action.payload,
      };
      state.notifications.push(notification);
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
  },
});

export const {
  setLeftPanelSearch,
  setRightPanelSearch,
  setLeftPanelFilter,
  setRightPanelFilter,
  toggleLeftPanel,
  toggleRightPanel,
  toggleSidebar,
  setSidebarWidth,
  setTheme,
  setLoading,
  addNotification,
  removeNotification,
  clearNotifications,
} = uiSlice.actions;

export default uiSlice.reducer;
