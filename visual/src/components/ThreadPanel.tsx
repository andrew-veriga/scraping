'use client';

import React, { useState, useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '@/hooks/redux';
import { fetchThreads, toggleNodeExpansion, setSelectedThread } from '@/store/slices/threadsSlice';
import { setLeftPanelSearch, setLeftPanelFilter } from '@/store/slices/uiSlice';
import HierarchyTree from './HierarchyTree';
import { Search, Filter, ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react';
import { SolutionStatus } from '@/types';

interface ThreadPanelProps {
  side: 'left' | 'right';
  title: string;
}

const ThreadPanel: React.FC<ThreadPanelProps> = ({ side, title }) => {
  const dispatch = useAppDispatch();
  const { threads, hierarchy, loading, error, selectedThreadId } = useAppSelector((state) => state.threads);
  const { leftPanel, rightPanel } = useAppSelector((state) => state.ui);
  
  const panelState = side === 'left' ? leftPanel : rightPanel;
  const [localSearch, setLocalSearch] = useState(panelState.searchQuery);
  const [showFilters, setShowFilters] = useState(false);

  useEffect(() => {
    // Load threads on component mount
    dispatch(fetchThreads());
  }, [dispatch]);

  const handleSearchChange = (value: string) => {
    setLocalSearch(value);
    if (side === 'left') {
      dispatch(setLeftPanelSearch(value));
    } else {
      dispatch(setRightPanelSearch(value));
    }
  };

  const handleFilterChange = (filters: { status?: SolutionStatus; technical?: boolean }) => {
    if (side === 'left') {
      dispatch(setLeftPanelFilter(filters));
    } else {
      dispatch(setRightPanelFilter(filters));
    }
  };

  const handleNodeSelect = (nodeId: string) => {
    dispatch(setSelectedThread(nodeId));
  };

  const handleNodeToggle = (nodeId: string) => {
    console.log(`ThreadPanel: handleNodeToggle called for node: ${nodeId}`);
    dispatch(toggleNodeExpansion(nodeId));
  };

  const handleRefresh = () => {
    dispatch(fetchThreads());
  };

  const filteredHierarchy = hierarchy.filter((node) => {
    // Apply search filter
    if (localSearch) {
      const searchLower = localSearch.toLowerCase();
      const thread = node.data as any;
      const matchesSearch = 
        thread.header?.toLowerCase().includes(searchLower) ||
        thread.topic_id?.toLowerCase().includes(searchLower) ||
        thread.solution?.toLowerCase().includes(searchLower);
      
      if (!matchesSearch) return false;
    }

    // Apply status filter
    if (panelState.filterStatus) {
      const thread = node.data as any;
      if (thread.label !== panelState.filterStatus) return false;
    }

    // Apply technical filter
    if (panelState.filterTechnical !== undefined) {
      const thread = node.data as any;
      if (thread.is_technical !== panelState.filterTechnical) return false;
    }

    return true;
  });


  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">{title}</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={handleRefresh}
            disabled={loading}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <Filter className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            id={`search-${side}`}
            name={`search-${side}`}
            type="text"
            placeholder="Search threads..."
            value={localSearch}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
      </div>

      {/* Filters */}
      {showFilters && (
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Status
              </label>
              <select
                id={`status-filter-${side}`}
                name={`status-filter-${side}`}
                value={panelState.filterStatus || ''}
                onChange={(e) => handleFilterChange({ 
                  status: e.target.value as SolutionStatus || undefined,
                  technical: panelState.filterTechnical 
                })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
              >
                <option value="">All Statuses</option>
                <option value={SolutionStatus.RESOLVED}>Resolved</option>
                <option value={SolutionStatus.UNRESOLVED}>Unresolved</option>
                <option value={SolutionStatus.SUGGESTION}>Suggestion</option>
                <option value={SolutionStatus.OUTSIDE}>Outside</option>
              </select>
            </div>
            <div>
              <label className="flex items-center">
                <input
                  id={`technical-filter-${side}`}
                  name={`technical-filter-${side}`}
                  type="checkbox"
                  checked={panelState.filterTechnical || false}
                  onChange={(e) => handleFilterChange({ 
                    status: panelState.filterStatus,
                    technical: e.target.checked 
                  })}
                  className="mr-2"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">Technical only</span>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Thread List */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center h-32">
            <RefreshCw className="w-6 h-6 animate-spin text-gray-400" />
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-32 text-red-500 dark:text-red-400">
            <div className="text-center">
              <div className="font-medium">Error loading threads</div>
              <div className="text-sm mt-1">{error}</div>
            </div>
          </div>
        ) : filteredHierarchy.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-gray-500 dark:text-gray-400">
            No threads found
          </div>
        ) : (
          <HierarchyTree
            nodes={filteredHierarchy}
            selectedNodeId={selectedThreadId}
            onNodeSelect={handleNodeSelect}
            onNodeToggle={handleNodeToggle}
          />
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
        <div className="text-sm text-gray-500 dark:text-gray-400">
          {filteredHierarchy.length} thread{filteredHierarchy.length !== 1 ? 's' : ''}
        </div>
      </div>
    </div>
  );
};

export default ThreadPanel;
