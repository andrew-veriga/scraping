'use client';

import React from 'react';
import ThreadPanel from '@/components/ThreadPanel';
import { useAppSelector } from '@/hooks/redux';
import { Menu, Settings, HelpCircle } from 'lucide-react';

export default function Home() {
  const { selectedThreadId } = useAppSelector((state) => state.threads);

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">
            Discord SUI Analyzer - Visual Editor
          </h1>
          {selectedThreadId && (
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Selected: {selectedThreadId}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
            <HelpCircle className="w-5 h-5" />
          </button>
          <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
            <Settings className="w-5 h-5" />
          </button>
          <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
            <Menu className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left Panel */}
        <div className="w-1/2 border-r border-gray-200 dark:border-gray-700">
          <ThreadPanel side="left" title="Source Threads" />
        </div>

        {/* Right Panel */}
        <div className="w-1/2">
          <ThreadPanel side="right" title="Target Threads" />
        </div>
      </main>

      {/* Footer */}
      <footer className="p-4 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
          <div>
            Drag and drop threads, conversations, or messages between panels to reorganize the hierarchy
          </div>
          <div>
            Use right-click for context menu options
          </div>
        </div>
      </footer>
    </div>
  );
}
