'use client';

import React, { useState } from 'react';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import { HierarchyNode as HierarchyNodeType, Thread, Conversation, MessageDetails } from '@/types';
import { ChevronRight, ChevronDown, MessageSquare, Users, FileText, MoreHorizontal } from 'lucide-react';
import { cn } from '@/utils/cn';

interface HierarchyNodeProps {
  node: HierarchyNodeType;
  level: number;
  onToggle: (nodeId: string) => void;
  onSelect: (nodeId: string) => void;
  onContextMenu?: (nodeId: string, event: React.MouseEvent) => void;
  selected?: boolean;
  dragType?: 'thread' | 'conversation' | 'message';
}

const HierarchyNode: React.FC<HierarchyNodeProps> = ({
  node,
  level,
  onToggle,
  onSelect,
  onContextMenu,
  selected = false,
  dragType,
}) => {
  const [showContextMenu, setShowContextMenu] = useState(false);

  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
    id: node.id,
    data: {
      node,
      type: node.type,
    },
    disabled: dragType === 'message' && node.type !== 'message',
  });

  const { setNodeRef: setDropRef, isOver } = useDroppable({
    id: `drop-${node.id}`,
    data: {
      node,
      type: node.type,
    },
  });

  const handleClick = () => {
    onSelect(node.id);
  };

  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    console.log(`Toggling node: ${node.id} (${node.type}), current expanded: ${isExpanded}`);
    onToggle(node.id);
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (onContextMenu) {
      onContextMenu(node.id, e);
    }
    setShowContextMenu(true);
  };

  const getNodeIcon = () => {
    switch (node.type) {
      case 'thread':
        return <FileText className="w-4 h-4" />;
      case 'conversation':
        return <Users className="w-4 h-4" />;
      case 'message':
        return <MessageSquare className="w-4 h-4" />;
      default:
        return <FileText className="w-4 h-4" />;
    }
  };

  const getNodeTitle = () => {
    switch (node.type) {
      case 'thread':
        return (node.data as Thread).header || (node.data as Thread).topic_id;
      case 'conversation':
        const conversation = node.data as Conversation;
        return `Conversation (${conversation.participants.length} participants)`;
      case 'message':
        const message = node.data as MessageDetails;
        return message.content?.substring(0, 50) + (message.content && message.content.length > 50 ? '...' : '');
      default:
        return node.id;
    }
  };

  const getNodeSubtitle = () => {
    switch (node.type) {
      case 'thread':
        const thread = node.data as Thread;
        return `${thread.label} • ${new Date(thread.actual_date).toLocaleDateString()}`;
      case 'conversation':
        const conversation = node.data as Conversation;
        return `${conversation.messages.length} messages • ${new Date(conversation.start_date).toLocaleDateString()}`;
      case 'message':
        const message = node.data as MessageDetails;
        return `Author: ${message.author_id} • ${new Date(message.datetime).toLocaleDateString()}`;
      default:
        return '';
    }
  };

  const hasChildren = node.children && node.children.length > 0;
  const isExpanded = node.expanded;
  
  // Temporary debug logging
  if (node.type === 'thread') {
    console.log(`Thread ${node.id}: hasChildren=${hasChildren}, childrenLength=${node.children?.length || 0}, children=`, node.children);
  }
  

  return (
    <div
      ref={(el) => {
        setNodeRef(el);
        setDropRef(el);
      }}
      className={cn(
        'group relative flex items-start gap-2 p-2 rounded-lg cursor-pointer transition-colors',
        'hover:bg-gray-50 dark:hover:bg-gray-800',
        selected && 'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700',
        isDragging && 'opacity-50',
        isOver && 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-700',
        level > 0 && 'ml-4'
      )}
      style={{
        paddingLeft: `${level * 16 + 8}px`,
        transform: transform ? `translate3d(${transform.x}px, ${transform.y}px, 0)` : undefined,
      }}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
      {...attributes}
      {...listeners}
    >
      {/* Expand/Collapse Button */}
      {hasChildren && (
        <button
          onClick={handleToggle}
          className="flex-shrink-0 p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
          title={`${isExpanded ? 'Collapse' : 'Expand'} thread`}
        >
          {isExpanded ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </button>
      )}
      {!hasChildren && node.type === 'thread' && (
        <div className="flex-shrink-0 p-1 w-6 h-6" title="No messages to expand">
          <div className="w-4 h-4 border border-gray-300 rounded"></div>
        </div>
      )}

      {/* Node Icon */}
      <div className="flex-shrink-0 mt-0.5">
        {getNodeIcon()}
      </div>

      {/* Node Content */}
      <div className="flex-1 min-w-0">
        <div className="font-medium text-sm truncate">
          {getNodeTitle()}
        </div>
        <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
          {getNodeSubtitle()}
        </div>
      </div>

      {/* Context Menu Button */}
      <button
        className="opacity-0 group-hover:opacity-100 flex-shrink-0 p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
        onClick={(e) => {
          e.stopPropagation();
          handleContextMenu(e);
        }}
      >
        <MoreHorizontal className="w-4 h-4" />
      </button>

      {/* Context Menu */}
      {showContextMenu && (
        <div className="absolute right-0 top-8 z-50 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg py-1 min-w-32">
          <button className="w-full px-3 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700">
            Move to Thread
          </button>
          <button className="w-full px-3 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700">
            Create New Thread
          </button>
          <button className="w-full px-3 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700">
            Collapse Branch
          </button>
          <button className="w-full px-3 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700">
            Delete
          </button>
        </div>
      )}
    </div>
  );
};

export default HierarchyNode;
