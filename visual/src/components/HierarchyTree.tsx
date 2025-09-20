'use client';

import React from 'react';
import { DndContext, DragEndEvent, DragOverEvent, DragStartEvent } from '@dnd-kit/core';
import { HierarchyNode as HierarchyNodeType } from '@/types';
import HierarchyNode from './HierarchyNode';
import { useAppDispatch, useAppSelector } from '@/hooks/redux';
import { startDrag, setDragOver, endDrag } from '@/store/slices/dragSlice';
import { performHierarchyOperation } from '@/store/slices/threadsSlice';

interface HierarchyTreeProps {
  nodes: HierarchyNodeType[];
  selectedNodeId?: string;
  onNodeSelect: (nodeId: string) => void;
  onNodeToggle: (nodeId: string) => void;
  onNodeContextMenu?: (nodeId: string, event: React.MouseEvent) => void;
  className?: string;
}

const HierarchyTree: React.FC<HierarchyTreeProps> = ({
  nodes,
  selectedNodeId,
  onNodeSelect,
  onNodeToggle,
  onNodeContextMenu,
  className = '',
}) => {
  const dispatch = useAppDispatch();
  const dragState = useAppSelector((state) => state.drag);

  const handleDragStart = (event: DragStartEvent) => {
    const { active } = event;
    const node = active.data.current?.node as HierarchyNodeType;
    const type = active.data.current?.type as 'thread' | 'conversation' | 'message';
    
    if (node && type) {
      dispatch(startDrag({ item: node, type }));
    }
  };

  const handleDragOver = (event: DragOverEvent) => {
    const { over } = event;
    const node = over?.data.current?.node as HierarchyNodeType;
    
    if (node) {
      dispatch(setDragOver(node));
    } else {
      dispatch(setDragOver(undefined));
    }
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    const draggedNode = active.data.current?.node as HierarchyNodeType;
    const targetNode = over?.data.current?.node as HierarchyNodeType;
    
    dispatch(endDrag());

    if (draggedNode && targetNode && draggedNode.id !== targetNode.id) {
      handleDrop(draggedNode, targetNode);
    }
  };

  const handleDrop = async (draggedNode: HierarchyNodeType, targetNode: HierarchyNodeType) => {
    try {
      let operation;
      
      // Determine operation based on node types
      if (draggedNode.type === 'thread' && targetNode.type === 'thread') {
        operation = 'merge_threads';
      } else if (draggedNode.type === 'conversation' && targetNode.type === 'thread') {
        operation = 'move_conversation';
      } else if (draggedNode.type === 'message' && targetNode.type === 'conversation') {
        operation = 'move_message';
      } else if (draggedNode.type === 'conversation' && targetNode.type === 'conversation') {
        operation = 'merge_conversations';
      } else {
        console.warn('Invalid drop operation:', { draggedNode: draggedNode.type, targetNode: targetNode.type });
        return;
      }

      await dispatch(performHierarchyOperation({
        operation,
        source_id: draggedNode.id,
        target_id: targetNode.id,
        data: {
          draggedNode,
          targetNode,
        },
      })).unwrap();

      // Refresh the hierarchy after successful operation
      // This will be handled by the Redux store
    } catch (error) {
      console.error('Failed to perform hierarchy operation:', error);
    }
  };

  const renderNode = (node: HierarchyNodeType, level: number = 0): React.ReactNode => {
    return (
      <div key={node.id}>
        <HierarchyNode
          node={node}
          level={level}
          onToggle={onNodeToggle}
          onSelect={onNodeSelect}
          onContextMenu={onNodeContextMenu}
          selected={selectedNodeId === node.id}
          dragType={dragState?.dragType || undefined}
        />
        {node.expanded && node.children && node.children.length > 0 && (
          <div>
            {node.children.map((child) => renderNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <DndContext
      onDragStart={handleDragStart}
      onDragOver={handleDragOver}
      onDragEnd={handleDragEnd}
    >
      <div className={`space-y-1 ${className}`}>
        {nodes.map((node) => renderNode(node))}
      </div>
    </DndContext>
  );
};

export default HierarchyTree;
