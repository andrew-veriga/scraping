// TypeScript типы на основе Pydantic моделей

export enum ThreadStatus {
  NEW = 'new',
  MODIFIED = 'modified',
  PERSISTED = 'persisted'
}

export enum SolutionStatus {
  RESOLVED = 'resolved',
  UNRESOLVED = 'unresolved',
  SUGGESTION = 'suggestion',
  OUTSIDE = 'outside'
}

export enum RevisedStatus {
  IMPROVED = 'improved',
  CHANGED = 'changed',
  MINORCHANGES = 'minor'
}

// Базовые типы сообщений
export interface MessageNode {
  message_id: string;
  parent_id?: string;
  content?: string;
  author_id?: string;
  datetime?: string;
  referenced_message_id?: string;
}

// Расширенная информация о сообщении для UI
export interface MessageDetails extends MessageNode {
  content: string;
  author_id: string;
  datetime: string;
  referenced_message_id?: string;
  thread_id?: string;
  is_root?: boolean;
  is_answer?: boolean;
  children?: MessageNode[];
  depth?: number;
}

// Узел иерархии для отображения
export interface HierarchyNode {
  id: string;
  type: 'thread' | 'conversation' | 'message';
  data: Thread | Conversation | MessageDetails;
  children: HierarchyNode[];
  parent?: string;
  expanded?: boolean;
  selected?: boolean;
}

// Беседа (группа сообщений в рамках thread)
export interface Conversation {
  id: string;
  thread_id: string;
  root_message_id: string;
  participants: string[];
  messages: MessageDetails[];
  start_date: string;
  end_date: string;
  is_technical: boolean;
  depth: number;
}

// Thread (основной объект)
export interface Thread {
  topic_id: string;
  header: string;
  actual_date: string;
  answer_id?: string;
  label: SolutionStatus;
  solution: string;
  status?: ThreadStatus;
  is_technical?: boolean;
  is_processed?: boolean;
  conversations?: Conversation[];
  messages?: MessageDetails[];
  created_at?: string;
  updated_at?: string;
}

// Raw Thread (для API)
export interface RawThread {
  topic_id: string;
  whole_thread: MessageNode[];
  status?: ThreadStatus;
}

// Списки для API
export interface RawThreadList {
  threads: RawThread[];
}

export interface ModifiedRawThreadList {
  threads: ModifiedRawThread[];
}

export interface ModifiedRawThread extends RawThread {
  status: ThreadStatus;
}

export interface ThreadList {
  threads: Thread[];
}

export interface TechnicalTopics {
  technical_topics: string[];
}

export interface RevisedSolution {
  topic_id: string;
  label: RevisedStatus;
}

export interface RevisedList {
  comparisions: RevisedSolution[];
}

// UI состояния
export interface DragState {
  isDragging: boolean;
  draggedItem?: HierarchyNode;
  dragOverItem?: HierarchyNode;
  dragType?: 'thread' | 'conversation' | 'message';
}

export interface PanelState {
  threads: Thread[];
  selectedThread?: string;
  expandedNodes: Set<string>;
  searchQuery: string;
  filterStatus?: SolutionStatus;
  filterTechnical?: boolean;
}

export interface AppState {
  leftPanel: PanelState;
  rightPanel: PanelState;
  dragState: DragState;
  loading: boolean;
  error?: string;
}

// API типы
export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
}

export interface ThreadUpdateRequest {
  topic_id: string;
  updates: Partial<Thread>;
}

export interface MoveRequest {
  source_thread_id: string;
  target_thread_id: string;
  source_node_id: string;
  target_node_id?: string;
  operation: 'move' | 'copy' | 'merge';
}

export interface CreateThreadRequest {
  header: string;
  messages: MessageNode[];
  label: SolutionStatus;
  solution?: string;
}

// Операции с иерархией
export type HierarchyOperation = 
  | 'move_conversation'
  | 'move_message'
  | 'merge_threads'
  | 'split_conversation'
  | 'create_thread'
  | 'delete_node'
  | 'collapse_branch'
  | 'expand_branch';

export interface HierarchyOperationRequest {
  operation: HierarchyOperation;
  source_id: string;
  target_id?: string;
  data?: any;
}
