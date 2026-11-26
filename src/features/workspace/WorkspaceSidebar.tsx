import React, { useState, useEffect, useRef } from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  CheckSquare,
  CalendarDays,
  BarChart3,
  Settings,
  Plus,
  LogOut,
  Users,
  ChevronDown,
  Clock,
  AlertCircle,
  Calendar,
  CheckCircle2,
  FlaskConical,
  MessageSquare,
  ArrowLeft,
  Pin,
  PinOff,
  Trash2,
} from 'lucide-react';
import { Button } from '../../components/ui';
import { Team } from '../teams/types';
import {
  createThread,
  deleteThread,
  getActiveChatId,
  readThreads,
  selectThread,
  subscribeThreads,
  toggleThreadPinned,
} from '../chat-lab/chatStore';

export const DEFAULT_SIDEBAR_WIDTH = 256;
export const MIN_SIDEBAR_WIDTH = 240;
export const MAX_SIDEBAR_WIDTH = 480;
export const SIDEBAR_WIDTH_STORAGE_KEY = 'vido_sidebar_width';

export interface WorkspaceSidebarProps {
  isPreview?: boolean;
  basePath?: string;
  user?: {
    id?: string;
    name: string;
    email?: string;
  };
  counts: {
    total?: number;
    today: number;
    needsAttention: number;
    upcoming: number;
    completed: number;
    archived: number;
  };
  teams?: Team[];
  sidebarWidth?: number;
  onSidebarWidthChange?: (width: number) => void;
  onNewTaskClick?: () => void;
  onCreateTeamClick?: () => void;
  onOpenSettings?: () => void;
  onSignOut?: () => void;
}

export const WorkspaceSidebar: React.FC<WorkspaceSidebarProps> = ({
  isPreview = true,
  basePath = '/preview',
  user,
  counts,
  teams = [],
  sidebarWidth: controlledWidth,
  onSidebarWidthChange,
  onNewTaskClick,
  onCreateTeamClick,
  onOpenSettings,
  onSignOut,
}) => {
  const location = useLocation();
  const chatUserId = user?.id || 'demo-user-uid';
  const isChatRoute =
    location.pathname === `${basePath}/converse` ||
    location.pathname === `${basePath}/chat-zone` ||
    location.pathname === `${basePath}/chat-lab`;
  const [chatThreads, setChatThreads] = useState(() => readThreads(chatUserId));

  useEffect(() => {
    const refresh = () => setChatThreads(readThreads(chatUserId));
    refresh();
    return subscribeThreads(chatUserId, refresh);
  }, [chatUserId]);

  // Resizable width state with localStorage persistence
  const [internalWidth, setInternalWidth] = useState<number>(() => {
    try {
      const saved = localStorage.getItem(SIDEBAR_WIDTH_STORAGE_KEY);
      if (saved) {
        const parsed = Number(saved);
        if (!isNaN(parsed) && parsed >= MIN_SIDEBAR_WIDTH && parsed <= MAX_SIDEBAR_WIDTH) {
          return parsed;
        }
      }
    } catch {
      // ignore
    }
    return DEFAULT_SIDEBAR_WIDTH;
  });

  const width = controlledWidth ?? internalWidth;
  const widthRef = useRef(width);
  widthRef.current = width;
  const [isResizing, setIsResizing] = useState(false);

  const updateWidth = (newWidth: number) => {
    const clamped = Math.min(Math.max(newWidth, MIN_SIDEBAR_WIDTH), MAX_SIDEBAR_WIDTH);
    setInternalWidth(clamped);
    onSidebarWidthChange?.(clamped);
    try {
      localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, String(clamped));
    } catch {
      // ignore
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = Math.min(Math.max(e.clientX, MIN_SIDEBAR_WIDTH), MAX_SIDEBAR_WIDTH);
      setInternalWidth(newWidth);
      onSidebarWidthChange?.(newWidth);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      try {
        localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, String(widthRef.current));
      } catch {
        // ignore
      }
    };

    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, onSidebarWidthChange]);

  const handleDoubleClick = () => updateWidth(DEFAULT_SIDEBAR_WIDTH);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      updateWidth(width - 16);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      updateWidth(width + 16);
    } else if (e.key === 'Home') {
      e.preventDefault();
      updateWidth(MIN_SIDEBAR_WIDTH);
    } else if (e.key === 'End') {
      e.preventDefault();
      updateWidth(MAX_SIDEBAR_WIDTH);
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      updateWidth(DEFAULT_SIDEBAR_WIDTH);
    }
  };

  const totalTasks =
    counts.total ?? counts.today + counts.upcoming + counts.needsAttention;

  const dashboardPath = basePath;
  const tasksPath = `${basePath}/tasks`;
  const calendarPath = `${basePath}/calendar`;
  const analyticsPath = `${basePath}/analytics`;
  const settingsPath = `${basePath}/settings`;
  const mlLabPath = `${basePath}/ml-lab`;
  const conversePath = `${basePath}/converse`;

  const isTaskRoute =
    location.pathname === tasksPath ||
    location.pathname.startsWith(`${tasksPath}/`) ||
    location.pathname === `${basePath}/today` ||
    location.pathname === `${basePath}/upcoming` ||
    location.pathname === `${basePath}/needs-attention` ||
    location.pathname === `${basePath}/completed` ||
    location.pathname === `${basePath}/archived`;

  const [isTasksOpen, setIsTasksOpen] = useState(isTaskRoute);

  useEffect(() => {
    if (isTaskRoute) {
      setIsTasksOpen(true);
    }
  }, [isTaskRoute]);

  const taskSubItems = [
    {
      label: 'All Tasks',
      path: tasksPath,
      icon: CheckSquare,
      iconColor: 'text-emerald-400',
      count: totalTasks,
    },
    {
      label: "Today's Tasks",
      path: `${basePath}/today`,
      icon: Clock,
      iconColor: 'text-[var(--accent)]',
      count: counts.today,
    },
    {
      label: 'Upcoming',
      path: `${basePath}/upcoming`,
      icon: Calendar,
      iconColor: 'text-indigo-400',
      count: counts.upcoming,
    },
    {
      label: 'Needs Attention',
      path: `${basePath}/needs-attention`,
      icon: AlertCircle,
      iconColor: 'text-amber-400',
      count: counts.needsAttention,
    },
    {
      label: 'Completed',
      path: `${basePath}/completed`,
      icon: CheckCircle2,
      iconColor: 'text-emerald-400',
      count: counts.completed,
    },
  ];

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `vdb-rail-link${isActive ? ' vdb-rail-link-active' : ''}`;

  // Converse gets its own sidebar: conversation history with a way back.
  if (isChatRoute) {
    const sortedChatThreads = [...chatThreads].sort(
      (a, b) => Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)) || b.updatedAt - a.updatedAt
    );

    return (
      <aside
        aria-label="Chat history"
        className={`vdb-chat-rail flex flex-col shrink-0 sticky ${
          isPreview
            ? 'top-[var(--preview-bar-height)] h-[calc(100vh-var(--preview-bar-height))]'
            : 'top-0 h-screen'
        } select-none`}
      >
        <div className="vdb-chat-brand">
          <div className="vdb-chat-brand-mark" aria-hidden="true">
            <MessageSquare className="w-4 h-4" strokeWidth={1.6} />
          </div>
          <div className="min-w-0">
            <div className="vdb-chat-brand-name">Converse</div>
            <span className="vdb-chat-brand-sub">Conversations</span>
          </div>
        </div>

        <div className="vdb-chat-section">
          <span className="vdb-chat-section-label">Chat history</span>
          <button
            type="button"
            aria-label="Start new chat"
            onClick={() => createThread(chatUserId)}
            className="vdb-chat-new"
          >
            <Plus className="w-3.5 h-3.5" />
          </button>
        </div>
        <div className="vdb-chat-rule" aria-hidden="true" />

        <nav className="vdb-chat-thread-list" aria-label="Chat threads">
          {sortedChatThreads.map((thread, index) => {
            const activeThread = thread.id === getActiveChatId(chatUserId);
            return (
              <div
                key={thread.id}
                className={`vdb-chat-thread ${
                  activeThread ? 'vdb-chat-thread-active' : ''
                }`}
              >
                <button
                  type="button"
                  onClick={() => selectThread(chatUserId, thread.id)}
                  className="vdb-chat-thread-main"
                  title={thread.pinned ? `${thread.title} (pinned)` : thread.title}
                >
                  <span className="vdb-chat-thread-index" aria-hidden="true">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  {thread.pinned && <Pin className="vdb-chat-pin-mark w-3 h-3 shrink-0" />}
                  <span className="truncate">{thread.title}</span>
                </button>
                <div className="vdb-chat-thread-actions">
                  <button
                    type="button"
                    aria-label={thread.pinned ? `Unpin ${thread.title}` : `Pin ${thread.title}`}
                    onClick={() => toggleThreadPinned(chatUserId, thread.id)}
                    className={`vdb-chat-action ${
                      thread.pinned ? 'vdb-chat-action-active' : ''
                    }`}
                  >
                    {thread.pinned ? (
                      <PinOff className="w-3.5 h-3.5" />
                    ) : (
                      <Pin className="w-3.5 h-3.5" />
                    )}
                  </button>
                  <button
                    type="button"
                    aria-label={`Delete ${thread.title}`}
                    onClick={() => deleteThread(chatUserId, thread.id)}
                    className="vdb-chat-action vdb-chat-action-danger"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })}
        </nav>

        <div className="vdb-chat-footnote" aria-hidden="true">
          <span>Personal index</span>
          <span>ViDo / 2026</span>
        </div>

        <Link to={basePath} className="vdb-chat-back">
          <ArrowLeft className="w-4 h-4" />
          <span>Back to application</span>
        </Link>
      </aside>
    );
  }

  return (
    <aside
      aria-label="Workspace navigation"
      style={{
        width: `${width}px`,
        minWidth: `${MIN_SIDEBAR_WIDTH}px`,
        maxWidth: `${MAX_SIDEBAR_WIDTH}px`,
      }}
      className={`vdb-workspace-rail relative flex flex-col justify-between shrink-0 sticky border-r ${
        isPreview
          ? 'top-[var(--preview-bar-height)] h-[calc(100vh-var(--preview-bar-height))]'
          : 'top-0 h-screen'
      } workspace-sidebar-scrollbar select-none overflow-hidden ${
        isResizing ? 'transition-none' : 'transition-[width] duration-75'
      }`}
    >
      {/* 1. Fixed Top Header: Brand & New Task */}
      <div className="vdb-rail-top shrink-0">
        {/* Brand Header */}
        <div className="vdb-rail-brand">
          <div className="vdb-rail-brand-mark" aria-hidden="true">
            V
          </div>
          <div className="min-w-0">
            <div className="vdb-rail-brand-heading">
              <span className="vdb-rail-brand-name">ViDo</span>
              <span className="vdb-rail-badge">{isPreview ? 'DEMO' : 'WORKSPACE'}</span>
            </div>
            <span className="vdb-rail-brand-sub">
              {isPreview ? 'Logged-in demo preview' : user?.name || 'Personal Workspace'}
            </span>
          </div>
        </div>

        {/* Quick New Task Button */}
        {onNewTaskClick && (
          <div className="vdb-rail-new-wrap">
            <Button
              variant="primary"
              size="sm"
              onClick={onNewTaskClick}
              className="vdb-rail-new"
            >
              <Plus className="w-4 h-4" /> New Task
            </Button>
          </div>
        )}
      </div>

      {/* 2. Scrollable Middle Navigation Area */}
      <div className="vdb-rail-scroll workspace-sidebar-scrollbar">
        <div className="vdb-rail-section-label">
          Workspace
        </div>
        <nav className="vdb-rail-nav">
          <NavLink to={dashboardPath} end className={linkClass}>
            <div className="flex items-center gap-2.5">
              <LayoutDashboard className="w-4 h-4 text-[var(--accent)]" />
              <span>Dashboard</span>
            </div>
          </NavLink>

          <div>
            <NavLink
              to={tasksPath}
              onClick={() => setIsTasksOpen(true)}
              className={() =>
                `vdb-rail-link${isTaskRoute ? ' vdb-rail-link-active' : ''}`
              }
            >
              <div className="flex items-center gap-2.5">
                <CheckSquare className="w-4 h-4 text-emerald-400" />
                <span>Tasks</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="vdb-rail-count">{totalTasks}</span>
                <button
                  type="button"
                  aria-label={isTasksOpen ? 'Collapse task views' : 'Expand task views'}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setIsTasksOpen((prev) => !prev);
                  }}
                  className="vdb-rail-toggle"
                >
                  <ChevronDown
                    className={`w-3.5 h-3.5 transition-transform duration-200 ${
                      isTasksOpen ? 'rotate-0' : '-rotate-90'
                    }`}
                  />
                </button>
              </div>
            </NavLink>

            {isTasksOpen && (
              <div className="vdb-rail-subnav">
                {taskSubItems.map((item) => (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    end
                    className={({ isActive }) =>
                      `vdb-rail-sublink${isActive ? ' vdb-rail-sublink-active' : ''}`
                    }
                  >
                    <div className="flex items-center gap-2 truncate">
                      <item.icon className={`w-3.5 h-3.5 shrink-0 ${item.iconColor}`} />
                      <span className="truncate">{item.label}</span>
                    </div>
                    <span className="vdb-rail-count-muted">
                      {item.count}
                    </span>
                  </NavLink>
                ))}
              </div>
            )}
          </div>
        </nav>

        <div className="vdb-rail-section-label">
          Planning
        </div>
        <nav className="vdb-rail-nav">
          <NavLink to={calendarPath} className={linkClass}>
            <div className="flex items-center gap-2.5">
              <CalendarDays className="w-4 h-4 text-indigo-400" />
              <span>Calendar</span>
            </div>
          </NavLink>

          <NavLink to={analyticsPath} className={linkClass}>
            <div className="flex items-center gap-2.5">
              <BarChart3 className="w-4 h-4 text-emerald-400" />
              <span>Analytics</span>
            </div>
          </NavLink>

          <NavLink to={mlLabPath} className={linkClass}>
            <div className="flex items-center gap-2.5">
              <FlaskConical className="w-4 h-4 text-purple-400" />
              <span>ML Lab</span>
            </div>
          </NavLink>

          <NavLink to={conversePath} className={linkClass}>
            <div className="flex items-center gap-2.5">
              <MessageSquare className="w-4 h-4 text-amber-300" />
              <span>Converse</span>
            </div>
          </NavLink>
        </nav>

        {/* Teams & Shared Projects Section */}
        <div className="vdb-rail-team-section">
          <div className="vdb-rail-section-label">
            Teams & Shared Projects
          </div>

          <div className="vdb-rail-nav">
            {teams && teams.length > 0 ? (
              teams.map((team) => (
                <NavLink
                  key={team.id}
                  to={`${basePath}/team/${team.id}`}
                  className={({ isActive }) =>
                    `vdb-rail-link${isActive ? ' vdb-rail-link-active' : ''}`
                  }
                >
                  <div className="flex items-center gap-2.5 truncate">
                    <Users className="w-4 h-4 text-sky-400 shrink-0" />
                    <span className="truncate">{team.name}</span>
                  </div>
                  {team.leadId === (user?.id || 'demo-user-1') && (
                    <span className="vdb-rail-lead">LEAD</span>
                  )}
                </NavLink>
              ))
            ) : (
              <div className="vdb-rail-muted">No teams yet</div>
            )}

            {onCreateTeamClick && (
              <button
                type="button"
                onClick={onCreateTeamClick}
                className="vdb-rail-link vdb-rail-button"
              >
                <Plus className="w-3.5 h-3.5 text-sky-400" />
                <span>Create Team</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* 3. Fixed Bottom Footer: Profile, Settings, Sign Out */}
      <div className="vdb-rail-footer shrink-0">
        {!isPreview && (
          <div className="vdb-rail-profile">
            <div className="flex items-center gap-2 min-w-0">
              <div className="vdb-rail-avatar">
                {(user?.name || 'Alex Morgan')
                  .split(' ')
                  .map((n) => n[0])
                  .join('')
                  .slice(0, 2)
                  .toUpperCase()}
              </div>
              <div className="min-w-0">
                <span className="vdb-rail-profile-name">
                  {user?.name || 'Alex Morgan'}
                </span>
                <span className="vdb-rail-profile-email">
                  {user?.email || 'alex.morgan@gmail.com'}
                </span>
              </div>
            </div>
          </div>
        )}

        <NavLink
          to={settingsPath}
          onClick={(e) => {
            e.preventDefault();
            onOpenSettings?.();
          }}
          className={({ isActive }) =>
            `vdb-rail-footer-link${isActive ? ' vdb-rail-footer-link-active' : ''}`
          }
        >
          <Settings className="w-4 h-4" />
          <span>Settings</span>
        </NavLink>

        {onSignOut && (
          <button
            type="button"
            onClick={onSignOut}
            className="vdb-rail-footer-link vdb-rail-signout"
          >
            <LogOut className="w-4 h-4" />
            <span>Sign Out</span>
          </button>
        )}
      </div>

      {/* 4. Resizable Right Gutter Handle */}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize sidebar"
        aria-valuenow={width}
        aria-valuemin={MIN_SIDEBAR_WIDTH}
        aria-valuemax={MAX_SIDEBAR_WIDTH}
        tabIndex={0}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
        onKeyDown={handleKeyDown}
        title="Drag to resize sidebar (Double-click to reset)"
        className={`vdb-rail-resize ${
          isResizing ? 'vdb-rail-resize-active' : ''
        }`}
      >
        <div className="vdb-rail-resize-grip" />
      </div>
    </aside>
  );
};
