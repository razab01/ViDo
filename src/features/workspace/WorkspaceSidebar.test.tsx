import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import {
  SIDEBAR_WIDTH_STORAGE_KEY,
  WorkspaceSidebar,
} from './WorkspaceSidebar';
import {
  addMessage,
  createThread,
  resetChatStore,
  setThreadPinned,
} from '../chat-lab/chatStore';

describe('WorkspaceSidebar', () => {
  beforeEach(() => {
    localStorage.clear();
    resetChatStore();
  });

  const mockCounts = {
    today: 6,
    needsAttention: 4,
    upcoming: 5,
    completed: 5,
    archived: 2,
  };

  it('renders dashboard, tasks, and planning navigation items', () => {
    render(
      <MemoryRouter initialEntries={['/preview']}>
        <WorkspaceSidebar counts={mockCounts} />
      </MemoryRouter>
    );

    expect(screen.getByLabelText('Workspace navigation')).toHaveClass('vdb-workspace-rail');
    expect(screen.getByText('DEMO')).toBeInTheDocument();
    expect(screen.getByText('Logged-in demo preview')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /dashboard/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /tasks 15/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /calendar/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /analytics/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /ml lab/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /converse/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /settings/i })).toHaveAttribute(
      'href',
      '/preview/settings'
    );
  });

  it('keeps the GitHub-style three-zone layout and persisted resized width', () => {
    localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, '320');

    render(
      <MemoryRouter initialEntries={['/preview']}>
        <WorkspaceSidebar counts={mockCounts} />
      </MemoryRouter>
    );

    const rail = screen.getByLabelText('Workspace navigation');
    expect(rail).toHaveClass('workspace-sidebar-scrollbar');
    expect(rail).toHaveStyle({ width: '320px' });
    expect(rail.querySelector('.vdb-rail-top')).not.toBeNull();
    expect(rail.querySelector('.vdb-rail-scroll')).not.toBeNull();
    expect(rail.querySelector('.vdb-rail-footer')).not.toBeNull();

    const separator = screen.getByRole('separator', { name: /resize sidebar/i });
    expect(separator).toHaveAttribute('aria-valuenow', '320');
    fireEvent.keyDown(separator, { key: 'ArrowRight' });
    expect(localStorage.getItem(SIDEBAR_WIDTH_STORAGE_KEY)).toBe('336');
  });

  it('calls the settings callback from the fixed footer', () => {
    const handleOpenSettings = vi.fn();

    render(
      <MemoryRouter initialEntries={['/preview']}>
        <WorkspaceSidebar counts={mockCounts} onOpenSettings={handleOpenSettings} />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByRole('link', { name: /settings/i }));
    expect(handleOpenSettings).toHaveBeenCalledTimes(1);
  });

  it('keeps the Converse link available while working in ML Lab', () => {
    render(
      <MemoryRouter initialEntries={['/preview/ml-lab']}>
        <WorkspaceSidebar counts={mockCounts} />
      </MemoryRouter>
    );

    expect(screen.getByRole('link', { name: /ml lab/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /converse/i })).toHaveAttribute(
      'href',
      '/preview/converse'
    );
  });

  it('triggers new task callback when button is clicked', () => {
    const handleNewTask = vi.fn();
    render(
      <MemoryRouter initialEntries={['/preview']}>
        <WorkspaceSidebar counts={mockCounts} onNewTaskClick={handleNewTask} />
      </MemoryRouter>
    );

    const btn = screen.getByRole('button', { name: /new task/i });
    fireEvent.click(btn);
    expect(handleNewTask).toHaveBeenCalledTimes(1);
  });

  it('renders teams and triggers create team callback', () => {
    const handleCreateTeam = vi.fn();
    const mockTeams = [
      {
        id: 'team-alpha',
        name: 'Alpha Launch',
        leadId: 'demo-user-1',
        leadEmail: 'demo@vido.app',
        createdAt: '2026-11-01T00:00:00Z',
        members: [],
      },
      {
        id: 'team-beta',
        name: 'Beta Platform',
        leadId: 'other-user',
        leadEmail: 'other@vido.app',
        createdAt: '2026-11-02T00:00:00Z',
        members: [],
      },
    ];

    render(
      <MemoryRouter initialEntries={['/preview']}>
        <WorkspaceSidebar
          counts={mockCounts}
          user={{ id: 'demo-user-1', name: 'Demo User', email: 'demo@vido.app' }}
          teams={mockTeams}
          onCreateTeamClick={handleCreateTeam}
        />
      </MemoryRouter>
    );

    expect(screen.getByText('Teams & Shared Projects')).toBeInTheDocument();
    expect(screen.queryByLabelText(/create team workspace/i)).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: /alpha launch/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /beta platform/i })).toBeInTheDocument();
    expect(screen.getByText('LEAD')).toBeInTheDocument();

    const createBtn = screen.getByRole('button', { name: /^create team$/i });
    fireEvent.click(createBtn);
    expect(handleCreateTeam).toHaveBeenCalled();
  });

  it('expands task sub-navigation when on tasks route and shows section options', () => {
    render(
      <MemoryRouter initialEntries={['/preview/tasks']}>
        <WorkspaceSidebar counts={mockCounts} />
      </MemoryRouter>
    );

    // Sub-navigation items are rendered with their badges
    expect(screen.getByRole('link', { name: /all tasks 15/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /today's tasks 6/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /upcoming 5/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /needs attention 4/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /completed 5/i })).toBeInTheDocument();

    // Toggle collapse via chevron button
    const collapseBtn = screen.getByRole('button', { name: /collapse task views/i });
    fireEvent.click(collapseBtn);

    expect(screen.queryByRole('link', { name: /all tasks 15/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /today's tasks 6/i })).not.toBeInTheDocument();

    // Toggle expand again
    const expandBtn = screen.getByRole('button', { name: /expand task views/i });
    fireEvent.click(expandBtn);

    expect(screen.getByRole('link', { name: /all tasks 15/i })).toBeInTheDocument();
  });

  it('replaces workspace navigation with chat history on the chat route', () => {
    const user = { id: 'sidebar-user', name: 'Demo User', email: 'demo@vido.app' };
    const first = createThread(user.id);
    addMessage(user.id, first.id, { id: 'u1', role: 'user', text: 'who is adam smith?' });
    const second = createThread(user.id);
    addMessage(user.id, second.id, { id: 'u2', role: 'user', text: 'what is 2+2' });

    render(
      <MemoryRouter initialEntries={['/preview/chat-lab']}>
        <WorkspaceSidebar counts={mockCounts} user={user} />
      </MemoryRouter>
    );

    expect(screen.getByLabelText('Chat history')).toBeInTheDocument();
    expect(screen.getByText('Converse')).toBeInTheDocument();
    expect(screen.getByText('Conversations')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'what is 2+2' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'who is adam smith?' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /back to application/i })).toHaveAttribute(
      'href',
      '/preview'
    );
    expect(screen.queryByText('DEMO')).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /ml lab/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /delete who is adam smith\?/i }));
    expect(screen.queryByRole('button', { name: 'who is adam smith?' })).not.toBeInTheDocument();
  });

  it('does not show an empty-state message when the chat history is empty', () => {
    const user = { id: 'empty-sidebar-user', name: 'Demo User', email: 'empty@vido.app' };
    render(
      <MemoryRouter initialEntries={['/preview/converse']}>
        <WorkspaceSidebar counts={mockCounts} user={user} />
      </MemoryRouter>
    );

    expect(screen.getByLabelText('Chat history')).toBeInTheDocument();
    expect(screen.getByText('Converse')).toBeInTheDocument();
    expect(screen.queryByText('ViDo')).not.toBeInTheDocument();
    expect(screen.queryByText(/no chats yet/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/no conversations/i)).not.toBeInTheDocument();
  });

  it('pins chats to the top of the chat history and can unpin them', () => {
    const user = { id: 'pin-sidebar-user', name: 'Demo User', email: 'demo@vido.app' };
    const first = createThread(user.id);
    addMessage(user.id, first.id, { id: 'u1', role: 'user', text: 'older chat' });
    const second = createThread(user.id);
    addMessage(user.id, second.id, { id: 'u2', role: 'user', text: 'newer chat' });
    setThreadPinned(user.id, first.id, true);

    render(
      <MemoryRouter initialEntries={['/preview/chat-lab']}>
        <WorkspaceSidebar counts={mockCounts} user={user} />
      </MemoryRouter>
    );

    const historyNav = screen.getByLabelText('Chat history').querySelector('nav');
    expect(historyNav).not.toBeNull();
    const titleButtons = within(historyNav as HTMLElement)
      .getAllByRole('button')
      .map((button) => button.textContent ?? '');
    expect(titleButtons[0]).toContain('older chat');
    expect(screen.getByRole('button', { name: 'Unpin older chat' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Unpin older chat' }));

    expect(screen.getByRole('button', { name: 'Pin older chat' })).toBeInTheDocument();
    const unpinnedTitles = within(historyNav as HTMLElement)
      .getAllByRole('button')
      .map((button) => button.textContent ?? '');
    expect(unpinnedTitles[0]).toContain('newer chat');
  });
});
