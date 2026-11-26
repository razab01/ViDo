import { describe, it, expect, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { SAMPLE_TASKS } from '../../data/sampleTasks';
import { TaskRow } from './TaskRow';
import { TaskCard } from './TaskCard';

describe('task row and card actions', () => {
  const task = { ...SAMPLE_TASKS[0], isPinned: false };

  it('pins and deletes from a task row without opening the task', () => {
    const onTaskClick = vi.fn();
    const onTogglePin = vi.fn();
    const onDelete = vi.fn();

    render(
      <TaskRow
        task={task}
        onClick={onTaskClick}
        onTogglePin={onTogglePin}
        onDelete={onDelete}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: `Pin "${task.title}"` }));
    expect(onTogglePin).toHaveBeenCalledWith(task);
    expect(onTaskClick).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: `Delete "${task.title}"` }));
    expect(onDelete).toHaveBeenCalledWith(task);
    expect(onTaskClick).not.toHaveBeenCalled();
  });

  it('shows an unpin action for a pinned task row', () => {
    render(
      <TaskRow
        task={{ ...task, isPinned: true }}
        onClick={vi.fn()}
        onTogglePin={vi.fn()}
        onDelete={vi.fn()}
      />
    );

    expect(screen.getByRole('button', { name: `Unpin "${task.title}"` })).toBeInTheDocument();
  });

  it('pins and deletes from a task card without opening the task', () => {
    const onTaskClick = vi.fn();
    const onTogglePin = vi.fn();
    const onDelete = vi.fn();

    render(
      <TaskCard
        task={task}
        onClick={onTaskClick}
        onTogglePin={onTogglePin}
        onDelete={onDelete}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: `Pin "${task.title}"` }));
    expect(onTogglePin).toHaveBeenCalledWith(task);
    expect(onTaskClick).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: `Delete "${task.title}"` }));
    expect(onDelete).toHaveBeenCalledWith(task);
    expect(onTaskClick).not.toHaveBeenCalled();
  });
});
