import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { TaskCreateModal } from './TaskCreateModal';
import type { TaskPrediction } from '../ml-lab/mlStore';

const estimate: TaskPrediction = {
  label: 'yes',
  probability: 0.91,
  model: 'Fake best',
  target: 'completed',
  kind: 'confidence',
  isCompletion: true,
};

describe('TaskCreateModal best-model probability', () => {
  it('shows and stores the winning-model estimate for a new task', () => {
    const onCreateTask = vi.fn();
    const predictTask = vi.fn(() => estimate);
    render(
      <TaskCreateModal
        isOpen
        onClose={() => undefined}
        onCreateTask={onCreateTask}
        predictTask={predictTask}
      />
    );

    fireEvent.change(screen.getByPlaceholderText(/Finalize customer onboarding/i), {
      target: { value: 'Finish quarterly report' },
    });

    expect(screen.getByText('Completion probability')).toBeInTheDocument();
    expect(screen.getByText('91%')).toBeInTheDocument();
    expect(screen.getByText(/completion chance based on your saved data/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Create Task' }));

    expect(predictTask).toHaveBeenCalled();
    expect(onCreateTask).toHaveBeenCalledTimes(1);
    expect(onCreateTask.mock.calls[0][0].checkpoints[0].note).toContain('Estimated completion chance: 91%.');
  });
});
