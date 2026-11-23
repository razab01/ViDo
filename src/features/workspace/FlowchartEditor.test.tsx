import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { FlowchartEditor } from './FlowchartEditor';
import { parseFlowchartSteps } from '../../data/flowchart';

describe('FlowchartEditor', () => {
  it('emits a parsed flowchart as steps are typed', () => {
    const onChange = vi.fn();
    render(<FlowchartEditor value={null} onChange={onChange} />);

    fireEvent.change(screen.getByLabelText(/flowchart steps/i), {
      target: { value: 'Plan -> Build -> Review' },
    });

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        nodes: [
          expect.objectContaining({ label: 'Plan' }),
          expect.objectContaining({ label: 'Build' }),
          expect.objectContaining({ label: 'Review' }),
        ],
        edges: expect.arrayContaining([expect.any(Object)]),
      })
    );
  });

  it('renders an SVG preview and can append a step', () => {
    const onChange = vi.fn();
    const value = parseFlowchartSteps('Plan -> Build');
    render(<FlowchartEditor value={value} onChange={onChange} />);

    expect(screen.getByRole('img', { name: /flowchart preview/i })).toBeInTheDocument();
    expect(screen.getByText('2 steps')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /add step/i }));
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        nodes: [
          expect.objectContaining({ label: 'Plan' }),
          expect.objectContaining({ label: 'Build' }),
          expect.objectContaining({ label: 'New step' }),
        ],
      })
    );
  });
});
