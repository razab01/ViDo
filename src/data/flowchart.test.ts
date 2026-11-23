import { describe, expect, it } from 'vitest';
import {
  flowchartToText,
  parseFlowchartSteps,
  updateFlowchartFromText,
  validateFlowchart,
} from './flowchart';

describe('flowchart data helpers', () => {
  it('parses arrow-separated steps into nodes and connected edges', () => {
    const flowchart = parseFlowchartSteps('Research -> Design -> Build');

    expect(flowchart?.nodes.map((node) => node.label)).toEqual(['Research', 'Design', 'Build']);
    expect(flowchart?.edges).toEqual([
      { id: 'edge-1', from: 'step-1', to: 'step-2' },
      { id: 'edge-2', from: 'step-2', to: 'step-3' },
    ]);
  });

  it('extracts steps from a task message that mentions a flowchart', () => {
    const flowchart = parseFlowchartSteps(
      'create a high priority task to launch with flowchart: Research -> Design -> Launch'
    );
    expect(flowchartToText(flowchart)).toBe('Research -> Design -> Launch');
  });

  it('supports then and comma separated steps', () => {
    expect(flowchartToText(parseFlowchartSteps('Plan, Build, Review'))).toBe(
      'Plan -> Build -> Review'
    );
    expect(flowchartToText(parseFlowchartSteps('Plan then Build then Review'))).toBe(
      'Plan -> Build -> Review'
    );
  });

  it('rejects a single step', () => {
    expect(parseFlowchartSteps('Just one step')).toBeNull();
    expect(parseFlowchartSteps('create a flowchart for Launch plan')).toBeNull();
  });

  it('validates missing nodes and broken edges', () => {
    expect(validateFlowchart(null)).toBeNull();
    expect(validateFlowchart({ nodes: [], edges: [] })).toMatch(/at least two steps/i);
    expect(
      validateFlowchart({
        nodes: [
          { id: 'a', label: 'Plan' },
          { id: 'b', label: 'Build' },
        ],
        edges: [{ id: 'e', from: 'a', to: 'missing' }],
      })
    ).toMatch(/existing steps/i);
  });

  it('preserves node and edge ids while editing step text', () => {
    const initial = parseFlowchartSteps('Plan -> Review');
    const updated = updateFlowchartFromText('Plan -> Draft -> Review', initial);

    expect(updated?.nodes.map((node) => node.id)).toEqual(['step-1', 'step-2', 'step-3']);
    expect(updated?.nodes.map((node) => node.label)).toEqual(['Plan', 'Draft', 'Review']);
    expect(updated?.edges).toHaveLength(2);
    expect(validateFlowchart(updated)).toBeNull();
  });
});
