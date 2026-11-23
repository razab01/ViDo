import React, { useId } from 'react';
import type { TaskFlowchart } from '../../data/flowchart';

export interface FlowchartPreviewProps {
  flowchart: TaskFlowchart | null | undefined;
  className?: string;
}

const NODE_WIDTH = 132;
const NODE_HEIGHT = 40;
const GAP = 46;

const shorten = (label: string): string =>
  label.length > 18 ? `${label.slice(0, 17)}…` : label;

/** Read-only SVG rendering of a task flowchart. */
export const FlowchartPreview: React.FC<FlowchartPreviewProps> = ({ flowchart, className = '' }) => {
  const markerId = `flow-arrow-${useId().replace(/[^a-zA-Z0-9-]/g, '')}`;
  const nodes = flowchart?.nodes ?? [];
  if (!nodes.length) return null;

  const width = Math.max(560, nodes.length * NODE_WIDTH + (nodes.length - 1) * GAP + 24);
  const height = NODE_HEIGHT + 32;
  const nodeX = (index: number) => 12 + index * (NODE_WIDTH + GAP);
  const indexById = new Map(nodes.map((node, index) => [node.id, index]));

  return (
    <div className={`overflow-x-auto rounded-xl border border-[var(--line)] bg-[var(--canvas)] p-2 ${className}`}>
      <svg
        role="img"
        aria-label="Flowchart preview"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
      >
        <defs>
          <marker
            id={markerId}
            markerWidth="8"
            markerHeight="8"
            refX="7"
            refY="4"
            orient="auto"
          >
            <path d="M 0 0 L 8 4 L 0 8 z" fill="var(--muted)" />
          </marker>
        </defs>

        {(flowchart?.edges ?? []).map((edge) => {
          const fromIndex = indexById.get(edge.from);
          const toIndex = indexById.get(edge.to);
          if (fromIndex === undefined || toIndex === undefined) return null;
          const x1 = nodeX(fromIndex) + NODE_WIDTH;
          const x2 = nodeX(toIndex) - 4;
          const y = 16 + NODE_HEIGHT / 2;
          return (
            <line
              key={edge.id}
              x1={x1}
              y1={y}
              x2={x2}
              y2={y}
              stroke="var(--muted)"
              strokeWidth="1.5"
              markerEnd={`url(#${markerId})`}
            />
          );
        })}

        {nodes.map((node, index) => (
          <g key={node.id}>
            <rect
              x={nodeX(index)}
              y={16}
              width={NODE_WIDTH}
              height={NODE_HEIGHT}
              rx="10"
              fill="var(--accent-soft)"
              stroke="var(--accent)"
            />
            <text
              x={nodeX(index) + NODE_WIDTH / 2}
              y={16 + NODE_HEIGHT / 2 + 4}
              textAnchor="middle"
              fontSize="11"
              fontWeight="600"
              fill="var(--ink)"
            >
              <title>{node.label}</title>
              {shorten(node.label)}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
};

export default FlowchartPreview;
