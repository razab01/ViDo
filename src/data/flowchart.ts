/**
 * Flowchart model and parsing helpers.
 *
 * A task flowchart is deliberately tiny: an ordered list of steps and the
 * connections between them. It is rendered with plain SVG so the feature keeps
 * working offline and does not require a diagramming dependency.
 */

export interface FlowchartNode {
  id: string;
  label: string;
}

export interface FlowchartEdge {
  id: string;
  from: string;
  to: string;
}

export interface TaskFlowchart {
  nodes: FlowchartNode[];
  edges: FlowchartEdge[];
}

const ARROW_SEPARATOR = /\s*(?:->|→|=>)\s*/;

const stripPrompt = (raw: string): string => {
  let value = String(raw || '').trim();

  // When a task message embeds a flowchart, everything before the word
  // "flowchart" is task context rather than diagram content.
  const marker = value.search(/\b(?:flowchart|flow\s*chart|diagram|workflow)\b/i);
  if (marker >= 0) value = value.slice(marker);

  value = value.replace(
    /^(?:(?:ok|okay|so|now|and|well|please|hey|yes|yeah|yep|sure)\b[\s,.:-]*)*/i,
    ''
  );
  value = value.replace(/^(?:can|could)\s+you\s+(?:please\s+)?/i, '');
  value = value.replace(
    /^(?:create|add|make|draw|build|show|generate|attach|include)\s+(?:a|an|the|new)?\s*(?:flowchart|flow\s*chart|diagram|workflow)\b\s*(?:for|of|about|to|:)?\s*/i,
    ''
  );
  value = value.replace(
    /\b(?:with|and|include|including|attach)\s+(?:a\s+)?(?:flowchart|flow\s*chart|diagram|workflow)\b\s*(?:for|of|about|to|:)?\s*/i,
    ''
  );
  value = value.replace(
    /^(?:create|add|make|new|set\s+up)\s+(?:a|an|the|new)?\s*(?:to-?do|todo|task|reminder|flowchart|flow\s*chart|diagram|workflow)\b\s*(?:for|to|about|:)?\s*/i,
    ''
  );
  value = value.replace(
    /^\s*(?:flowchart|flow\s*chart|diagram|workflow)\b\s*(?:steps?\s*)?(?:for|of|about|to|:)?\s*/i,
    ''
  );
  value = value.replace(/^\s*(?:flowchart|flow\s*chart|diagram|workflow)\s*[:：-]\s*/i, '');

  // "Launch plan: Research -> Build" means the steps start after the colon.
  if (ARROW_SEPARATOR.test(value) && !value.includes('\n')) {
    value = value.replace(/^[^:]+:\s*(?=.*(?:->|→|=>))/, '');
  }

  return value.trim();
};

const cleanStep = (raw: string): string =>
  String(raw || '')
    .trim()
    .replace(/^(?:step\s*\d+\s*[:.)-]\s*)/i, '')
    .replace(/[.;,]+$/, '')
    .trim();

const splitLine = (line: string): string[] => {
  const value = line
    .trim()
    .replace(/^(?:steps?|flow|plan)\s*[:：]\s*/i, '')
    .trim();
  if (!value) return [];
  if (ARROW_SEPARATOR.test(value)) return value.split(ARROW_SEPARATOR);
  if (/\bthen\b/i.test(value)) return value.split(/\s+\bthen\b\s+/i);
  if (value.includes(',')) return value.split(/\s*,\s*/);
  return [value];
};

/** Parse user text such as "Research -> Design -> Build" into a flowchart. */
export function parseFlowchartSteps(raw: string): TaskFlowchart | null {
  const value = stripPrompt(raw);
  if (!value) return null;

  const labels: string[] = [];
  const lines = value.split(/\n+|;+/);
  for (const line of lines) {
    for (const part of splitLine(line)) {
      const label = cleanStep(part);
      if (!label) continue;
      if (labels[labels.length - 1]?.toLowerCase() === label.toLowerCase()) continue;
      labels.push(label);
    }
  }

  if (labels.length < 2) return null;

  const nodes = labels.map((label, index) => ({ id: `step-${index + 1}`, label }));
  const edges = nodes.slice(1).map((node, index) => ({
    id: `edge-${index + 1}`,
    from: nodes[index].id,
    to: node.id,
  }));
  return { nodes, edges };
}

/** Serialize a flowchart to the arrow notation used by chat and the editor. */
export function flowchartToText(flowchart: TaskFlowchart | null | undefined): string {
  if (!flowchart?.nodes?.length) return '';
  return flowchart.nodes.map((node) => node.label).join(' -> ');
}

/** Return a readable error when a flowchart cannot be attached to a task. */
export function validateFlowchart(flowchart: TaskFlowchart | null | undefined): string | null {
  if (!flowchart) return null;
  if (flowchart.nodes.length < 2) return 'Add at least two steps to the flowchart.';
  if (flowchart.nodes.some((node) => !node.label.trim())) return 'Every flowchart step needs a label.';

  const ids = new Set<string>();
  for (const node of flowchart.nodes) {
    if (ids.has(node.id)) return 'Flowchart steps must have unique IDs.';
    ids.add(node.id);
  }

  for (const edge of flowchart.edges) {
    if (!ids.has(edge.from) || !ids.has(edge.to)) {
      return 'Flowchart connections must reference existing steps.';
    }
  }
  return null;
}

/** Build a flowchart from editable step text, preserving IDs where possible. */
export function updateFlowchartFromText(
  raw: string,
  previous: TaskFlowchart | null | undefined
): TaskFlowchart | null {
  const parsed = parseFlowchartSteps(raw);
  if (!parsed) return null;
  const nodes = parsed.nodes.map((node, index) => ({
    id: previous?.nodes?.[index]?.id ?? node.id,
    label: node.label,
  }));
  return {
    nodes,
    edges: nodes.slice(1).map((node, index) => ({
      id: previous?.edges?.[index]?.id ?? `edge-${index + 1}`,
      from: nodes[index].id,
      to: node.id,
    })),
  };
}

/** Build a sensible process graph for a task the user already created. */
export function buildTaskFlowchart(title: string): TaskFlowchart {
  const subject = String(title || '').trim().replace(/[.!?]+$/, '') || 'Task';
  const labels = [
    `Plan ${subject}`,
    'Gather what you need',
    `Do the work: ${subject}`,
    'Review the result',
    'Mark it complete',
  ];
  const nodes = labels.map((label, index) => ({ id: `step-${index + 1}`, label }));
  return {
    nodes,
    edges: nodes.slice(1).map((node, index) => ({
      id: `edge-${index + 1}`,
      from: nodes[index].id,
      to: node.id,
    })),
  };
}
