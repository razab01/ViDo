import React from 'react';
import { Plus, Trash2 } from 'lucide-react';
import { Badge, Button } from '../../components/ui';
import {
  flowchartToText,
  updateFlowchartFromText,
  validateFlowchart,
  type TaskFlowchart,
} from '../../data/flowchart';
import { FlowchartPreview } from './FlowchartPreview';

export interface FlowchartEditorProps {
  value: TaskFlowchart | null | undefined;
  onChange: (next: TaskFlowchart | null) => void;
}

/** Compact editor: edit steps as arrow-separated text and see the SVG live. */
export const FlowchartEditor: React.FC<FlowchartEditorProps> = ({ value, onChange }) => {
  const text = flowchartToText(value);
  const validationError = text ? validateFlowchart(value) : null;

  const handleTextChange = (raw: string) => {
    onChange(updateFlowchartFromText(raw, value));
  };

  const addStep = () => {
    if (!value?.nodes?.length) {
      onChange(updateFlowchartFromText('Plan -> Review', null));
      return;
    }
    const labels = [...value.nodes.map((node) => node.label), 'New step'];
    onChange(updateFlowchartFromText(labels.join(' -> '), value));
  };

  return (
    <section className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-3.5 space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold text-[var(--ink)]">Flowchart (optional)</p>
          <p className="text-[11px] text-[var(--muted)] mt-0.5">
            Map the steps for this task. Separate each step with -&gt;.
          </p>
        </div>
        {value?.nodes?.length ? (
          <Badge size="sm" variant="accent">
            {value.nodes.length} steps
          </Badge>
        ) : null}
      </div>

      <textarea
        rows={2}
        aria-label="Flowchart steps"
        value={text}
        onChange={(event) => handleTextChange(event.target.value)}
        placeholder="Research -> Design -> Build -> Review"
        className="w-full px-3.5 py-2 text-sm bg-[var(--surface)] text-[var(--ink)] placeholder-[var(--faint)] border border-[var(--line)] rounded-lg focus-visible:outline-2 focus-visible:outline-[var(--accent)] focus-visible:outline-offset-1 resize-none"
      />

      {validationError && <p className="text-[11px] text-[var(--red)]">{validationError}</p>}

      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Button type="button" variant="secondary" size="sm" onClick={addStep}>
            <Plus className="w-3.5 h-3.5" /> Add step
          </Button>
          {value?.nodes?.length ? (
            <Button type="button" variant="ghost" size="sm" onClick={() => onChange(null)}>
              <Trash2 className="w-3.5 h-3.5" /> Clear
            </Button>
          ) : null}
        </div>
        <span className="text-[10px] text-[var(--faint)]">Example: Plan -&gt; Build -&gt; Review</span>
      </div>

      {value?.nodes?.length ? <FlowchartPreview flowchart={value} /> : null}
    </section>
  );
};

export default FlowchartEditor;
