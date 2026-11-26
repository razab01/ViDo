import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Badge, Button, Input } from '../../components/ui';
import { SampleTask } from '../../data/sampleTasks';
import type { TaskFlowchart } from '../../data/flowchart';
import { buildSampleTask } from '../../data/taskFactory';
import { FlowchartEditor } from './FlowchartEditor';
import { X, Tag, CheckCircle2, AlertCircle, Sparkles } from 'lucide-react';
import { formatTaskProbability } from '../ml-lab/mlStore';
import type { TaskPrediction, TaskPredictionInput } from '../ml-lab/mlStore';
import {
  getTodayDateString,
  getSevenDaysAgoDateString,
  getCurrentTimeString,
  getDefaultDueTime,
  validateChronologicalDeadline,
} from '../../utils/dateUtils';

export interface TaskCreateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreateTask?: (task: SampleTask) => void;
  /** Supplying a task switches this modal into edit mode. */
  initialTask?: SampleTask | null;
  onUpdateTask?: (task: SampleTask) => void;
  initialStartDate?: string;
  initialDueDate?: string;
  initialDueTime?: string;
  initialFlowchart?: TaskFlowchart | null;
  /** Reference "today" for validation/defaults. Preview passes its benchmark date. */
  referenceDate?: string;
  predictTask?: (input: TaskPredictionInput) => TaskPrediction | null;
}

export const TaskCreateModal: React.FC<TaskCreateModalProps> = ({
  isOpen,
  onClose,
  onCreateTask,
  initialTask,
  onUpdateTask,
  initialStartDate,
  initialDueDate,
  initialDueTime,
  initialFlowchart,
  referenceDate,
  predictTask,
}) => {
  // Preview mode has a fixed benchmark date. Editing uses it for validation so
  // seeded tasks stay editable even when the real clock has moved on. Creation
  // keeps the real clock because buildSampleTask timestamps against Date.now().
  const validationNow = useMemo(
    () => (referenceDate ? new Date(`${referenceDate}T12:00:00`) : new Date()),
    [referenceDate]
  );
  const validationTodayStr = getTodayDateString(validationNow);
  const realTodayStr = getTodayDateString();
  const sevenDaysAgoStr = getSevenDaysAgoDateString(validationNow);
  const currentTimeStr = getCurrentTimeString(validationNow);

  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [startDate, setStartDate] = useState(() => initialStartDate || '');
  const [dueDate, setDueDate] = useState(() => initialDueDate || realTodayStr);
  const [dueTime, setDueTime] = useState(() => initialDueTime ?? getDefaultDueTime());
  const [priority, setPriority] = useState<'high' | 'medium' | 'low' | ''>('medium');
  const [category, setCategory] = useState<'personal' | 'home' | 'work'>('personal');
  const [flowchart, setFlowchart] = useState<TaskFlowchart | null>(initialFlowchart ?? null);
  const [submittedNotice, setSubmittedNotice] = useState(false);

  const titleInputRef = useRef<HTMLInputElement>(null);

  // Compute live validation using useMemo
  const validation = useMemo(() => {
    return validateChronologicalDeadline({
      startDate,
      dueDate,
      dueTime,
      now: validationNow,
    });
  }, [startDate, dueDate, dueTime, validationNow]);

  const isFormValid = Boolean(title.trim()) && validation.isValid;

  const probability = useMemo(
    () =>
      predictTask?.({
        title,
        description,
        priority,
        category,
        startDate,
        dueDate,
        dueTime,
      }) ?? null,
    [predictTask, title, description, priority, category, startDate, dueDate, dueTime]
  );

  const isSelectedDateToday = dueDate === validationTodayStr;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Re-seed the form each time it opens so editing one task, then another,
  // never shows the previous task's values. Create mode keeps its own defaults.
  useEffect(() => {
    if (!isOpen) return;
    setSubmittedNotice(false);
    if (initialTask) {
      setTitle(initialTask.title || '');
      setDescription(initialTask.description || '');
      setStartDate(initialTask.startDate || '');
      setDueDate(initialTask.dueDate || validationTodayStr);
      setDueTime(initialTask.dueTime ?? getDefaultDueTime());
      setPriority(initialTask.priority ?? '');
      setCategory(initialTask.category ?? 'personal');
      setFlowchart(initialTask.flowchart ?? null);
    } else {
      setTitle('');
      setDescription('');
      setStartDate(initialStartDate || '');
      setDueDate(initialDueDate || realTodayStr);
      setDueTime(initialDueTime ?? getDefaultDueTime());
      setPriority('medium');
      setCategory('personal');
      setFlowchart(initialFlowchart ?? null);
    }
    setTimeout(() => titleInputRef.current?.focus(), 50);
  }, [
    isOpen,
    initialTask,
    initialStartDate,
    initialDueDate,
    initialDueTime,
    initialFlowchart,
    validationTodayStr,
    realTodayStr,
  ]);

  if (!isOpen) return null;

  const isEditMode = Boolean(initialTask);
  const submitLabel = isEditMode ? 'Save Changes' : 'Create Task';

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !validation.isValid) return;

    const probabilityNote = probability
      ? probability.isCompletion
        ? `Estimated completion chance: ${formatTaskProbability(probability.probability, probability.kind)}.`
        : `Estimated outcome: ${formatTaskProbability(probability.probability, probability.kind)} ${probability.label}.`
      : '';

    if (initialTask) {
      const updatedTask: SampleTask = {
        ...initialTask,
        title: title.trim(),
        description: description.trim() || undefined,
        startDate: startDate || undefined,
        dueDate,
        dueTime: dueTime || undefined,
        priority: (priority as 'high' | 'medium' | 'low') || undefined,
        category,
        flowchart: flowchart ?? undefined,
        checkpoints: probabilityNote
          ? [
              ...(initialTask.checkpoints || []),
              {
                id: `cp-${Date.now()}`,
                timestamp: new Date().toISOString(),
                note: `Task updated. ${probabilityNote}`,
              },
            ]
          : initialTask.checkpoints,
      };
      setSubmittedNotice(true);
      onUpdateTask?.(updatedTask);
      onClose();
      return;
    }

    const newTask: SampleTask = buildSampleTask({
      title,
      description,
      startDate,
      dueDate,
      dueTime,
      priority: (priority as 'high' | 'medium' | 'low') || undefined,
      category,
      flowchart,
      checkpointNote: probabilityNote || undefined,
    });

    setSubmittedNotice(true);
    onCreateTask?.(newTask);

    // Reset inputs
    setTitle('');
    setDescription('');
    setStartDate('');
    setDueDate(realTodayStr);
    setDueTime(getDefaultDueTime());
    setPriority('medium');
    setCategory('personal');
    setFlowchart(null);

    onClose();
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="create-task-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs animate-in fade-in duration-150"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg bg-[var(--surface)] border border-[var(--line)] rounded-2xl p-6 shadow-2xl relative max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          aria-label={isEditMode ? 'Close edit task modal' : 'Close create task modal'}
          className="absolute top-5 right-5 text-[var(--muted)] hover:text-[var(--ink)] p-1 rounded-lg hover:bg-[var(--canvas)] transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        <h2 id="create-task-title" className="text-xl font-bold text-[var(--ink)]">
          {isEditMode ? 'Edit Task' : 'Create New Task'}
        </h2>
        <p className="text-xs text-[var(--muted)] mt-1 mb-5">
          {isEditMode
            ? 'Update the details, schedule, priority, or flowchart for this task.'
            : 'Define exact deadlines and chronological ownership for your next responsibility.'}
        </p>

        {submittedNotice && (
          <div className="mb-5 p-3.5 rounded-xl bg-[var(--green-soft)] border border-[var(--green)]/30 text-xs text-[var(--green)] flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <p className="text-xs font-semibold leading-relaxed">
              {isEditMode ? 'Task changes saved to your workspace.' : 'Task successfully saved to your workspace.'}
            </p>
          </div>
        )}

        {validation.generalError && (
          <div
            role="alert"
            className="mb-5 p-3.5 rounded-xl bg-[var(--red-soft)] border border-[var(--red)]/30 text-xs text-[var(--red)] flex items-start gap-2.5 animate-in fade-in duration-150"
          >
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Invalid Chronological Combination</p>
              <p className="mt-0.5 text-[11px] leading-relaxed opacity-90">{validation.generalError}</p>
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            ref={titleInputRef}
            label="Task Title *"
            required
            placeholder="e.g., Finalize customer onboarding slide deck"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />

          <div>
            <label className="block text-xs font-semibold text-[var(--muted)] mb-1.5">
              Description (Optional)
            </label>
            <textarea
              rows={3}
              placeholder="Add key context, requirements, or links..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full px-3.5 py-2 text-sm bg-[var(--surface)] text-[var(--ink)] placeholder-[var(--faint)] border border-[var(--line)] rounded-lg transition-colors focus-visible:outline-2 focus-visible:outline-[var(--accent)] focus-visible:outline-offset-1 resize-none"
            />
          </div>

          <FlowchartEditor value={flowchart} onChange={setFlowchart} />

          <div className="rounded-xl border border-[var(--accent)]/25 bg-[var(--accent-soft)] px-3.5 py-3">
            <div className="flex flex-wrap items-center gap-2">
              <Sparkles className="w-3.5 h-3.5 text-[var(--accent)]" />
              <span className="text-xs font-bold text-[var(--ink)]">
                {probability?.isCompletion ? 'Completion probability' : 'Best-model estimate'}
              </span>
              {probability && (
                <Badge size="sm" variant="accent">
                  {formatTaskProbability(probability.probability, probability.kind)}
                </Badge>
              )}
            </div>
            {probability ? (
              <p className="text-[11px] text-[var(--muted)] mt-1.5 leading-relaxed">
                {probability.isCompletion
                  ? 'Completion chance based on your saved data.'
                  : `Estimated outcome: ${probability.label}.`}
                {title.trim() ? '' : ' Add a title to refine this estimate.'}
              </p>
            ) : (
              <p className="text-[11px] text-[var(--muted)] mt-1.5 leading-relaxed">
                Train a model in ML Lab to see this task&apos;s predicted outcome and probability.
              </p>
            )}
          </div>

          {/* Date & Time Selection */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <Input
              label="Start Date (Optional)"
              type="date"
              min={sevenDaysAgoStr}
              value={startDate}
              error={validation.startError}
              helperText="Defaults to today if omitted (max 7 days in past)"
              onChange={(e) => setStartDate(e.target.value)}
            />

            <Input
              label="Target Due Date *"
              type="date"
              required
              min={startDate || sevenDaysAgoStr}
              value={dueDate}
              error={validation.dateError}
              onChange={(e) => setDueDate(e.target.value)}
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <Input
              label="Due Time (Optional)"
              type="time"
              value={dueTime}
              error={validation.timeError}
              helperText={
                isSelectedDateToday && !startDate
                  ? `Must be after ${currentTimeStr} (defaults to 11:59 PM if omitted)`
                  : 'Defaults to 11:59 PM if omitted'
              }
              onChange={(e) => setDueTime(e.target.value)}
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            <div>
              <label className="block text-xs font-semibold text-[var(--muted)] mb-1.5 flex items-center gap-1">
                <Tag className="w-3 h-3 text-[var(--faint)]" /> Category
              </label>
              <select
                aria-label="Category"
                value={category}
                onChange={(e) => setCategory(e.target.value as any)}
                className="w-full px-3 py-2 text-xs bg-[var(--surface)] text-[var(--ink)] border border-[var(--line)] rounded-lg focus-visible:outline-2 focus-visible:outline-[var(--accent)]"
              >
                <option value="personal">Personal</option>
                <option value="home">Household / Living</option>
                <option value="work">Work / Professional</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-[var(--muted)] mb-1.5">
                Priority Level
              </label>
              <select
                aria-label="Priority Level"
                value={priority}
                onChange={(e) => setPriority(e.target.value as any)}
                className="w-full px-3 py-2 text-xs bg-[var(--surface)] text-[var(--ink)] border border-[var(--line)] rounded-lg focus-visible:outline-2 focus-visible:outline-[var(--accent)]"
              >
                <option value="">None</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
          </div>

          <div className="pt-4 border-t border-[var(--line)] flex items-center justify-end gap-3">
            <Button type="button" variant="secondary" size="sm" onClick={onClose}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              size="sm"
              disabled={!isFormValid}
              aria-label={submitLabel}
            >
              {submitLabel}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};
