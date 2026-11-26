import React, { useState } from 'react';
import { SampleTask } from '../../../data/sampleTasks';
import { TaskCard } from '../TaskCard';
import { Button } from '../../../components/ui';
import { sortTasksPinnedFirst } from '../../../data/taskFilters';
import { Inbox } from 'lucide-react';

export interface ListViewProps {
  title: string;
  description: string;
  tasks: SampleTask[];
  onTaskClick: (task: SampleTask) => void;
  onEditTask?: (task: SampleTask) => void;
  onTogglePin?: (task: SampleTask) => void;
  onDeleteTask?: (task: SampleTask) => void;
  onNewTaskClick?: () => void;
  isPreview?: boolean;
}

export const ListView: React.FC<ListViewProps> = ({
  title,
  description,
  tasks,
  onTaskClick,
  onEditTask,
  onTogglePin,
  onDeleteTask,
  onNewTaskClick,
  isPreview = true,
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const orderedTasks = sortTasksPinnedFirst(tasks);

  const filteredTasks =
    selectedCategory === 'all'
      ? orderedTasks
      : orderedTasks.filter((t) => t.category === selectedCategory);

  const categories = [
    { id: 'all', label: 'All Tasks', count: orderedTasks.length },
    {
      id: 'personal',
      label: 'Personal',
      count: orderedTasks.filter((t) => t.category === 'personal').length,
    },
    {
      id: 'home',
      label: 'Household',
      count: orderedTasks.filter((t) => t.category === 'home').length,
    },
    {
      id: 'work',
      label: 'Work',
      count: orderedTasks.filter((t) => t.category === 'work').length,
    },
  ];

  return (
    <div className="vdb-list space-y-6 p-8 max-w-6xl mx-auto">
      {/* Header and Filter Tabs */}
      <div className="vdb-list-header flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[var(--line)]">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-[var(--ink)]">{title}</h2>
          <p className="text-xs text-[var(--muted)] mt-0.5">{description}</p>
        </div>

        {/* Category Filter Pills */}
        <div className="vdb-filter-pills flex items-center gap-1.5 bg-[var(--canvas)] p-1 rounded-xl border border-[var(--line)] self-start sm:self-auto">
          {categories.map((cat) => (
            <button
              key={cat.id}
              onClick={() => setSelectedCategory(cat.id)}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                selectedCategory === cat.id
                  ? 'bg-[var(--surface)] text-[var(--ink)] shadow-xs'
                  : 'text-[var(--muted)] hover:text-[var(--ink)]'
              }`}
            >
              {cat.label} ({cat.count})
            </button>
          ))}
        </div>
      </div>

      {/* Task Cards Grid */}
      {filteredTasks.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredTasks.map((task) => (
            <TaskCard
              key={task.id}
              task={task}
              onClick={onTaskClick}
              onEdit={onEditTask}
              onTogglePin={onTogglePin}
              onDelete={onDeleteTask}
              isPreview={isPreview}
            />
          ))}
        </div>
      ) : (
        <div className="vdb-empty p-12 text-center border border-dashed border-[var(--line)] rounded-2xl bg-[var(--surface)] my-8">
          <div className="w-12 h-12 rounded-full bg-[var(--canvas)] text-[var(--faint)] flex items-center justify-center mx-auto mb-3">
            <Inbox className="w-6 h-6" />
          </div>
          <h3 className="text-sm font-semibold text-[var(--ink)]">No tasks in this category</h3>
          <p className="text-xs text-[var(--muted)] mt-1 max-w-sm mx-auto">
            All clear. There are no items matching this view and category filter.
          </p>
          {onNewTaskClick && (
            <div className="mt-4">
              <Button variant="secondary" size="sm" onClick={onNewTaskClick}>
                Create New Task
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
