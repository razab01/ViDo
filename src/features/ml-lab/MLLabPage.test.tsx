import { describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { MLLabPage } from './MLLabPage';

describe('MLLabPage', () => {
  it('loads demo data, trains with the offline tokenizer, and runs RAG', async () => {
    render(<MLLabPage userId="mllab-train-user" />);

    expect(screen.getByText(/no trained model yet/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /load demo data/i }));

    expect(await screen.findByText(/300 rows/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/target column/i)).toHaveValue('completed');

    fireEvent.change(screen.getByLabelText(/gpt tokenizer/i), { target: { value: 'builtin' } });
    fireEvent.change(screen.getByLabelText(/lstm epochs/i), { target: { value: '2' } });
    fireEvent.click(screen.getByRole('button', { name: /train 10 models/i }));

    expect(await screen.findByText(/winner/i, {}, { timeout: 30000 })).toBeInTheDocument();
    expect(screen.queryByText(/tokenizer verification/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /show technical details/i }));
    expect(screen.getByText(/tokenizer verification/i)).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /suggestions/i })).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText(/ask about/i), {
      target: { value: 'Which task_text records look likely to be completed?' },
    });
    fireEvent.click(screen.getByRole('button', { name: /run query/i }));

    expect(await screen.findByText(/^answer$/i)).toBeInTheDocument();
  }, 60000);

  it('shows a friendly hint instead of an error badge for an empty RAG query', async () => {
    localStorage.clear();
    render(<MLLabPage userId="mllab-empty-rag-user" />);

    fireEvent.click(screen.getByRole('button', { name: /load demo data/i }));
    expect(await screen.findByText(/300 rows/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /run query/i }));

    expect(screen.getByText(/type a question or pick an example/i)).toBeInTheDocument();
    expect(screen.queryByText('NONE')).not.toBeInTheDocument();
  });
});
