import { describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { MLLabPage } from './MLLabPage';

const setOfflineTrainingOptions = () => {
  fireEvent.click(screen.getByRole('button', { name: /advanced options/i }));
  fireEvent.change(screen.getByLabelText(/gpt tokenizer/i), { target: { value: 'builtin' } });
  fireEvent.change(screen.getByLabelText(/deep model epochs/i), { target: { value: '2' } });
  fireEvent.click(screen.getByRole('button', { name: /hide advanced options/i }));
};

describe('MLLabPage', () => {
  it('trains automatically after upload and keeps technical details out of the simple view', async () => {
    render(<MLLabPage userId="mllab-train-user" />);

    expect(screen.getByText(/upload a file to get started/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /train 10 deep models/i })).not.toBeInTheDocument();

    setOfflineTrainingOptions();
    fireEvent.click(screen.getByRole('button', { name: /load demo data/i }));

    expect(await screen.findByText(/learning from your file/i)).toBeInTheDocument();
    expect(await screen.findByText(/ready to answer questions/i, {}, { timeout: 30000 })).toBeInTheDocument();
    expect(screen.getByText(/learned from 300 rows/i)).toBeInTheDocument();
    expect(screen.queryByText(/tokenizer verification/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /advanced options/i }));
    expect(screen.getByText(/tokenizer verification/i)).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /suggestions/i })).toBeInTheDocument();
    expect(screen.getByText(/always uses graph rag/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Graph' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Agentic' })).not.toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText(/ask about your data or any topic/i), {
      target: { value: 'Which task_text records look likely to be completed?' },
    });
    fireEvent.click(screen.getByRole('button', { name: /find answer/i }));

    expect(screen.getByText(/finding the answer/i)).toBeInTheDocument();
    expect(await screen.findByText(/^answer$/i)).toBeInTheDocument();
  }, 60000);

  it('shows a friendly hint instead of an error badge for an empty question', async () => {
    localStorage.clear();
    render(<MLLabPage userId="mllab-empty-rag-user" />);

    setOfflineTrainingOptions();
    fireEvent.click(screen.getByRole('button', { name: /load demo data/i }));
    expect(await screen.findByText(/ready to answer questions/i, {}, { timeout: 30000 })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /find answer/i }));

    expect(screen.getByText(/type a question or pick an example/i)).toBeInTheDocument();
    expect(screen.queryByText('NONE')).not.toBeInTheDocument();
  }, 60000);
});
