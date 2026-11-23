import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ChatLabPage } from './ChatLabPage';
import { isPythonAvailable, answerWithPython } from '../ml-lab/pythonApi';
import { polishRagAnswer } from '../ml-lab/answerEngine';

vi.mock('../ml-lab/pythonApi', () => ({
  isPythonAvailable: vi.fn(),
  answerWithPython: vi.fn(),
}));

vi.mock('../ml-lab/answerEngine', () => ({
  polishRagAnswer: vi.fn(),
}));

const mockedAvailable = vi.mocked(isPythonAvailable);
const mockedAnswer = vi.mocked(answerWithPython);
const mockedPolish = vi.mocked(polishRagAnswer);

describe('ChatLabPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedAvailable.mockResolvedValue(true);
    mockedAnswer.mockResolvedValue('Spain has no president; the head of state is King Felipe VI.');
    mockedPolish.mockResolvedValue({
      mode: 'graph',
      why: '',
      plan: [],
      trail: '',
      ranked: [],
      answer: 'Browser fallback answer.',
    });
  });

  it('renders the empty state with suggestions', () => {
    render(<ChatLabPage />);
    expect(screen.getByText('Start a conversation')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/ask anything/i)).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'who is president of spain' })
    ).toBeInTheDocument();
  });

  it('answers through Python and shows the Python badge', async () => {
    render(<ChatLabPage />);
    const input = screen.getByPlaceholderText(/ask anything/i);
    fireEvent.change(input, { target: { value: 'who is president of spain' } });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));

    expect(screen.getByText('who is president of spain')).toBeInTheDocument();
    await waitFor(() => {
      expect(
        screen.getByText('Spain has no president; the head of state is King Felipe VI.')
      ).toBeInTheDocument();
    });
    expect(screen.getByText('Python')).toBeInTheDocument();
    expect(mockedAnswer).toHaveBeenCalledTimes(1);
    expect(mockedPolish).not.toHaveBeenCalled();
  });

  it('falls back to the browser engine when Python is unavailable', async () => {
    mockedAvailable.mockResolvedValue(false);
    render(<ChatLabPage />);
    fireEvent.change(screen.getByPlaceholderText(/ask anything/i), {
      target: { value: 'Tell me about India' },
    });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));

    await waitFor(() => {
      expect(screen.getByText('Browser fallback answer.')).toBeInTheDocument();
    });
    expect(screen.getByText('Local')).toBeInTheDocument();
    expect(mockedAnswer).not.toHaveBeenCalled();
    expect(mockedPolish).toHaveBeenCalledTimes(1);
  });

  it('falls back to the browser engine when the Python call fails', async () => {
    mockedAnswer.mockRejectedValue(new Error('service down'));
    render(<ChatLabPage />);
    fireEvent.change(screen.getByPlaceholderText(/ask anything/i), {
      target: { value: 'what is 2+2' },
    });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));

    await waitFor(() => {
      expect(screen.getByText('Browser fallback answer.')).toBeInTheDocument();
    });
    expect(screen.getByText('Local')).toBeInTheDocument();
  });

  it('sends on Enter and keeps multi-turn history', async () => {
    render(<ChatLabPage />);
    const input = screen.getByPlaceholderText(/ask anything/i);
    fireEvent.change(input, { target: { value: 'first question' } });
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(
        screen.getByText('Spain has no president; the head of state is King Felipe VI.')
      ).toBeInTheDocument();
    });

    fireEvent.change(input, { target: { value: 'second question' } });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));

    await waitFor(() => {
      expect(screen.getAllByText('Spain has no president; the head of state is King Felipe VI.')).toHaveLength(2);
    });
    expect(screen.getByText('first question')).toBeInTheDocument();
    expect(screen.getByText('second question')).toBeInTheDocument();
  });

  it('starts a new chat and clears the transcript', async () => {
    render(<ChatLabPage />);
    fireEvent.click(screen.getByRole('button', { name: 'who is president of spain' }));
    await waitFor(() => {
      expect(
        screen.getByText('Spain has no president; the head of state is King Felipe VI.')
      ).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: /new chat/i }));
    expect(screen.getByText('Start a conversation')).toBeInTheDocument();
    expect(
      screen.queryByText('Spain has no president; the head of state is King Felipe VI.')
    ).not.toBeInTheDocument();
  });
});
