import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ChatLabPage } from './ChatLabPage';
import { isPythonAvailable, answerWithPython } from '../ml-lab/pythonApi';
import { answerLocally } from './chatBrain';
import { getActiveChatId, readThreads, resetChatStore } from './chatStore';

vi.mock('../ml-lab/pythonApi', () => ({
  isPythonAvailable: vi.fn(),
  answerWithPython: vi.fn(),
}));

vi.mock('./chatBrain', async () => {
  const actual = await vi.importActual<typeof import('./chatBrain')>('./chatBrain');
  return {
    ...actual,
    answerLocally: vi.fn(),
  };
});

const mockedAvailable = vi.mocked(isPythonAvailable);
const mockedAnswer = vi.mocked(answerWithPython);
const mockedLocal = vi.mocked(answerLocally);

describe('ChatLabPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    resetChatStore();
    mockedAvailable.mockResolvedValue(true);
    mockedAnswer.mockResolvedValue('Spain has no president; the head of state is King Felipe VI.');
    mockedLocal.mockResolvedValue('India is a country in South Asia.');
  });

  it('renders the empty state with suggestions', async () => {
    render(<ChatLabPage />);
    expect(await screen.findByText('Start a conversation')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/ask anything/i)).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'who is president of spain' })
    ).toBeInTheDocument();
  });

  it('answers through Python without showing any engine badge', async () => {
    render(<ChatLabPage />);
    await screen.findByText('Start a conversation');
    fireEvent.change(screen.getByPlaceholderText(/ask anything/i), {
      target: { value: 'who is president of spain' },
    });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));

    await waitFor(() => {
      expect(
        screen.getByText('Spain has no president; the head of state is King Felipe VI.')
      ).toBeInTheDocument();
    });
    expect(screen.queryByText('Python')).not.toBeInTheDocument();
    expect(screen.queryByText('Local')).not.toBeInTheDocument();
    expect(mockedAnswer).toHaveBeenCalledTimes(1);
  });

  it('uses the local answer when Python is unavailable', async () => {
    mockedAvailable.mockResolvedValue(false);
    render(<ChatLabPage />);
    await screen.findByText('Start a conversation');
    fireEvent.click(screen.getByRole('button', { name: 'Tell me about India' }));

    await waitFor(() => {
      expect(screen.getByText('India is a country in South Asia.')).toBeInTheDocument();
    });
    expect(mockedAnswer).not.toHaveBeenCalled();
    expect(mockedLocal).toHaveBeenCalledTimes(1);
  });

  it('resolves pronouns from the previous question', async () => {
    mockedAnswer
      .mockResolvedValueOnce('Adam Smith was a Scottish economist.')
      .mockResolvedValueOnce('Adam Smith wrote The Wealth of Nations.');
    render(<ChatLabPage />);
    await screen.findByText('Start a conversation');
    const input = screen.getByPlaceholderText(/ask anything/i);

    fireEvent.change(input, { target: { value: 'who is adam smith' } });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));
    await screen.findByText('Adam Smith was a Scottish economist.');

    fireEvent.change(input, { target: { value: 'what did he write?' } });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));
    await screen.findByText('Adam Smith wrote The Wealth of Nations.');

    expect(mockedAnswer.mock.calls[1][0].question).toBe('what did Adam Smith write?');
  });

  it('persists chat history across remounts', async () => {
    const first = render(<ChatLabPage />);
    await first.findByText('Start a conversation');
    fireEvent.change(screen.getByPlaceholderText(/ask anything/i), {
      target: { value: 'who is president of spain' },
    });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));
    await screen.findByText('Spain has no president; the head of state is King Felipe VI.');
    const chatId = getActiveChatId('demo-user-uid');
    first.unmount();

    render(<ChatLabPage />);
    expect(await screen.findByText('who is president of spain')).toBeInTheDocument();
    expect(getActiveChatId('demo-user-uid')).toBe(chatId);
    expect(readThreads('demo-user-uid')[0].messages).toHaveLength(2);
  });

  it('starts a new chat and keeps the old one in history', async () => {
    mockedAvailable.mockResolvedValue(false);
    render(<ChatLabPage />);
    await screen.findByText('Start a conversation');
    fireEvent.click(screen.getByRole('button', { name: 'Tell me about India' }));
    await screen.findByText('India is a country in South Asia.');

    fireEvent.click(screen.getByRole('button', { name: /new chat/i }));
    expect(await screen.findByText('Start a conversation')).toBeInTheDocument();
    expect(readThreads('demo-user-uid')).toHaveLength(2);
  });
});
