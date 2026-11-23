import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ChatLabPage } from './ChatLabPage';
import {
  extractDocumentWithPython,
  isPythonAvailable,
  answerWithPython,
} from '../ml-lab/pythonApi';
import { answerLocally } from './chatBrain';
import { getActiveChatId, readThreads, resetChatStore } from './chatStore';
import { getTodayDateString } from '../../utils/dateUtils';

const tomorrowDateString = (): string => {
  const date = new Date();
  date.setDate(date.getDate() + 1);
  return getTodayDateString(date);
};

vi.mock('../ml-lab/pythonApi', () => ({
  isPythonAvailable: vi.fn(),
  answerWithPython: vi.fn(),
  extractDocumentWithPython: vi.fn(),
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
const mockedExtract = vi.mocked(extractDocumentWithPython);

describe('ChatLabPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    resetChatStore();
    mockedAvailable.mockResolvedValue(true);
    mockedAnswer.mockResolvedValue('Spain has no president; the head of state is King Felipe VI.');
    mockedLocal.mockResolvedValue('India is a country in South Asia.');
    mockedExtract.mockResolvedValue({
      name: 'report.pdf',
      text: 'The project deadline is 2026-12-01.',
      truncated: false,
    });
  });

  const sendMessage = (value: string) => {
    fireEvent.change(screen.getByPlaceholderText(/ask anything/i), { target: { value } });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));
  };

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
    sendMessage('who is president of spain');

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

    sendMessage('who is adam smith');
    await screen.findByText('Adam Smith was a Scottish economist.');

    sendMessage('what did he write?');
    await screen.findByText('Adam Smith wrote The Wealth of Nations.');

    expect(mockedAnswer.mock.calls[1][0].question).toBe('what did Adam Smith write?');
  });

  it('persists chat history across remounts', async () => {
    const first = render(<ChatLabPage />);
    await first.findByText('Start a conversation');
    sendMessage('who is president of spain');
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

  it('understands relative dates and always asks about a flowchart', async () => {
    const onCreateTask = vi.fn();
    render(<ChatLabPage onCreateTask={onCreateTask} />);
    await screen.findByText('Start a conversation');

    sendMessage('hey create a task for tomorrow morning 4 am for walk.');
    expect(await screen.findByText(/would you like a flowchart/i)).toBeInTheDocument();
    expect(onCreateTask).not.toHaveBeenCalled();
    expect(readThreads('demo-user-uid')[0].taskDraft).toEqual(
      expect.objectContaining({
        title: 'walk',
        startDate: getTodayDateString(),
        dueDate: tomorrowDateString(),
        dueTime: '04:00',
        flowchartPrompted: true,
      })
    );

    sendMessage('no');
    await waitFor(() => expect(onCreateTask).toHaveBeenCalledTimes(1));
    expect(onCreateTask).toHaveBeenCalledWith(
      expect.objectContaining({
        title: 'walk',
        startDate: getTodayDateString(),
        dueDate: tomorrowDateString(),
        dueTime: '04:00',
        state: 'active',
      })
    );
    expect(mockedAnswer).not.toHaveBeenCalled();
    expect(mockedLocal).not.toHaveBeenCalled();
  });

  it('creates a task with a flowchart from chat', async () => {
    const onCreateTask = vi.fn();
    render(<ChatLabPage onCreateTask={onCreateTask} />);
    await screen.findByText('Start a conversation');

    sendMessage('create a flowchart for Launch plan');
    expect(await screen.findByText(/flowchart steps/i)).toBeInTheDocument();

    sendMessage('Research -> Design -> Build');
    expect(await screen.findByText(/what due date/i)).toBeInTheDocument();
    expect(screen.getAllByRole('img', { name: /flowchart preview/i })).toHaveLength(1);

    sendMessage('tomorrow');
    await waitFor(() => expect(onCreateTask).toHaveBeenCalledTimes(1));
    expect(onCreateTask).toHaveBeenCalledWith(
      expect.objectContaining({
        title: 'Launch plan',
        flowchart: expect.objectContaining({
          nodes: [
            expect.objectContaining({ label: 'Research' }),
            expect.objectContaining({ label: 'Design' }),
            expect.objectContaining({ label: 'Build' }),
          ],
        }),
      })
    );
    expect(mockedAnswer).not.toHaveBeenCalled();
  });

  it('shows a process graph for the last task created in chat', async () => {
    const onCreateTask = vi.fn();
    const onUpdateTask = vi.fn();
    render(<ChatLabPage onCreateTask={onCreateTask} onUpdateTask={onUpdateTask} />);
    await screen.findByText('Start a conversation');

    sendMessage('create a task to review payroll tomorrow');
    expect(await screen.findByText(/would you like a flowchart/i)).toBeInTheDocument();
    sendMessage('no');
    await waitFor(() => expect(onCreateTask).toHaveBeenCalledTimes(1));

    sendMessage('ok can u show me a process graph for this');
    expect(await screen.findByText(/process graph for "review payroll"/i)).toBeInTheDocument();
    expect(screen.getAllByRole('img', { name: /flowchart preview/i })).toHaveLength(1);
    expect(onUpdateTask).toHaveBeenCalledWith(
      expect.objectContaining({
        title: 'review payroll',
        flowchart: expect.objectContaining({ nodes: expect.any(Array) }),
      })
    );
    expect(mockedAnswer).not.toHaveBeenCalled();
    expect(mockedLocal).not.toHaveBeenCalled();
  });

  it('loads a text document and answers questions from it', async () => {
    mockedAvailable.mockResolvedValue(false);
    const actual = await vi.importActual<typeof import('./chatBrain')>('./chatBrain');
    mockedLocal.mockImplementation((question, context) =>
      actual.answerLocally(question, context)
    );
    render(<ChatLabPage />);
    await screen.findByText('Start a conversation');

    const file = new File(
      ['The project deadline is 2026-12-01.'],
      'plan.txt',
      { type: 'text/plain' }
    );
    fireEvent.change(screen.getByLabelText('Choose document'), {
      target: { files: [file] },
    });

    expect(await screen.findByText(/Loaded "plan.txt"/i)).toBeInTheDocument();
    expect(screen.getByText(/Attached "plan.txt"/i)).toBeInTheDocument();
    expect(screen.getAllByText('plan.txt').length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/Text · .* · text extracted/i)).toBeInTheDocument();
    expect(readThreads('demo-user-uid')[0].document).toEqual(
      expect.objectContaining({ name: 'plan.txt' })
    );
    expect(readThreads('demo-user-uid')[0].messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          role: 'user',
          text: 'Attached "plan.txt"',
          attachment: expect.objectContaining({ name: 'plan.txt' }),
        }),
      ])
    );

    sendMessage('what is the project deadline?');
    expect(
      await screen.findByText(/From "plan.txt": The project deadline is 2026-12-01/i)
    ).toBeInTheDocument();
    expect(mockedExtract).not.toHaveBeenCalled();
    expect(mockedAnswer).not.toHaveBeenCalled();
  });

  it('uses Python extraction for PDF documents and rejects images', async () => {
    render(<ChatLabPage />);
    await screen.findByText('Start a conversation');

    const pdf = new File(['%PDF-1.4'], 'report.pdf', { type: 'application/pdf' });
    fireEvent.change(screen.getByLabelText('Choose document'), {
      target: { files: [pdf] },
    });
    expect(await screen.findByText(/Loaded "report.pdf"/i)).toBeInTheDocument();
    expect(mockedExtract).toHaveBeenCalledTimes(1);
    expect(readThreads('demo-user-uid')[0].document?.text).toContain('2026-12-01');

    const image = new File(['image-bytes'], 'photo.png', { type: 'image/png' });
    fireEvent.change(screen.getByLabelText('Choose document'), {
      target: { files: [image] },
    });
    expect(await screen.findByRole('alert')).toHaveTextContent(/Images are not supported/i);
    expect(readThreads('demo-user-uid')[0].document?.name).toBe('report.pdf');
    expect(mockedExtract).toHaveBeenCalledTimes(1);
  });
});
