import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ChatLabPage } from './ChatLabPage';
import {
  extractDocumentWithPython,
  isPythonAvailable,
  answerWithPython,
  transcribeAudioWithPython,
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
  transcribeAudioWithPython: vi.fn(),
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
const mockedTranscribe = vi.mocked(transcribeAudioWithPython);

describe('ChatLabPage', () => {
  let originalMediaDevices: PropertyDescriptor | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    resetChatStore();
    originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices');
    mockedAvailable.mockResolvedValue(true);
    mockedAnswer.mockResolvedValue('Spain has no president; the head of state is King Felipe VI.');
    mockedLocal.mockResolvedValue('India is a country in South Asia.');
    mockedExtract.mockResolvedValue({
      name: 'report.pdf',
      text: 'The project deadline is 2026-12-01.',
      truncated: false,
    });
    mockedTranscribe.mockResolvedValue({
      text: 'who is president of spain',
      model: 'tiny',
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    if (originalMediaDevices) {
      Object.defineProperty(navigator, 'mediaDevices', originalMediaDevices);
    } else {
      Reflect.deleteProperty(navigator, 'mediaDevices');
    }
  });

  const sendMessage = (value: string) => {
    fireEvent.change(screen.getByPlaceholderText(/chat, book a task/i), { target: { value } });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));
  };

  it('renders the empty state with booking prompts', async () => {
    render(<ChatLabPage />);
    expect(await screen.findByText('Start a conversation')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/chat, book a task/i)).toBeInTheDocument();
    expect(screen.queryByText(/vido \/ workspace companion/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/offline-ready/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/think out loud\. turn it into action\./i)).not.toBeInTheDocument();
    expect(screen.queryByText(/natural conversation/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/task booking/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/attachment planning/i)).not.toBeInTheDocument();
    const prompts = screen
      .getAllByRole('button')
      .filter((button) =>
        /^(Book a task for tomorrow at 9am|Book the tasks in my attachment)$/.test(
          button.textContent ?? ''
        )
      );
    expect(prompts).toHaveLength(2);
    expect(screen.queryByRole('button', { name: 'Hey, I just want to chat' })).not.toBeInTheDocument();
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

  it('does not drop a message sent immediately after render', async () => {
    render(<ChatLabPage />);
    sendMessage('who is president of spain');

    await waitFor(() => {
      expect(
        screen.getByText('Spain has no president; the head of state is King Felipe VI.')
      ).toBeInTheDocument();
    });
    expect(mockedAnswer).toHaveBeenCalledTimes(1);
  });

  it('uses the local answer when Python is unavailable', async () => {
    mockedAvailable.mockResolvedValue(false);
    render(<ChatLabPage />);
    await screen.findByText('Start a conversation');
    sendMessage('Tell me about India');

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
    sendMessage('Tell me about India');
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

  it('stages a document until the chat message is sent, then answers from it', async () => {
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

    expect(await screen.findByText(/ready to send/i)).toBeInTheDocument();
    expect(screen.queryByText(/Loaded "plan.txt"/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Attached "plan.txt"/i)).not.toBeInTheDocument();
    expect(readThreads('demo-user-uid')[0].document).toBeUndefined();
    expect(readThreads('demo-user-uid')[0].messages).toHaveLength(0);

    sendMessage('what is the project deadline?');
    expect(await screen.findByText(/Loaded "plan.txt"/i)).toBeInTheDocument();
    expect(screen.getByText(/Text · .* · text extracted/i)).toBeInTheDocument();
    expect(readThreads('demo-user-uid')[0].document).toEqual(
      expect.objectContaining({ name: 'plan.txt' })
    );
    expect(readThreads('demo-user-uid')[0].messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          role: 'user',
          text: 'what is the project deadline?',
          attachment: expect.objectContaining({ name: 'plan.txt' }),
        }),
      ])
    );
    expect(
      await screen.findByText(/From "plan.txt": The project deadline is 2026-12-01/i)
    ).toBeInTheDocument();
    expect(mockedExtract).not.toHaveBeenCalled();
    expect(mockedAnswer).not.toHaveBeenCalled();
  });

  it('uses Python extraction for PDF documents and rejects images without posting either', async () => {
    render(<ChatLabPage />);
    await screen.findByText('Start a conversation');

    const pdf = new File(['%PDF-1.4'], 'report.pdf', { type: 'application/pdf' });
    fireEvent.change(screen.getByLabelText('Choose document'), {
      target: { files: [pdf] },
    });
    expect(await screen.findByText(/ready to send/i)).toBeInTheDocument();
    expect(mockedExtract).toHaveBeenCalledTimes(1);
    expect(readThreads('demo-user-uid')[0].document).toBeUndefined();
    expect(screen.queryByText(/Loaded "report.pdf"/i)).not.toBeInTheDocument();

    const image = new File(['image-bytes'], 'photo.png', { type: 'image/png' });
    fireEvent.change(screen.getByLabelText('Choose document'), {
      target: { files: [image] },
    });
    expect(await screen.findByRole('alert')).toHaveTextContent(/Images are not supported/i);
    expect(readThreads('demo-user-uid')[0].document).toBeUndefined();
    expect(mockedExtract).toHaveBeenCalledTimes(1);
  });

  it('books Graph RAG plan items through the task callback', async () => {
    const onCreateTask = vi.fn();
    const dueDate = tomorrowDateString();
    mockedAnswer.mockImplementation(async (_input, onResult) => {
      onResult?.({
        actions: {
          type: 'book_plans',
          tasks: [
            {
              title: 'Draft spec',
              dueDate,
              priority: 'high',
              category: 'work',
            },
          ],
        },
        booking: null,
      });
      return 'I found 1 plan item and I’m booking it now.';
    });

    render(<ChatLabPage onCreateTask={onCreateTask} />);
    await screen.findByText('Start a conversation');
    sendMessage('book this plan');

    expect(await screen.findByText(/All set — I booked "Draft spec"/)).toBeInTheDocument();
    expect(onCreateTask).toHaveBeenCalledTimes(1);
    expect(onCreateTask).toHaveBeenCalledWith(
      expect.objectContaining({
        title: 'Draft spec',
        dueDate,
        priority: 'high',
        category: 'work',
        state: 'active',
      })
    );
    expect(readThreads('demo-user-uid')[0].booking).toBeUndefined();
  });

  it('transcribes a voice note and books a task through the normal chat pipeline', async () => {
    const transcript = 'book a task for morning walk tomorrow at 4am';
    const dueDate = tomorrowDateString();
    const onCreateTask = vi.fn();
    mockedTranscribe.mockResolvedValue({ text: transcript, model: 'tiny' });
    mockedAnswer.mockImplementation(async (input, onResult) => {
      expect(input.question).toBe(transcript);
      onResult?.({
        actions: {
          type: 'book_plans',
          tasks: [{ title: 'morning walk', dueDate, dueTime: '04:00' }],
        },
        booking: null,
      });
      return 'Got it — I’m booking 1 task now.';
    });

    class MockMediaRecorder {
      static isTypeSupported = () => true;
      state: 'inactive' | 'recording' = 'inactive';
      mimeType = 'audio/webm';
      ondataavailable: ((event: { data: Blob }) => void) | null = null;
      onstop: (() => void) | null = null;

      start() {
        this.state = 'recording';
      }

      stop() {
        this.state = 'inactive';
        this.ondataavailable?.({ data: new Blob(['voice'], { type: this.mimeType }) });
        this.onstop?.();
      }
    }
    vi.stubGlobal('MediaRecorder', MockMediaRecorder);
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: () => [{ stop: vi.fn() }],
        }),
      },
    });

    render(<ChatLabPage onCreateTask={onCreateTask} />);
    await screen.findByText('Start a conversation');

    fireEvent.click(screen.getByRole('button', { name: /record voice note/i }));
    const stopButton = await screen.findByRole('button', { name: /stop voice note/i });
    fireEvent.click(stopButton);

    await waitFor(() => expect(mockedTranscribe).toHaveBeenCalledTimes(1));
    expect(mockedTranscribe.mock.calls[0][1]).toMatch(/^voice-.*\.webm$/);
    expect(await screen.findByText(transcript)).toBeInTheDocument();
    expect(await screen.findByText(/All set — I booked "morning walk"/)).toBeInTheDocument();
    expect(onCreateTask).toHaveBeenCalledTimes(1);
    expect(onCreateTask).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'morning walk', dueDate, dueTime: '04:00' })
    );
    expect(screen.queryByText('tiny')).not.toBeInTheDocument();
  });

  it('books a described task instead of answering the attachment', async () => {
    const onCreateTask = vi.fn();
    const dueDate = tomorrowDateString();
    mockedAnswer.mockImplementation(async (input, onResult) => {
      expect(input.documents).toHaveLength(1);
      onResult?.({
        actions: {
          type: 'book_plans',
          tasks: [{ title: 'morning walk', dueDate, dueTime: '04:00' }],
        },
        booking: null,
      });
      return 'Got it — I’m booking 1 task now.';
    });

    render(<ChatLabPage onCreateTask={onCreateTask} />);
    await screen.findByText('Start a conversation');
    fireEvent.change(screen.getByLabelText('Choose document'), {
      target: {
        files: [
          new File(
            ['Meeting Plans Handbook\nMEETING PLAN 01\nProject Kickoff Meeting'],
            'Meeting_Plans_Handbook.txt',
            { type: 'text/plain' }
          ),
        ],
      },
    });
    expect(await screen.findByText(/ready to send/i)).toBeInTheDocument();

    sendMessage('hey book a task for morning walk for tomorrow at 4am');

    expect(await screen.findByText(/All set — I booked "morning walk"/)).toBeInTheDocument();
    expect(screen.queryByText(/Project Kickoff Meeting/)).not.toBeInTheDocument();
    expect(onCreateTask).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'morning walk', dueDate, dueTime: '04:00' })
    );
    expect(mockedLocal).not.toHaveBeenCalled();
  });

  it('asks for a task name without reading the attachment for a bare todo booking', async () => {
    mockedAnswer.mockImplementationOnce(async (_input, onResult) => {
      onResult?.({
        actions: null,
        booking: {
          tasks: [{ title: '', dueDate: tomorrowDateString() }],
          pending: { taskIndex: 0, field: 'title', question: 'What should I call this task?' },
        },
      });
      return 'What should I call this task?';
    });

    render(<ChatLabPage onCreateTask={vi.fn()} />);
    await screen.findByText('Start a conversation');
    fireEvent.change(screen.getByLabelText('Choose document'), {
      target: {
        files: [
          new File(['Meeting Plans Handbook\nMEETING PLAN 01\nProject Kickoff Meeting'], 'handbook.txt', {
            type: 'text/plain',
          }),
        ],
      },
    });
    expect(await screen.findByText(/ready to send/i)).toBeInTheDocument();

    sendMessage('hey wanted to book todo task for tomorrow');

    expect(await screen.findByText('What should I call this task?')).toBeInTheDocument();
    expect(screen.queryByText(/Project Kickoff Meeting/)).not.toBeInTheDocument();
    expect(mockedLocal).not.toHaveBeenCalled();
  });

  it('books an attachment without posting the loaded-document aside', async () => {
    mockedAnswer.mockImplementationOnce(async (_input, onResult) => {
      onResult?.({
        actions: {
          type: 'book_plans',
          tasks: [
            {
              title: 'Monthly Goal Setting & Planning',
              description: 'Define objectives and key results for December',
              startDate: tomorrowDateString(),
              dueDate: tomorrowDateString(),
              dueTime: '09:00',
              priority: 'high',
            },
          ],
        },
        booking: null,
      });
      return 'Got it — I’m booking 1 task now.';
    });

    render(<ChatLabPage onCreateTask={vi.fn()} />);
    await screen.findByText('Start a conversation');
    fireEvent.change(screen.getByLabelText('Choose document'), {
      target: {
        files: [
          new File(
            ['December 2026 | Monthly Task Planner'],
            'Monthly_Task_Planner_December_2026.pdf',
            { type: 'application/pdf' }
          ),
        ],
      },
    });
    expect(await screen.findByText(/ready to send/i)).toBeInTheDocument();

    sendMessage('book the task in my attachment');

    expect(
      await screen.findByText(/All set — I booked "Monthly Goal Setting & Planning"/)
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/Loaded "Monthly_Task_Planner_December_2026\.pdf"/i)
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/one date for all/i)).not.toBeInTheDocument();
  });

  it('persists Graph RAG clarification and completes booking from the reply', async () => {
    const onCreateTask = vi.fn();
    const dueDate = tomorrowDateString();
    mockedAnswer
      .mockImplementationOnce(async (_input, onResult) => {
        onResult?.({
          actions: null,
          booking: {
            tasks: [{ title: 'Draft spec', dueDate: '' }],
            pending: {
              taskIndex: 0,
              field: 'dueDate',
              question: 'What due date should I use for "Draft spec"?',
            },
          },
        });
        return 'What due date should I use for "Draft spec"?';
      })
      .mockImplementationOnce(async (input, onResult) => {
        expect(input.booking).toEqual(
          expect.objectContaining({
            pending: expect.objectContaining({ field: 'dueDate' }),
          })
        );
        onResult?.({
          actions: {
            type: 'book_plans',
            tasks: [{ title: 'Draft spec', dueDate }],
          },
          booking: null,
        });
        return 'Got it.';
      });

    render(<ChatLabPage onCreateTask={onCreateTask} />);
    await screen.findByText('Start a conversation');
    sendMessage('book this plan');
    expect(await screen.findByText(/What due date should I use/)).toBeInTheDocument();
    expect(readThreads('demo-user-uid')[0].booking?.pending?.field).toBe('dueDate');
    expect(onCreateTask).not.toHaveBeenCalled();

    sendMessage(dueDate);
    expect(await screen.findByText(/All set — I booked "Draft spec"/)).toBeInTheDocument();
    expect(onCreateTask).toHaveBeenCalledTimes(1);
    expect(readThreads('demo-user-uid')[0].booking).toBeUndefined();
  });

  it('routes short document booking requests to Graph RAG instead of the local task draft', async () => {
    const booking = {
      tasks: [{ title: 'Project Kickoff Meeting', dueDate: '' }],
      pending: {
        taskIndex: 0,
        taskIndices: [0],
        group: true,
        field: 'dueDate' as const,
        question: 'What due date should I use for all 8 plan items?',
      },
    };
    mockedAnswer
      .mockImplementationOnce(async (input, onResult) => {
        expect(input.documents).toEqual([
          expect.objectContaining({ name: 'Meeting_Plans_Handbook.pdf' }),
        ]);
        onResult?.({ actions: null, booking });
        return booking.pending.question;
      })
      .mockImplementationOnce(async (input, onResult) => {
        expect(input.booking).toEqual(
          expect.objectContaining({
            pending: expect.objectContaining({ field: 'dueDate' }),
          })
        );
        onResult?.({ actions: null, booking });
        return 'The document has no calendar due date. Give one date.';
      });

    render(<ChatLabPage onCreateTask={vi.fn()} />);
    await screen.findByText('Start a conversation');

    const file = new File(
      ['MEETING PLAN 01\nProject Kickoff Meeting\nPURPOSE\nAlign the team.'],
      'Meeting_Plans_Handbook.pdf',
      { type: 'application/pdf' }
    );
    fireEvent.change(screen.getByLabelText('Choose document'), {
      target: { files: [file] },
    });
    expect(await screen.findByText(/ready to send/i)).toBeInTheDocument();
    expect(mockedExtract).toHaveBeenCalledTimes(1);
    expect(screen.queryByText(/Loaded "Meeting_Plans_Handbook.pdf"/i)).not.toBeInTheDocument();

    sendMessage('schedule task');
    expect(await screen.findByText(/all 8 plan items/)).toBeInTheDocument();
    expect(screen.queryByText(/What should I call the task\?/)).not.toBeInTheDocument();
    expect(mockedAnswer).toHaveBeenCalledTimes(1);

    sendMessage('mentioned in the document');
    expect(await screen.findByText(/no calendar due date/i)).toBeInTheDocument();
    expect(screen.queryByText(/What due date should I use\? Try/)).not.toBeInTheDocument();
    expect(mockedAnswer).toHaveBeenCalledTimes(2);
  });
});
