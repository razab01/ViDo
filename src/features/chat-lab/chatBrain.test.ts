import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  answerLocally,
  datasetAnswer,
  extractSubject,
  resolveQuestion,
  wikipediaAnswer,
} from './chatBrain';
import { lookupTopic, lookupTopicWord } from '../ml-lab/topicLookup';
import { profileData } from '../ml-lab/engine';

vi.mock('../ml-lab/topicLookup', () => ({
  lookupTopic: vi.fn(),
  lookupTopicWord: vi.fn(),
}));

const mockedLookup = vi.mocked(lookupTopic);
const mockedLookupWord = vi.mocked(lookupTopicWord);

const dataset = {
  cols: ['priority', 'has_desc', 'task_id', 'task_text', 'completed', 'due_date'],
  rows: [
    { priority: 'low', has_desc: '1', task_id: 'T09', task_text: 'Morning walk', completed: 'no', due_date: '2026-11-23' },
    { priority: 'high', has_desc: '0', task_id: 'T03', task_text: 'Review budget', completed: 'yes', due_date: '2026-11-24' },
    { priority: 'medium', has_desc: '1', task_id: 'T15', task_text: 'Write report', completed: 'yes', due_date: '2026-11-25' },
  ],
  target: 'completed',
};

const context = {
  dataset,
  profile: profileData(dataset.cols, dataset.rows),
  suggestions: [],
};

describe('chatBrain', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedLookupWord.mockResolvedValue(null);
  });

  it('extracts readable subjects from common question shapes', () => {
    expect(extractSubject('who is adam smith ?')).toBe('Adam Smith');
    expect(extractSubject('Tell me about India.')).toBe('India');
    expect(extractSubject('what did Adam Smith write?')).toBe('Adam Smith');
    expect(extractSubject('what did he write?')).toBeNull();
    expect(extractSubject('ok what is game theory')).toBe('Game Theory');
    expect(extractSubject('ok so what is nash equilibrium')).toBe('Nash Equilibrium');
  });

  it('resolves pronouns against the last extracted subject', () => {
    const history = [
      { question: 'who is adam smith ?', answer: 'Adam Smith was a Scottish economist.' },
    ];
    expect(resolveQuestion('what did he write?', history)).toBe('what did Adam Smith write?');
    expect(resolveQuestion('what did his friend write?', history)).toBe(
      'what did Adam Smith\'s friend write?'
    );
    expect(resolveQuestion('what is 2+2', history)).toBe('what is 2+2');
  });

  it('answers completed-task questions from actual rows', () => {
    const answer = datasetAnswer('which task are mostly completed?', context);
    expect(answer).toContain('Review budget');
    expect(answer).toContain('Write report');
    expect(answer).not.toContain('Review budget — Review budget');
    expect(answer).not.toContain('closest records');
  });

  it('answers exact task IDs, including quoted phrasing with typos', () => {
    const plain = datasetAnswer('T09', context);
    const quoted = datasetAnswer('"T09" tell me about thus task id', context);
    expect(plain).toContain('T09 — Morning walk.');
    expect(plain).toContain('Status: no.');
    expect(plain).toContain('Due: 2026-11-23.');
    expect(quoted).toBe(plain);
  });

  it('answers high-priority questions from actual rows', () => {
    expect(datasetAnswer('which tasks are high priority?', context)).toContain('Review budget');
  });

  it('handles maths and curated world facts without any service', async () => {
    expect(await answerLocally('what is 2+2', context)).toBe('2+2 = 4');
    expect(await answerLocally('who is president of spain', context)).toContain('King Felipe VI');
  });

  it('explains math concepts and simple integrals instead of guessing symbols', async () => {
    const concept = await answerLocally('how is integration happen', context);
    expect(concept).toMatch(/adding up many tiny pieces/i);
    expect(concept).not.toMatch(/a\*e\*h/);

    const integral = await answerLocally('what is integration of x^2', context);
    expect(integral).toContain('x^3/3 + C');
    expect(integral).toMatch(/reverse of differentiation/i);
  });

  it('strips conversational filler before looking up a topic', async () => {
    mockedLookup.mockResolvedValueOnce({
      title: 'Nash equilibrium',
      extract: 'In game theory, a Nash equilibrium is a situation where no player gains by changing strategy alone.',
    });

    const answer = await answerLocally('ok so what is nash equilibrium', context);
    expect(answer).toContain('no player gains');
    expect(mockedLookup).toHaveBeenCalledWith('Nash Equilibrium');
  });

  it('uses Wikipedia for general topics and keeps pronoun answers locally reliable', async () => {
    mockedLookup.mockResolvedValueOnce({
      title: 'Quantum computing',
      extract: 'Quantum computing uses quantum-mechanical phenomena such as superposition and entanglement.',
    });

    expect(await wikipediaAnswer('tell me about quantum computing')).toContain('superposition');
    const resolved = resolveQuestion('what did he write?', [
      { question: 'who is adam smith?', answer: 'Adam Smith was a Scottish economist and philosopher.' },
    ]);
    expect(resolved).toBe('what did Adam Smith write?');
    expect(await answerLocally(resolved, context)).toContain('The Wealth of Nations');
    expect(mockedLookup).not.toHaveBeenCalledWith('Adam Smith');
  });
});
