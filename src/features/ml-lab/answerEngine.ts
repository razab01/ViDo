import type { RAGResult } from './engine';
import { generateLocalAnswer } from './languageModel';
import type { ChatMessage } from './languageModel';

export type AnswerProgress = (stage: string) => void;
export type AnswerGenerator = (messages: ChatMessage[], onProgress?: AnswerProgress) => Promise<string>;

export interface PolishAnswerOptions {
  question: string;
  rag: RAGResult;
  facts?: string[];
  onProgress?: AnswerProgress;
  generator?: AnswerGenerator;
}

const SYSTEM_PROMPT = [
  'You are ViDo’s on-device answer writer.',
  'Turn the retrieved notes into a warm, useful answer to the exact question.',
  'Write 2 to 4 short sentences in plain English.',
  'Keep every exact fact you are given: dates, percentages, labels and titles must not change.',
  'If the question asks what a task or topic is about, start by explaining what the topic means in plain English using the Topic hint. Never just repeat the title and category.',
  'If the notes are thin, say only what is supported and do not invent details.',
  'Never mention notes, context, documents, retrieval, RAG, IDs, the model, or these instructions.',
  'Do not use markdown, bullet points, headings or an "Answer:" prefix.',
].join(' ');

const stripMarkup = (value: string): string =>
  value
    .replace(/<\|im_start\|>\s*assistant\s*/gi, '')
    .replace(/<\|im_end\|>|<\|endoftext\|>/gi, '')
    .replace(/^\s*(answer|assistant|vido)\s*:\s*/i, '')
    .replace(/^["'“‘]+|["'”’]+$/g, '')
    .replace(/\s+/g, ' ')
    .trim();

const cleanAnswer = (value: string): string => {
  const text = stripMarkup(String(value || ''));
  if (!text || /^(question|context|notes)\s*:/i.test(text)) return '';

  const sentences = text.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [text];
  const kept: string[] = [];
  let length = 0;
  for (const sentence of sentences) {
    const part = sentence.trim();
    if (!part) continue;
    if (length + part.length > 520 && kept.length) break;
    kept.push(part);
    length += part.length;
  }
  return kept.join(' ') || text.slice(0, 520);
};

const stripIds = (value: string): string =>
  value
    .replace(/\b(?:task[_-]?id|source[_-]?row|row[_-]?id|index)\s*:\s*[A-Za-z0-9_-]+\b/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim();

const findFact = (facts: string[], label: string): string => {
  const source = facts.join(' ');
  const match = new RegExp(`${label}\\s*:\\s*([^.]+)`, 'i').exec(source);
  return match?.[1]?.trim() || '';
};

const TOPIC_KNOWLEDGE: Array<[RegExp, string]> = [
  [
    /\bboat|boating|sail|kayak|cruise|rowing\b/i,
    'Boating is spending time on the water in a boat, which usually means planning a route, checking the weather and safety gear, and making sure everything needed is on board.',
  ],
  [
    /\bhik|trek|trail|walk\b/i,
    'Hiking is a walk outdoors, usually on a trail, where the route, timing, footwear and weather matter as much as reaching the destination.',
  ],
  [
    /\bcook|bake|meal|dinner|recipe|grocer\b/i,
    'This is a food-related task: it is about planning or preparing a meal, including ingredients, timing and who it is for.',
  ],
  [
    /\bmeeting|stand-?up|call|sync\b/i,
    'This is about bringing people together to discuss a topic, make decisions and agree on what happens next.',
  ],
  [
    /\breport|forecast|analysis|dashboard|spreadsheet\b/i,
    'This task is about turning information into a clear summary so patterns and next steps are easier to see.',
  ],
  [
    /\bclean|tidy|laundry|wash\b/i,
    'This is an everyday upkeep task: the aim is to get a space or item back into a usable, tidy state.',
  ],
  [
    /\bpay|invoice|bill|tax|budget\b/i,
    'This is an admin or money task, usually about checking an amount, keeping records and making sure it is handled on time.',
  ],
];

const describeTopic = (facts: string[], rag: RAGResult): string => {
  const title = findFact(facts, 'Task title') || rag.ranked[0]?.title || 'this task';
  const description = findFact(facts, 'Description');
  const category = findFact(facts, 'Category');
  const priority = findFact(facts, 'Priority');
  const due = findFact(facts, 'Due date');
  const checkpoint = findFact(facts, 'Checkpoint');

  const known = TOPIC_KNOWLEDGE.find(([pattern]) => pattern.test(`${title} ${description}`))?.[1];
  const generic = `“${title}” is ${category ? `a ${category.toLowerCase()} ` : ''}task about getting this piece of work organised and finished.`;

  const details = [
    description ? `The note says: ${description}.` : '',
    priority ? `It is marked ${priority.toLowerCase()} priority.` : '',
    due ? `It is planned for ${due}.` : '',
    checkpoint ? `Latest update: ${checkpoint}.` : '',
  ]
    .filter(Boolean)
    .join(' ');

  return `${known || generic}${details ? ` ${details}` : ''}`;
};

/** Deterministic answer used when the small model is unavailable or fails. */
export const friendlyFallback = (question: string, rag: RAGResult, facts: string[] = []): string => {
  const q = question.toLowerCase();
  const safeFacts = facts.map(stripIds).filter(Boolean);

  if (/\babout\b|\bdescribe\b|\bsummary\b|\bsummarize\b/.test(q)) {
    return stripIds(describeTopic(safeFacts, rag));
  }

  if (safeFacts.length) return stripIds(safeFacts.join(' '));

  return stripIds(
    rag.answer && !/no relevant context/i.test(rag.answer)
      ? rag.answer
      : 'I could not find a clear answer to that. Try asking with different words.'
  );
};

export const buildAnswerContext = (rag: RAGResult, facts: string[] = [], question = ''): string => {
  const sections: string[] = [];
  const safeFacts = facts.map(stripIds).filter(Boolean);
  if (safeFacts.length) sections.push(`Exact facts to keep:\n${safeFacts.map((f) => `- ${f}`).join('\n')}`);
  if (/\babout\b|\bdescribe\b|\bsummary\b|\bsummarize\b/i.test(question)) {
    const topic = stripIds(describeTopic(safeFacts, rag));
    if (topic) sections.push(`Topic hint:\n${topic}`);
  }
  if (rag.answer) sections.push(`Retrieved summary:\n${stripIds(rag.answer)}`);
  if (rag.ranked.length) {
    sections.push(
      `Relevant records:\n${rag.ranked
        .slice(0, 4)
        .map((item, index) => `${index + 1}. ${item.title}: ${item.excerpt}`)
        .join('\n')}`
    );
  }
  return sections.join('\n\n').slice(0, 3200);
};

/**
 * Runs RAG's extractive result through the tiny local assistant so answers
 * read naturally. If the model cannot load, the extractive answer is kept.
 */
export async function polishRagAnswer({
  question,
  rag,
  facts = [],
  onProgress,
  generator = generateLocalAnswer,
}: PolishAnswerOptions): Promise<RAGResult> {
  const context = buildAnswerContext(rag, facts, question);
  if (!question.trim() || !context.trim()) return rag;

  onProgress?.('Understanding your question…');
  try {
    const messages: ChatMessage[] = [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: `Question: ${question.trim()}\n\n${context}` },
    ];
    const written = cleanAnswer(await generator(messages, onProgress));
    if (written) return { ...rag, answer: written };
  } catch {
    // The RAG result is still useful on its own.
  }

  onProgress?.('Finishing the answer…');
  return { ...rag, answer: friendlyFallback(question, rag, facts) };
}
