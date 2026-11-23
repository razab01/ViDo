import { describe, expect, it } from 'vitest';
import {
  documentAnswer,
  loadChatDocument,
  validateDocumentFile,
  type ChatDocument,
} from './documentAnswer';

const document: ChatDocument = {
  id: 'doc-1',
  name: 'handbook.txt',
  text:
    'Project overview. The retry policy uses three attempts with exponential backoff.\n\n' +
    'Deployment notes. The production freeze starts on 2026-12-01 and ends on 2026-12-05.',
  loadedAt: 1,
  size: 240,
  mime: 'text/plain',
};

describe('Chat Lab documents', () => {
  it('accepts text documents and blocks images and legacy Word files', () => {
    const text = new File(['hello'], 'notes.md', { type: 'text/markdown' });
    const image = new File(['image'], 'photo.png', { type: 'image/png' });
    const legacy = new File(['word'], 'old.doc', { type: 'application/msword' });

    expect(validateDocumentFile(text)).toBeNull();
    expect(validateDocumentFile(image)).toMatch(/Images are not supported/i);
    expect(validateDocumentFile(legacy)).toMatch(/Save the file as \.docx/i);
  });

  it('reads a plain-text document locally', async () => {
    const file = new File(['Deadline: 2026-12-01'], 'dates.txt', { type: 'text/plain' });
    const loaded = await loadChatDocument(file);

    expect(loaded.name).toBe('dates.txt');
    expect(loaded.text).toBe('Deadline: 2026-12-01');
    expect(loaded.truncated).toBe(false);
  });

  it('answers from the most relevant document chunk', () => {
    const answer = documentAnswer('when does the production freeze start?', document);
    expect(answer).toMatch(/From "handbook.txt"/i);
    expect(answer).toContain('production freeze starts on 2026-12-01');
  });

  it('returns null when the document has no matching text', () => {
    expect(
      documentAnswer('what is the capital of France?', {
        ...document,
        text: 'Retry policy. Three attempts with exponential backoff.',
      })
    ).toBeNull();
  });
});
