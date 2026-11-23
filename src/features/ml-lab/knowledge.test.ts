import { describe, expect, it } from 'vitest';
import { profileData, runRAG, USER_RAG_MODE } from './engine';
import { findKnowledge, KNOWLEDGE_DOCS, KNOWLEDGE_ENTRIES } from './knowledge';

describe('bundled knowledge', () => {
  it('ships a broad offline library of articles and book notes', () => {
    expect(KNOWLEDGE_ENTRIES.length).toBeGreaterThan(40);
    expect(KNOWLEDGE_DOCS.length).toBe(KNOWLEDGE_ENTRIES.length);
    expect(KNOWLEDGE_DOCS.some((doc) => /Getting Things Done/i.test(doc.title))).toBe(true);
    expect(KNOWLEDGE_DOCS.some((doc) => /Boating/i.test(doc.title))).toBe(true);
  });

  it('matches a subject to the right article', () => {
    expect(findKnowledge('Boating')?.id).toBe('kb-boating');
    expect(findKnowledge('Deep Work by Cal Newport')?.topic).toBe('books');
    expect(findKnowledge('')).toBeNull();
  });

  it('does not match an article from one shared, generic word', () => {
    expect(findKnowledge('financial report for this quarter')).toBeNull();
    expect(findKnowledge('What is this task about?')).toBeNull();
    expect(findKnowledge('probability confidence prediction chance')?.id).toBe('kb-confidence');
  });

  it('answers from graph retrieval with no uploaded dataset', () => {
    const profile = profileData([], []);
    const result = runRAG('What is prioritisation?', USER_RAG_MODE, KNOWLEDGE_DOCS, profile, []);

    expect(result.mode).toBe('graph');
    expect(result.ranked.length).toBeGreaterThan(0);
    expect(result.ranked[0].title).toMatch(/prioritisation/i);
  });
});
