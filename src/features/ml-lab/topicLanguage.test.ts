import { describe, expect, it } from 'vitest';
import { describeSubject, subjectKeywords, topicPhrase } from './topicLanguage';

describe('offline topic language', () => {
  it('explains every meaningful word in a multi-word task title', () => {
    const boating = describeSubject('Boating with sibling');
    expect(boating).toMatch(/^Boating with sibling is spending time on the water in a boat with a brother or sister\.$/);

    const report = describeSubject('financial report for this quarter');
    expect(report).toMatch(/clear summary of what happened/i);
    expect(report).toMatch(/money, budgets and accounts/i);
    expect(report).toMatch(/three-month period/i);
    expect(report).not.toMatch(/status report/i);
  });

  it('works for topics that are not in the bundled articles', () => {
    const answer = describeSubject('Urgent quarterly invoice review');
    expect(answer).toMatch(/request for payment/i);
    expect(answer).toMatch(/needing attention immediately/i);
    expect(answer).toMatch(/three-month period/i);
  });

  it('lets a specific subject word lead a longer task title', () => {
    const answer = describeSubject('Schedule for calculus class');
    expect(answer).toMatch(/the branch of mathematics that studies continuous change/i);
    expect(answer).not.toMatch(/^Schedule for calculus class is choosing the time for something\.$/);
  });

  it('returns nothing when no word is understood', () => {
    expect(describeSubject('xyzzy qwerty')).toBe('');
    expect(describeSubject('')).toBe('');
  });

  it('extracts search topics from questions', () => {
    expect(topicPhrase('What is quantum computing?')).toBe('quantum computing');
    expect(topicPhrase('Tell me about boating')).toBe('boating');
    expect(subjectKeywords('Boating with sibling')).toContain('boat');
  });
});
