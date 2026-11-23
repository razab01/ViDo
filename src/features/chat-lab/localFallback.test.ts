import { describe, it, expect } from 'vitest';
import { localFallbackAnswer } from './localFallback';

describe('localFallbackAnswer', () => {
  it('evaluates basic arithmetic safely', () => {
    expect(localFallbackAnswer('what is 2+2')).toBe('2+2 = 4');
    expect(localFallbackAnswer('(3+5)*2')).toBe('(3+5)*2 = 16');
    expect(localFallbackAnswer('10/4')).toBe('10/4 = 2.5');
  });

  it('answers common world facts', () => {
    expect(localFallbackAnswer('Tell me about India')).toContain('Droupadi Murmu');
    expect(localFallbackAnswer('who is president of America')).toBe(
      'The current President of the United States is Donald Trump.'
    );
    expect(localFallbackAnswer('president of Spain')).toContain('no president');
  });

  it('returns null for questions it cannot handle', () => {
    expect(localFallbackAnswer('What patterns predict completed?')).toBeNull();
    expect(localFallbackAnswer('tell me about quantum tunnelling')).toBeNull();
  });
});
