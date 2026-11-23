import { describe, it, expect } from 'vitest';
import { localFallbackAnswer } from './localFallback';

describe('localFallbackAnswer', () => {
  it('evaluates basic arithmetic safely', () => {
    expect(localFallbackAnswer('what is 2+2')).toBe('2+2 = 4');
    expect(localFallbackAnswer('(3+5)*2')).toBe('(3+5)*2 = 16');
    expect(localFallbackAnswer('10/4')).toBe('10/4 = 2.5');
  });

  it('explains integration concepts and computes simple powers', () => {
    const concept = localFallbackAnswer('how is integration happen');
    expect(concept).toMatch(/adding up many tiny pieces/i);
    expect(concept).not.toMatch(/a\*e\*h/);

    const integral = localFallbackAnswer('what is integration of x^2');
    expect(integral).toContain('x^3/3 + C');
    expect(integral).toMatch(/reverse of differentiation/i);

    expect(localFallbackAnswer('what is derivative of x^2')).toContain('2x');
  });

  it('ignores conversational filler before answering', () => {
    expect(localFallbackAnswer('ok so what is 2+2')).toBe('2+2 = 4');
    expect(localFallbackAnswer('ok who is president of Spain')).toContain('King Felipe VI');
  });

  it('answers common world facts', () => {
    expect(localFallbackAnswer('Tell me about India')).toContain('Droupadi Murmu');
    expect(localFallbackAnswer('who is president of America')).toBe(
      'The current President of the United States is Donald Trump.'
    );
    expect(localFallbackAnswer('president of Spain')).toContain('no president');
  });

  it('answers contextual questions without a Wikipedia article', () => {
    expect(localFallbackAnswer('Tell me about capital of india')).toContain('New Delhi');
    expect(localFallbackAnswer('which oart are located in NCT')).toContain('11 districts');
    expect(localFallbackAnswer('so NCR and NCT both are different')).toContain('union territory');
    expect(localFallbackAnswer('ok tell me about best city to explore in this world')).toContain(
      'depends on what you enjoy'
    );
    expect(localFallbackAnswer('so new york is best city in world')).toContain('subjective');
  });

  it('answers semantic math questions about named objects', () => {
    expect(localFallbackAnswer('what is eigenvalue of null matrix')).toContain('only eigenvalue');
    expect(localFallbackAnswer('ok but what is eigenvalues of null matrix')).toContain('multiplicity n');
    expect(localFallbackAnswer('what is eigenvalue of nilpotent matrix')).toContain(
      'All eigenvalues'
    );
    expect(
      localFallbackAnswer('tell me determinant of a skew symmetric matrix which is in odd shape')
    ).toContain('determinant of an odd-order skew-symmetric matrix is 0');
    expect(
      localFallbackAnswer('determinant of a skew symmetric matrix which is in dd shape')
    ).toContain('odd-order');
  });

  it('answers both halves of a combined math question', () => {
    const answer = localFallbackAnswer(
      'what is eigenvalue of nilpotent matrix and also tell me determinant of a skew symmetric matrix which is in dd shape'
    );
    expect(answer).toContain('nilpotent matrix are 0');
    expect(answer).toContain('odd-order skew-symmetric matrix is 0');
  });

  it('returns null for questions it cannot handle', () => {
    expect(localFallbackAnswer('What patterns predict completed?')).toBeNull();
    expect(localFallbackAnswer('tell me about quantum tunnelling')).toBeNull();
  });
});
