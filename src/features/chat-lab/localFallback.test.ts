import { describe, it, expect } from 'vitest';
import { localFallbackAnswer, normalizeQuestion } from './localFallback';

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

  it('counts digit arrangements without falling back to dataset records', () => {
    const answer = localFallbackAnswer('how many 5 digit number can be formed by 1,2,3,4,5');
    expect(answer).toContain('120');
    expect(answer).toContain('5 × 4 × 3 × 2 × 1');
    expect(answer).toContain('3125');

    const repeat = localFallbackAnswer(
      'how many 5 digit numbers can be formed from 1,2,3,4,5 with repetition allowed'
    );
    expect(repeat).toContain('3125');
    expect(repeat).toContain('digits may repeat');

    const zero = localFallbackAnswer(
      'how many 4 digit numbers can be formed from 0,1,2,3 without repetition'
    );
    expect(zero).toContain('18');
    expect(zero).toContain('cannot start with 0');
  });

  it('answers coin probability and even digit counts offline', () => {
    const probability = localFallbackAnswer(
      'what is the probability of getting head if we toss coin single time'
    );
    expect(probability).toContain('1/2');
    expect(probability).toContain('50%');
    expect(probability).toContain('fair coin');

    const twoTosses = localFallbackAnswer('probability of at least one head in 2 coin tosses');
    expect(twoTosses).toContain('3/4');

    const even = localFallbackAnswer(
      'how many 5 digit even number can be formed by 1,2,3,4,5'
    );
    expect(even).toContain('48');
    expect(even).toContain('2 × 4 × 3 × 2 × 1');
    expect(even).toContain('1250');
  });

  it('answers exact head counts and statistics definitions offline', () => {
    const exactlyOne = localFallbackAnswer(
      'what is the probability of getting a exactly single head by tossing coin two time'
    );
    expect(exactlyOne).toContain('1/2');
    expect(exactlyOne).toContain('exactly 1 head');
    expect(exactlyOne).not.toContain('at least one head');

    const twoHeads = localFallbackAnswer(
      'what is the probability of getting two heads in two tosses'
    );
    expect(twoHeads).toContain('1/4');

    const overview = localFallbackAnswer('what is probability and statistics');
    expect(overview).toContain('sample space');
    expect(overview).toContain('central limit theorem');

    const mode = localFallbackAnswer('what is mode');
    expect(mode).toContain('occurs most often');
    expect(mode).not.toContain('electromagnetic');
  });

  it('greets warmly and recovers from common typos offline', () => {
    expect(localFallbackAnswer('hi')).toContain('How can I help today');
    expect(normalizeQuestion('hi')).toBe('hi');
    expect(normalizeQuestion('hi what is federalism')).toBe('what is federalism');
    for (const greeting of [
      'hey',
      'hello there',
      'good morning',
      'helo',
      'gud morning',
    ]) {
      const answer = localFallbackAnswer(greeting);
      expect(answer).toContain('Good to see you');
      expect(answer).toContain('How can I help today');
    }
    expect(localFallbackAnswer('how r u')).toContain('doing well');
    expect(localFallbackAnswer('I just want to chat')).toContain('Good to see you');

    const sectors = localFallbackAnswer(
      'what is primary and secondry and teritiary sector?'
    );
    expect(sectors).toContain('Primary extracts raw materials');
    expect(sectors).toContain('Tertiary provides services');

    const bfs = localFallbackAnswer('what is bredth first search');
    expect(bfs).toContain('Breadth-first search');
    expect(bfs).toContain('queue');
    expect(bfs).not.toContain('Maine');

    const aStar = localFallbackAnswer('what is A* Algorithm');
    expect(aStar).toContain('f(n) = g(n) + h(n)');
    expect(aStar).toContain('pathfinding');
  });

  it('answers conversational knowledge questions without a Wikipedia article', () => {
    const federalism = localFallbackAnswer('hi what is federalism');
    expect(federalism).toContain('power is divided');
    expect(federalism).toContain('regional');

    const brics = localFallbackAnswer('tell me about BRICS?');
    expect(brics).toContain('South Africa joined in 2010');
    expect(brics).toContain('multipolar');

    const identity = localFallbackAnswer('tell me about who are you');
    expect(identity).toContain('ViDo’s chat assistant');
    expect(identity).toContain('flowcharts');

    const barter = localFallbackAnswer('is barter exchange still used somewhere??');
    expect(barter).toMatch(/^Yes/);
    expect(barter).toContain('double coincidence of wants');

    const java = localFallbackAnswer('What is java?');
    expect(java).toContain('programming language');
    expect(java).toContain('JVM');
    expect(localFallbackAnswer('What is java island?')).toBeNull();

    const wora = localFallbackAnswer('Ok why it is (WORA)');
    expect(wora).toContain('Write Once, Run Anywhere');
    expect(wora).toContain('bytecode');
    expect(wora).toContain('JVM');

    const schur = localFallbackAnswer('tell me about schur component.');
    expect(schur).toContain('Schur complement');
    expect(schur).toContain('D - C A⁻¹ B');
    expect(localFallbackAnswer('ok what is formula for schur component')).toContain(
      'det(M) = det(A) · det(S)'
    );
  });

  it('answers procedural database questions directly', () => {
    const create = localFallbackAnswer('how to create a new database');
    expect(create).toContain('CREATE DATABASE');
    expect(create).toContain('SQLite');

    const column = localFallbackAnswer('how to create a new column in database');
    expect(column).toContain('ALTER TABLE');
    expect(column).toContain('email VARCHAR(255)');

    const sqlalchemy = localFallbackAnswer('what is sqlalchemy');
    expect(sqlalchemy).toContain('object-relational mapper');
    expect(sqlalchemy).toContain('create_engine');
  });

  it('explains the Python if statement directly', () => {
    const answer = localFallbackAnswer('what is if command in python');
    expect(answer).toContain('conditional statement');
    expect(answer).toContain('elif');
    expect(answer).toContain("print('positive')");
  });

  it('returns null for questions it cannot handle', () => {
    expect(localFallbackAnswer('What patterns predict completed?')).toBeNull();
    expect(localFallbackAnswer('tell me about quantum tunnelling')).toBeNull();
  });
});
