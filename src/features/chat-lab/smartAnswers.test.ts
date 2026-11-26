import { describe, expect, it } from 'vitest';
import { greetingAnswer } from './smartAnswers';

describe('greetingAnswer', () => {
  it('answers a bare greeting warmly', () => {
    expect(greetingAnswer('hi')).toContain('Hi there');
    expect(greetingAnswer('hi')).toContain('How can I help today');
    expect(greetingAnswer('hello there')).toContain('Good to see you');
  });

  it('answers greeting plus small talk conversationally', () => {
    for (const phrase of ['hi how are you', 'how are you', "how's it going", 'how do you do']) {
      expect(greetingAnswer(phrase)).toContain('doing well');
    }
    expect(greetingAnswer("what's up")).toContain('Good to see you');
    expect(greetingAnswer('I just want to chat')).toContain('Good to see you');
    expect(greetingAnswer('I wanna talk')).toContain('Good to see you');
  });

  it('answers slightly misspelled greetings politely', () => {
    for (const phrase of ['hii', 'helo', 'heyyyya', 'gud morning', 'how r u']) {
      expect(greetingAnswer(phrase)).toBeTruthy();
    }
    expect(greetingAnswer('how r u')).toContain('doing well');
  });

  it('does not swallow a real question after a greeting', () => {
    expect(greetingAnswer('hi what is federalism')).toBeNull();
    expect(greetingAnswer('hey book this plan')).toBeNull();
  });
});
