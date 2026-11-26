/**
 * Small local safety net used only when the Python service is unreachable.
 * It covers the common questions that the dataset RAG engine cannot answer,
 * so a missing backend never produces an unrelated task-management snippet.
 */
import { greetingAnswer, smartAnswer } from './smartAnswers';

const WORLD_FACTS: Array<{ keys: string[]; about: string; president: string }> = [
  {
    keys: ['america', 'united states', 'usa', 'u.s.a.', 'u.s.'],
    about:
      'The United States is a federal republic in North America. The current President of the United States is Donald Trump.',
    president: 'The current President of the United States is Donald Trump.',
  },
  {
    keys: ['india', 'bharat'],
    about: 'India is a country in South Asia. The current President of India is Droupadi Murmu.',
    president: 'The current President of India is Droupadi Murmu.',
  },
  {
    keys: ['netherlands', 'holland'],
    about:
      'The Netherlands is a constitutional monarchy, so it has no president. The head of state is King Willem-Alexander and the head of government is Prime Minister Dick Schoof.',
    president:
      'The Netherlands is a constitutional monarchy, so it has no president. The head of state is King Willem-Alexander and the head of government is Prime Minister Dick Schoof.',
  },
  {
    keys: ['spain'],
    about:
      'Spain is a constitutional monarchy, so it has no president as head of state. The head of state is King Felipe VI and the head of government is Prime Minister Pedro Sánchez.',
    president:
      'Spain is a constitutional monarchy, so it has no president as head of state. The head of state is King Felipe VI and the head of government is Prime Minister Pedro Sánchez.',
  },
];

const PEOPLE_FACTS: Array<{ keys: string[]; about: string; works: string }> = [
  {
    keys: ['adam smith'],
    about:
      'Adam Smith was a Scottish economist and philosopher, best known as a founder of modern economics.',
    works:
      'Adam Smith wrote An Inquiry into the Nature and Causes of the Wealth of Nations (1776), commonly called The Wealth of Nations.',
  },
  {
    keys: ['albert einstein'],
    about:
      'Albert Einstein was a German-born physicist who developed the special and general theories of relativity.',
    works:
      'Albert Einstein published papers on the photoelectric effect, Brownian motion, special relativity and mass–energy equivalence.',
  },
];

const PRECEDENCE: Record<string, number> = { '+': 1, '-': 1, '*': 2, '/': 2, '^': 3 };

const FILLER_PREFIX =
  /^(?:ok(?:ay)?|so|and|now|well|yeah|yes|but|please|also|hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening))\b[\s,.!]*/i;
const COMMON_TYPOS: Array<[RegExp, string]> = [
  [/\boart\b/gi, 'part'],
  [/\bthus\b/gi, 'this'],
  [/\bschur\s+components?\b/gi, 'schur complement'],
  [/\bsecondry\b|\bseconday\b/gi, 'secondary'],
  [/\bteritiary\b|\btertuary\b/gi, 'tertiary'],
  [/\bbredth\b/gi, 'breadth'],
  [/\bbreath\s+first\b/gi, 'breadth first'],
  [/\bdd\s+(shape|order|sized?)\b/gi, 'odd $1'],
  [/\bdd\s+matrix\b/gi, 'odd matrix'],
];

const MATH_CONCEPTS: Record<string, string> = {
  integration:
    'Integration is a way of adding up many tiny pieces to find a total. On a graph it gives the area under a curve, and as an antiderivative it reverses differentiation: if F\'(x) = f(x), then the integral of f(x) is F(x) + C. The “+ C” is the constant of integration because differentiating a constant gives zero.',
  differentiation:
    'Differentiation finds the instantaneous rate of change of a function — the slope of its graph at a point. If integration combines tiny pieces into a total, differentiation splits a total into its tiny changes. For a power such as x^n, bring the exponent down in front and reduce the exponent by one.',
  calculus:
    'Calculus is the branch of mathematics that studies continuous change. Differential calculus measures rates of change, while integral calculus adds up small pieces to find totals such as areas.',
};

const NILPOTENT_EIGENVALUES =
  'All eigenvalues of a nilpotent matrix are 0. If N^k = 0 and Nv = λv, then λ^k = 0, so every eigenvalue λ must be 0.';

const SKEW_ODD_DETERMINANT =
  'The determinant of an odd-order skew-symmetric matrix is 0. Since Aᵀ = -A, det(A) = det(Aᵀ) = det(-A) = (-1)^n det(A); for odd n this gives det(A) = -det(A), hence det(A) = 0.';

const SKEW_EVEN_DETERMINANT =
  'For an even-order skew-symmetric matrix the determinant is not forced to be 0. It equals the square of the Pfaffian, so a real example can have a positive determinant.';

const SKEW_GENERAL_DETERMINANT =
  'For a skew-symmetric matrix A, det(A) = det(-A) = (-1)^n det(A). So an odd-order matrix always has determinant 0; an even-order matrix may have a non-zero determinant.';

/** Remove conversational lead-ins so "ok so what is X" routes like "what is X". */
export function normalizeQuestion(raw: string): string {
  let value = String(raw || '').trim();
  // Keep a bare greeting intact so the backend can answer it directly. Leading
  // filler stripping must still apply to "hi what is X".
  if (greetingAnswer(value)) return value;
  while (FILLER_PREFIX.test(value)) value = value.replace(FILLER_PREFIX, '').trim();
  for (const [pattern, replacement] of COMMON_TYPOS) value = value.replace(pattern, replacement);
  return value;
}

const looksLikeExpression = (value: string): boolean =>
  /\d|[+\-*/^=()]|\b(?:sin|cos|tan|sqrt|log|ln|exp)\b|\b[xytuvn]\b/i.test(value);

type Power = { coefficient: number; exponent: number };

/** Parse simple powers such as x^2, 3x^2, x squared or 2*x^3. */
function parsePower(raw: string): Power | null {
  const text = String(raw || '')
    .replace(/\s+/g, '')
    .replace(/\*\*/g, '^')
    .replace(/×/g, '*')
    .replace(/x²/gi, 'x^2')
    .replace(/x³/gi, 'x^3')
    .replace(/xsquared/gi, 'x^2')
    .replace(/xcubed/gi, 'x^3');
  const match = text.match(/^(?:(\d+(?:\.\d+)?)\*?)?x(?:\^(\d+(?:\.\d+)?))?$/i);
  if (!match) return null;
  return {
    coefficient: match[1] ? Number(match[1]) : 1,
    exponent: match[2] !== undefined ? Number(match[2]) : 1,
  };
}

const formatNumber = (value: number): string =>
  Number.isInteger(value) ? String(value) : String(Number(value.toFixed(6)));

const gcd = (a: number, b: number): number => (b === 0 ? Math.abs(a) : gcd(b, a % b));

const powerTerm = (coefficient: number, exponent: number): string => {
  const prefix = coefficient === 1 ? '' : formatNumber(coefficient);
  if (exponent === 0) return prefix || '1';
  if (exponent === 1) return `${prefix}x`;
  return `${prefix}x^${formatNumber(exponent)}`;
};

const integralTerm = ({ coefficient, exponent }: Power): string => {
  const nextExponent = exponent + 1;
  if (Number.isInteger(coefficient)) {
    const divisor = gcd(coefficient, nextExponent) || 1;
    const numerator = coefficient / divisor;
    const denominator = nextExponent / divisor;
    if (denominator === 1) return powerTerm(numerator, nextExponent);
    return `${powerTerm(numerator, nextExponent)}/${formatNumber(denominator)}`;
  }
  return `${formatNumber(coefficient / nextExponent)}x^${formatNumber(nextExponent)}`;
};

const derivativeTerm = ({ coefficient, exponent }: Power): string => {
  const nextCoefficient = coefficient * exponent;
  if (exponent === 0) return '0';
  return powerTerm(nextCoefficient, exponent - 1);
};

const fallingFactors = (available: number, take: number): number[] =>
  take < 0 || take > available ? [] : Array.from({ length: take }, (_, index) => available - index);

const productOf = (values: number[]): number => values.reduce((product, value) => product * value, 1);

/** Count digit arrangements so the offline fallback matches the Python math router. */
function localCountingAnswer(question: string): string | null {
  const low = normalizeQuestion(question).toLowerCase();
  const countMatch = low.match(
    /\bhow\s+many\s+(?:(?:even|odd)\s+)?(?:(\d+)\s*-?\s*digit\s+)?(?:(?:even|odd)\s+)?(?:numbers?|ways?)\b/
  );
  if (!countMatch) return null;
  if (!/\b(?:form(?:ed)?|make|made|arrange|arranged|construct|created?)\b/.test(low)) return null;

  const sourceMatch = low.match(/\b(?:by|from|using|with|out\s+of|among)\s+(.+?)(?:[.?!]|$)/);
  if (!sourceMatch) return null;
  const digits = [...new Set((sourceMatch[1].match(/\b\d\b/g) ?? []).map(Number))];
  if (!digits.length) return null;

  const available = digits.length;
  const requested = countMatch[1] ? Number(countMatch[1]) : available;
  if (requested <= 0) return null;

  const withRepeat =
    /\b(?:with|allow(?:ing|ed)?|permit(?:ting|ted)?)\s+(?:repetition|repeats?|repeating)\b|\bdigits?\s+(?:can|may)\s+be\s+(?:repeated|reused|re-used)\b/.test(
      low
    );
  const withoutRepeat =
    /\b(?:without|no)\s+(?:repetition|repeats?|repeating)\b|\bdigits?\s+(?:cannot|can't|must\s+not)\s+be\s+(?:repeated|reused|re-used)\b|\beach\s+digit\s+(?:used|appears)\s+(?:once|only\s+once)\b/.test(
      low
    );
  const repeat = withRepeat && !withoutRepeat;

  // A multi-digit number cannot start with zero.
  const zeroBlocksFirst = digits.includes(0);
  const firstChoices = available - (zeroBlocksFirst ? 1 : 0);
  const digitText = digits.join(', ');
  const setup = countMatch[1]
    ? `For ${requested}-digit numbers from ${digitText}`
    : `Using all ${available} digits ${digitText}`;

  const hasEven = /\beven\b/.test(low);
  const hasOdd = /\bodd\b/.test(low);
  const parity: 'even' | 'odd' | null = hasEven && !hasOdd ? 'even' : hasOdd && !hasEven ? 'odd' : null;

  if (!parity) {
    const factors = repeat
      ? [firstChoices, ...Array(Math.max(0, requested - 1)).fill(available)]
      : [firstChoices, ...fallingFactors(available - 1, requested - 1)];

    let answer: string;
    if (factors.length) {
      const total = productOf(factors);
      const wording = repeat ? 'digits may repeat' : 'digits are used at most once';
      answer = `${total}. ${setup}, ${wording}: ${factors.join(' × ')} = ${total}.`;
    } else {
      answer = `${0}. ${setup}, digits are used at most once, so a ${requested}-digit number is not possible.`;
    }
    if (zeroBlocksFirst) {
      answer += ` A number cannot start with 0, so the first place has ${firstChoices} choices.`;
    }
    if (!repeat) {
      const repeatTotal = productOf([
        firstChoices,
        ...Array(Math.max(0, requested - 1)).fill(available),
      ]);
      answer += ` If repetition were allowed, it would be ${repeatTotal}.`;
    }
    return answer;
  }

  const parityDigits = digits.filter((digit) => digit % 2 === (parity === 'even' ? 0 : 1));
  const parityText = parityDigits.join(', ');
  if (!parityDigits.length) return `0. ${setup}, there is no ${parity} digit in the supplied set.`;

  if (repeat) {
    const factors =
      requested === 1
        ? [parityDigits.length]
        : [firstChoices, ...Array(Math.max(0, requested - 2)).fill(available), parityDigits.length];
    const total = productOf(factors);
    return (
      `${total}. ${setup}, the last digit must be ${parity} (${parityText}) ` +
      `and digits may repeat: ${factors.join(' × ')} = ${total}.`
    );
  }

  if (requested > available) {
    return `${0}. ${setup}, digits are used at most once, so a ${requested}-digit number is not possible.`;
  }
  if (requested === 1) {
    const total = parityDigits.length;
    return `A 1-digit ${parity} number from ${digitText}: ${total}.`;
  }

  let middle = fallingFactors(available - 2, requested - 2);
  if (!middle.length && requested > 2) {
    return `0. ${setup}, a ${requested}-digit number is not possible.`;
  }
  if (!middle.length) middle = [1];

  const lastNonZero = parityDigits.filter((digit) => digit !== 0);
  const terms: number[][] = [];
  if (parityDigits.includes(0)) terms.push([1, available - 1, ...middle]);
  if (lastNonZero.length) {
    terms.push([lastNonZero.length, available - 1 - (zeroBlocksFirst ? 1 : 0), ...middle]);
  }
  const totals = terms.map(productOf);
  const total = totals.reduce((sum, value) => sum + value, 0);
  const calculation =
    totals.length === 1
      ? `${terms[0].join(' × ')} = ${total}`
      : `${terms.map((term) => term.join(' × ')).join(' + ')} = ${totals.join(' + ')} = ${total}`;

  let answer =
    `${total}. ${setup}, the last digit must be ${parity} (${parityText}) ` +
    `and digits are used at most once: ${calculation}.`;
  if (zeroBlocksFirst) answer += ' A number cannot start with 0, so the first place excludes 0.';

  const repeatFactors =
    requested === 1
      ? [parityDigits.length]
      : [firstChoices, ...Array(Math.max(0, requested - 2)).fill(available), parityDigits.length];
  answer += ` If repetition were allowed, it would be ${productOf(repeatFactors)}.`;
  return answer;
}

function binomial(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let value = 1;
  for (let index = 1; index <= k; index++) value = (value * (n - k + index)) / index;
  return Math.round(value);
}

const NUMBER_WORDS: Record<string, number> = {
  zero: 0, a: 1, an: 1, one: 1, single: 1, two: 2, twice: 2, three: 3,
  four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15,
  sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20,
};

const NUMBER_PATTERN =
  '\\d+|zero|one|single|two|twice|three|four|five|six|seven|eight|nine|ten|' +
  'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty';

const STATISTICS_OVERVIEW =
  'Probability measures how likely an event is, from 0 (impossible) to 1 (certain). Start with ' +
  'the sample space of all outcomes; when outcomes are equally likely, P(A) = favourable/total. ' +
  'Core rules: P(not A) = 1 - P(A); P(A or B) = P(A) + P(B) - P(A and B); for independent events ' +
  'P(A and B) = P(A)·P(B); and P(A given B) = P(A and B)/P(B). A random variable assigns numbers ' +
  'to outcomes, and its distribution gives their probabilities or density. Statistics uses that ' +
  'probability model to collect, summarise and infer from data: the mean and variance describe a ' +
  'sample (the mean is the expected value), while estimation and hypothesis tests generalise to a ' +
  'population. The law of large ' +
  'numbers and central limit theorem are the bridge between the two.';

const STATISTICS_TERMS: Record<string, string> = {
  mean:
    'The mean is the arithmetic average: add all values and divide by how many there are. ' +
    'For a random variable, the equivalent long-run average is the expected value, E[X] = Σx·P(X=x).',
  median:
    'The median is the middle value after sorting the data. With an even number of values it is ' +
    'the average of the two middle ones. It resists extreme values better than the mean.',
  mode:
    'The mode is the value that occurs most often in a data set. A set can have no mode, one ' +
    'mode or several modes, and the mode also works for categorical data.',
  range:
    'The range is the largest value minus the smallest value. It is a quick measure of spread ' +
    'but is sensitive to outliers.',
  variance:
    'Variance measures average squared spread around the mean: Var(X) = E[(X - μ)²]. ' +
    'Its square root is the standard deviation, which is in the original units.',
  'standard deviation':
    'Standard deviation is the square root of variance. It describes the typical distance of ' +
    'values from the mean in the same units as the data; a smaller value means tighter clustering.',
  'expected value':
    'Expected value is the long-run weighted average of a random variable: E[X] = Σx·P(X=x) ' +
    'for discrete outcomes, or ∫x·f(x)dx for a continuous density.',
  'binomial distribution':
    'A binomial distribution counts successes in n independent yes/no trials with the same ' +
    'success probability p: P(X=k) = C(n,k)·p^k·(1-p)^(n-k). Its mean is np and its variance ' +
    'is np(1-p).',
  'normal distribution':
    'A normal distribution is a symmetric bell-shaped continuous distribution defined by its ' +
    'mean μ and standard deviation σ. About 68% of values lie within one σ of the mean and ' +
    'about 95% within two σ.',
};

/** Give complete definitions for common probability and statistics ideas. */
function localStatisticsAnswer(question: string): string | null {
  const low = normalizeQuestion(question).toLowerCase();
  if (!/\b(?:probability|statistics|mean|median|mode|variance|standard deviation|expected value|binomial distribution|normal distribution)\b/.test(low)) {
    return null;
  }
  if (/\b(?:difference|versus|vs\.?|compare)\b/.test(low) && /\bprobability\b/.test(low) && /\bstatistics\b/.test(low)) {
    return STATISTICS_OVERVIEW;
  }
  const definition = low.match(
    /^(?:what\s+is|what's|define|meaning\s+of|explain|tell\s+me\s+about)\s+(?:the\s+)?(.+?)[.?!]*$/
  );
  if (!definition) return null;
  const topic = definition[1].trim();
  if (/\bprobability\s+and\s+statistics\b|\bstatistics\s+and\s+probability\b/.test(topic)) {
    return STATISTICS_OVERVIEW;
  }
  if (['probability', 'probability theory', 'statistics', 'statistical inference'].includes(topic)) {
    return STATISTICS_OVERVIEW;
  }
  return STATISTICS_TERMS[topic] ?? null;
}

/** Answer simple fair-coin and fair-die probability questions offline. */
function localProbabilityAnswer(question: string): string | null {
  const low = normalizeQuestion(question).toLowerCase();
  if (!/\b(?:probability|chance|likelihood)\b/.test(low)) return null;

  const coin = /\bcoin\b|\bheads?\b|\btails?\b/.test(low);
  const die = /\b(?:die|dice)\b/.test(low);
  if (!coin && !die) return null;
  if (/\b(?:biased|unfair|loaded)\b/.test(low)) {
    return 'That depends on the bias. For a fair coin or die, each outcome is equally likely.';
  }

  const probabilityText = (numerator: number, denominator: number, explanation: string): string => {
    const divisor = gcd(numerator, denominator) || 1;
    const reducedDenominator = denominator / divisor;
    const fraction =
      reducedDenominator === 1 ? String(numerator / divisor) : `${numerator / divisor}/${reducedDenominator}`;
    const value = numerator / denominator;
    const decimal = Number(value.toFixed(10));
    const percent = Number((value * 100).toFixed(10));
    return `${fraction} (${decimal}, or ${percent}%). ${explanation}`;
  };

  if (coin) {
    let count = 1;
    const match = low.match(
      new RegExp(`\\b(${NUMBER_PATTERN})\\s+(?:fair\\s+)?(?:coin\\s+)?(?:times|tosses|flips|time|toss|flip)\\b`)
    );
    if (match) {
      const raw = match[1];
      const parsed = /^\d+$/.test(raw) ? Number(raw) : NUMBER_WORDS[raw.toLowerCase()];
      if (parsed === undefined) return null;
      count = parsed;
    } else if (/\btwice\b/.test(low)) {
      count = 2;
    }
    if (count <= 0) return null;
    const total = 2 ** count;
    const tossWord = count === 1 ? 'toss' : 'tosses';

    const asCount = (raw: string): number | null =>
      /^\d+$/.test(raw) ? Number(raw) : NUMBER_WORDS[raw.toLowerCase()] ?? null;
    const singular = (raw: string): string => (raw.toLowerCase().startsWith('head') ? 'head' : 'tail');
    const plural = (word: string, amount: number): string => (amount === 1 ? word : `${word}s`);
    const rangeCount = (lower: number, upper: number): number => {
      let totalOutcomes = 0;
      for (let k = Math.max(0, lower); k <= Math.min(count, upper); k++) totalOutcomes += binomial(count, k);
      return totalOutcomes;
    };

    const atLeast = low.match(
      new RegExp(`\\bat\\s+least\\s+(${NUMBER_PATTERN})\\s+(heads?|tails?)\\b`)
    );
    if (atLeast) {
      const wanted = asCount(atLeast[1]);
      const target = singular(atLeast[2]);
      if (wanted === null) return null;
      const numerator = rangeCount(wanted, count);
      return probabilityText(
        numerator,
        total,
        `In ${count} fair ${tossWord}, there are ${total} equally likely outcomes; ` +
          `${numerator} ${numerator === 1 ? 'is' : 'are'} at least ${wanted} ${plural(target, wanted)}.`
      );
    }

    const atMost = low.match(
      new RegExp(`\\bat\\s+most\\s+(${NUMBER_PATTERN})\\s+(heads?|tails?)\\b`)
    );
    if (atMost) {
      const wanted = asCount(atMost[1]);
      const target = singular(atMost[2]);
      if (wanted === null) return null;
      const numerator = rangeCount(0, wanted);
      return probabilityText(
        numerator,
        total,
        `In ${count} fair ${tossWord}, there are ${total} equally likely outcomes; ` +
          `${numerator} ${numerator === 1 ? 'is' : 'are'} at most ${wanted} ${plural(target, wanted)}.`
      );
    }

    const noMatch = low.match(/\b(?:no|none)\s+(heads?|tails?)\b/);
    if (noMatch) {
      const target = singular(noMatch[1]);
      return probabilityText(
        1,
        total,
        `In ${count} fair ${tossWord}, there are ${total} equally likely outcomes; only 1 has no ${target}.`
      );
    }

    const all = low.match(/\b(?:all|both)\s+(heads?|tails?)\b/);
    if (all) {
      const target = singular(all[1]);
      return probabilityText(
        1,
        total,
        `In ${count} fair ${tossWord}, there are ${total} equally likely outcomes; only 1 is all ${target}.`
      );
    }

    const exact = low.match(
      new RegExp(
        `\\b(?:exactly|precisely|just|only)?\\s*(?:a\\s+|an\\s+)?(${NUMBER_PATTERN})\\s+(heads?|tails?)\\b`
      )
    );
    if (exact) {
      const wanted = asCount(exact[1]);
      const target = singular(exact[2]);
      if (wanted === null) return null;
      const numerator = binomial(count, wanted);
      return probabilityText(
        numerator,
        total,
        `In ${count} fair ${tossWord}, there are ${total} equally likely outcomes; ` +
          `${numerator} ${numerator === 1 ? 'is' : 'are'} exactly ${wanted} ${plural(target, wanted)}.`
      );
    }

    const target = /\bheads?\b/.test(low) ? 'head' : /\btails?\b/.test(low) ? 'tail' : null;
    if (!target) return null;
    if (count === 1) {
      return probabilityText(
        1,
        2,
        `A fair coin has 2 equally likely outcomes, head and tail, so P(${target}) = 1/2.`
      );
    }
    return (
      `Specify how many ${target}s you mean. ` +
      probabilityText(binomial(count, 1), total, `exactly one ${target} in ${count} fair ${tossWord}`) +
      ' ' +
      probabilityText(total - 1, total, `at least one ${target} in ${count} fair ${tossWord}`)
    );
  }

  const faceWords: Record<string, number> = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6 };
  const faceMatch = low.match(
    /\b(?:getting|rolling|roll|show(?:s|ing)?)\s+(?:a\s+)?(one|two|three|four|five|six|[1-6])\b/
  );
  if (faceMatch) {
    const raw = faceMatch[1];
    const value = /^\d+$/.test(raw) ? Number(raw) : faceWords[raw];
    if (value >= 1 && value <= 6) {
      return probabilityText(1, 6, `A fair die has 6 equally likely faces, so P(${raw}) = 1/6.`);
    }
  }
  if (/\beven\b|\bodd\b/.test(low)) {
    const parity = /\beven\b/.test(low) ? 'even' : 'odd';
    return probabilityText(3, 6, `A fair die has 3 ${parity} faces out of 6, so P(${parity}) = 1/2.`);
  }
  return null;
}

/** Explanatory answers for concept questions and simple powers. */
function localMathAnswer(question: string): string | null {
  const low = normalizeQuestion(question).toLowerCase();
  const probability = localProbabilityAnswer(low);
  if (probability) return probability;

  const statistics = localStatisticsAnswer(low);
  if (statistics) return statistics;

  const counting = localCountingAnswer(low);
  if (counting) return counting;

  if (/\bnilpotent\b/.test(low) && /\beigen\s*-?\s*values?\b|\beigenvalues?\b/.test(low)) {
    return NILPOTENT_EIGENVALUES;
  }

  if (/\b(?:skew[\s-]*symmetric|antisymmetric)\b/.test(low) && /\b(?:determinant|det)\b/.test(low)) {
    if (/\bodd\b/.test(low)) return SKEW_ODD_DETERMINANT;
    if (/\beven\b/.test(low)) return SKEW_EVEN_DETERMINANT;
    return SKEW_GENERAL_DETERMINANT;
  }

  if (/\b(?:null|zero)\s+matrix\b/.test(low)) {
    if (/\beigen\s*-?\s*vectors?\b|\beigenvectors?\b/.test(low)) {
      return 'For a null (zero) matrix, every nonzero vector is an eigenvector, because A·v = 0 = 0·v for every vector v. The only eigenvalue is 0.';
    }
    if (/\beigen\s*-?\s*values?\b|\beigenvalues?\b/.test(low)) {
      return 'All eigenvalues of a null (zero) matrix are 0. For an n×n zero matrix, 0 is the only eigenvalue, with algebraic multiplicity n, and every nonzero vector is an eigenvector.';
    }
  }

  if (/\bcalculus\b/.test(low) && !/\d|[+\-*/^]/.test(low)) return MATH_CONCEPTS.calculus;

  if (/\b(?:integrat(?:e|ion|ing)|integral|antiderivative)\b/.test(low)) {
    const match = low.match(/\b(?:integration|integrate|integral|antiderivative)\s+(?:of\s+)?([a-z0-9\s*^()+\-.]+)$/i);
    const source = match?.[1]?.trim() ?? '';
    if (!source || !looksLikeExpression(source)) return MATH_CONCEPTS.integration;
    const power = parsePower(source);
    if (!power) {
      return `${MATH_CONCEPTS.integration}\n\nI can show exact steps for a simple power such as “x^2”.`;
    }
    return (
      `The integral of ${source} is ${integralTerm(power)} + C.\n\n` +
      'Integration is the reverse of differentiation: it adds up tiny pieces to find a total, such as the area under a curve. ' +
      'For a power, raise the exponent by one and divide by the new exponent. ' +
      'The “+ C” appears because a constant disappears when you differentiate.'
    );
  }

  if (/\b(?:differentiat(?:e|ion|ing)|derivative)\b/.test(low)) {
    const match = low.match(/\b(?:differentiate|differentiation(?:\s+of)?|derivative(?:\s+of)?)\s+([a-z0-9\s*^()+\-.]+)$/i);
    const source = match?.[1]?.trim() ?? '';
    if (!source || !looksLikeExpression(source)) return MATH_CONCEPTS.differentiation;
    const power = parsePower(source);
    if (!power) {
      return `${MATH_CONCEPTS.differentiation}\n\nI can show exact steps for a simple power such as “x^2”.`;
    }
    return (
      `The derivative of ${source} is ${derivativeTerm(power)}.\n\n` +
      'Differentiation measures the instantaneous rate of change — the slope of the curve at a point. ' +
      'For a power, bring the exponent down and reduce it by one.'
    );
  }

  return null;
}

/** Safe Shunting Yard evaluator for basic arithmetic (no eval / new Function). */
function evaluateArithmetic(question: string): string | null {
  const expression = question
    .replace(/^(?:what is|whats|calculate|solve|compute)\s+/i, '')
    .replace(/[?=]\s*$/, '')
    .trim();
  if (!/^[\d\s+\-*/().^]+$/.test(expression) || !/[+\-*/^]/.test(expression)) return null;

  const tokens = expression.match(/\d+(?:\.\d+)?|[+\-*/^()]/g);
  if (!tokens) return null;

  const output: string[] = [];
  const operators: string[] = [];
  for (const token of tokens) {
    if (/^\d/.test(token)) {
      output.push(token);
    } else if (token === '(') {
      operators.push(token);
    } else if (token === ')') {
      while (operators.length && operators[operators.length - 1] !== '(') {
        output.push(operators.pop()!);
      }
      if (!operators.length) return null;
      operators.pop();
    } else {
      while (
        operators.length &&
        operators[operators.length - 1] !== '(' &&
        (PRECEDENCE[operators[operators.length - 1]] ?? 0) >= (PRECEDENCE[token] ?? 0)
      ) {
        output.push(operators.pop()!);
      }
      operators.push(token);
    }
  }
  while (operators.length) {
    const op = operators.pop()!;
    if (op === '(') return null;
    output.push(op);
  }

  const stack: number[] = [];
  for (const token of output) {
    if (/^\d/.test(token)) {
      stack.push(Number(token));
      continue;
    }
    const right = stack.pop();
    const left = stack.pop();
    if (right === undefined || left === undefined) return null;
    switch (token) {
      case '+': stack.push(left + right); break;
      case '-': stack.push(left - right); break;
      case '*': stack.push(left * right); break;
      case '/':
        if (right === 0) return null;
        stack.push(left / right);
        break;
      case '^': stack.push(left ** right); break;
      default: return null;
    }
  }
  if (stack.length !== 1 || !Number.isFinite(stack[0])) return null;
  const value = Number(stack[0].toFixed(10));
  return `${expression} = ${value}`;
}

/** Answer "X and also tell me Y" when both halves are math questions. */
function combinedMathAnswer(question: string): string | null {
  const parts = normalizeQuestion(question)
    .split(/\s+\band\s+(?:also\s+)?(?:tell\s+me\s+)?|\s+\balso\s+tell\s+me\s+/gi)
    .map((part) => part.trim().replace(/^[.!?]+|[.!?]+$/g, ''))
    .filter(Boolean);
  if (parts.length < 2) return null;

  const answers = parts.map((part) => localMathAnswer(part));
  if (answers.some((answer) => !answer)) return null;
  return answers.join('\n\n');
}

/** Return a reliable local answer for common general questions, else null. */
export function localFallbackAnswer(question: string): string | null {
  // A bare greeting disappears under filler stripping, so answer it first.
  const greeting = greetingAnswer(question);
  if (greeting) return greeting;

  const trimmed = normalizeQuestion(question);
  const combined = combinedMathAnswer(trimmed);
  if (combined) return combined;

  const direct = smartAnswer(trimmed);
  if (direct) return direct;

  const math = localMathAnswer(trimmed);
  if (math) return math;

  const arithmetic = evaluateArithmetic(trimmed);
  if (arithmetic) return arithmetic;

  const low = trimmed.toLowerCase();
  const person = PEOPLE_FACTS.find((fact) => fact.keys.some((key) => low.includes(key)));
  if (person) {
    if (/\b(write|wrote|written|writes|book|books|work|works|author|publish|published)\b/.test(low)) {
      return person.works;
    }
    return person.about;
  }

  const country = WORLD_FACTS.find((fact) => fact.keys.some((key) => low.includes(key)));
  if (!country) return null;
  if (/\bpresident\b/.test(low)) return country.president;
  if (/^(?:tell me about|what about|who is|describe|explain)\b/.test(low)) return country.about;
  return null;
}
