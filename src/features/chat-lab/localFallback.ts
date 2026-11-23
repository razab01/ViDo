/**
 * Small local safety net used only when the Python service is unreachable.
 * It covers the common questions that the dataset RAG engine cannot answer,
 * so a missing backend never produces an unrelated task-management snippet.
 */
import { smartAnswer } from './smartAnswers';

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

const FILLER_PREFIX = /^(?:ok(?:ay)?|so|and|now|well|yeah|yes|but|please|also)\b[\s,.!]*/i;
const COMMON_TYPOS: Array<[RegExp, string]> = [
  [/\boart\b/gi, 'part'],
  [/\bthus\b/gi, 'this'],
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

/** Explanatory answers for concept questions and simple powers. */
function localMathAnswer(question: string): string | null {
  const low = normalizeQuestion(question).toLowerCase();

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
