/**
 * Small local safety net used only when the Python service is unreachable.
 * It covers the common questions that the dataset RAG engine cannot answer,
 * so a missing backend never produces an unrelated task-management snippet.
 */

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

const PRECEDENCE: Record<string, number> = { '+': 1, '-': 1, '*': 2, '/': 2, '^': 3 };

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

/** Return a reliable local answer for common general questions, else null. */
export function localFallbackAnswer(question: string): string | null {
  const trimmed = question.trim();
  const arithmetic = evaluateArithmetic(trimmed);
  if (arithmetic) return arithmetic;

  const low = trimmed.toLowerCase();
  const country = WORLD_FACTS.find((fact) => fact.keys.some((key) => low.includes(key)));
  if (!country) return null;
  if (/\bpresident\b/.test(low)) return country.president;
  if (/^(?:tell me about|what about|who is|describe|explain)\b/.test(low)) return country.about;
  return null;
}
