/**
 * Curated answers for conversational questions that keyword search mishandles.
 *
 * This is the browser-side twin of python/smart_answers.py. It keeps Converse
 * intelligent even when the Python service is unavailable.
 */

const FILLER_PREFIX =
  /^(?:ok(?:ay)?|so|and|now|well|yeah|yes|but|also|please|hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening))\b[\s,.!]*/i;
const LEAD_IN = /^(?:tell me about|what about|what is|what are|what's|who is|describe|explain)\s+/i;
const COMMON_TYPOS: Array<[RegExp, string]> = [
  [/\boart\b/gi, 'part'],
  [/\bthus\b/gi, 'this'],
  [/\bschur\s+components?\b/gi, 'schur complement'],
  [/\bsecondry\b|\bseconday\b/gi, 'secondary'],
  [/\bteritiary\b|\btertuary\b/gi, 'tertiary'],
  [/\bbredth\b/gi, 'breadth'],
  [/\bbreath\s+first\b/gi, 'breadth first'],
];

const GREETING =
  /^(?:(?:(?:hi|hello|hey|yo|howdy|greetings|namaste)(?:\s+there)?|good\s+(?:morning|afternoon|evening))\b(?:\s*[,!.]?\s*(?:how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up|how\s+do\s+you\s+do))?|(?:how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up|how\s+do\s+you\s+do)\b)[\s.!?]*$/i;
const SMALL_TALK =
  /^(?:i\s+(?:just\s+)?(?:want|wanna|wanted|would\s+like)(?:\s+to)?\s+(?:chat|talk)|(?:let'?s|lets)\s+(?:chat|talk)|can\s+we\s+(?:chat|talk)|i\s+feel\s+like\s+(?:chatting|talking)|chat\s+with\s+me)[\s.!?]*$/i;

/** Mirror the Python greeting map so offline chat is polite about small slips. */
const GREETING_TYPOS: Array<[RegExp, string]> = [
  [/\bhii+\b/gi, 'hi'],
  [/\bhelo\b|\bhelloo+\b|\bhelooo\b/gi, 'hello'],
  [/\bhey{2,}a?\b|\bhei\b/gi, 'hey'],
  [/\bgud\b|\bgoood\b|\bgoodd\b/gi, 'good'],
  [/\bmornin\b|\bmorningg\b/gi, 'morning'],
  [/\bafternon\b|\bafternoonn\b/gi, 'afternoon'],
  [/\bevenin\b|\bevenning\b/gi, 'evening'],
  [/\bhow\s+r\s+u\b|\bhw\s+r\s+u\b|\bhow\s+r\s+ya\b/gi, 'how are you'],
  [/\bwat'?s\s+up\b|\bwhas\s+up\b/gi, "what's up"],
];

const correctGreeting = (raw: string): string => {
  let value = String(raw || '').trim();
  for (const [pattern, replacement] of GREETING_TYPOS) {
    value = value.replace(pattern, replacement);
  }
  return value.trim();
};

const CAPITALS: Record<string, [string, string]> = {
  india: ['India', 'New Delhi'],
  bharat: ['India', 'New Delhi'],
  spain: ['Spain', 'Madrid'],
  france: ['France', 'Paris'],
  japan: ['Japan', 'Tokyo'],
  germany: ['Germany', 'Berlin'],
  italy: ['Italy', 'Rome'],
  uk: ['the United Kingdom', 'London'],
  'united kingdom': ['the United Kingdom', 'London'],
  usa: ['the United States', 'Washington, D.C.'],
  'u.s.a.': ['the United States', 'Washington, D.C.'],
  'united states': ['the United States', 'Washington, D.C.'],
  america: ['the United States', 'Washington, D.C.'],
};

const NCT_PARTS =
  'The National Capital Territory (NCT) of Delhi has 11 districts: Central Delhi, East Delhi, ' +
  'New Delhi, North Delhi, North East Delhi, North West Delhi, Shahdara, South Delhi, ' +
  'South East Delhi, South West Delhi and West Delhi. New Delhi is one district inside the NCT, ' +
  'not the whole territory.';

const NCR_VS_NCT =
  "Yes, they are different. NCT (National Capital Territory of Delhi) is a union territory that " +
  'contains New Delhi and is administered by the Delhi government. NCR (National Capital Region) ' +
  'is a much larger planning region around it, covering the whole NCT plus nearby districts in ' +
  'Haryana, Uttar Pradesh and Rajasthan.';

const BEST_CITY =
  "There isn't one official best city — it depends on what you enjoy. New York is a strong pick " +
  'for food, museums and energy; Paris for art and history; Tokyo for food and neighbourhoods; ' +
  'and Istanbul for history across two continents. Tell me what you like and I can narrow it down.';

const NEW_YORK_BEST =
  "New York is a great choice, but 'best' is subjective. It stands out for museums, theatre, food " +
  'and neighbourhood variety. Paris, Tokyo and Istanbul are just as strong if you prefer art, ' +
  'history or a slower pace, so treat New York as one of the top contenders rather than an official winner.';

const FEDERALISM =
  'Federalism is a system of government where power is divided between a central/national ' +
  'government and smaller state or regional governments. Both levels have their own ' +
  'responsibilities — national defence and foreign policy usually sit with the centre, while ' +
  'education, health and local policing are often regional — and a constitution defines the ' +
  'boundary. The United States, India, Germany and Brazil are examples.';

const BRICS =
  'BRICS is a grouping of major emerging economies that promotes economic and political ' +
  'cooperation and a more multipolar world. It began as BRIC with Brazil, Russia, India and ' +
  'China; South Africa joined in 2010, and Egypt, Ethiopia, Iran, the United Arab Emirates ' +
  'and Indonesia joined later. Membership and partner status can change, so check the latest ' +
  'official list for a precise current count.';

const WHO_ARE_YOU =
  'I’m ViDo’s chat assistant. I’m here for a quick chat, to book a task, or to plan tasks from ' +
  'an attachment you share. I can also create tasks with dates and flowcharts and show them ' +
  'inside the chat.';

const GREETING_ANSWER =
  'Hi there! Good to see you. How can I help today — a question, a quick chat, or a task ' +
  'you’d like me to book?';

const HOW_ARE_YOU_ANSWER =
  'I’m doing well, thanks for asking! I’m here for a chat or to help you book a task. ' +
  'Ask me anything, or attach a plan and say `book this plan`.';

const A_STAR =
  'A* (A-star) is a pathfinding algorithm. At each step it chooses the node with the smallest ' +
  'f(n) = g(n) + h(n): g(n) is the cost already paid to reach the node, and h(n) is an estimated ' +
  'cost to the goal. With an admissible and consistent heuristic, A* is complete and finds the ' +
  'cheapest path. It is widely used in maps, games and robotics.';

const BREADTH_FIRST =
  'Breadth-first search (BFS) explores a graph level by level from a starting node, using a queue. ' +
  'It visits every neighbour at the current depth before moving deeper, so on an unweighted graph ' +
  'it finds the shortest path by number of edges. Its time complexity is O(V + E).';

const ECONOMIC_SECTORS =
  'Economies are commonly grouped into three sectors. Primary extracts raw materials — farming, ' +
  'mining, fishing and forestry. Secondary turns those materials into goods — manufacturing, ' +
  'construction and energy production. Tertiary provides services — transport, retail, healthcare, ' +
  'education and banking. Some models add a quaternary sector for research, information and technology.';

const BARTER =
  'Yes. Barter is still used, usually alongside money: neighbours swap skills or goods, ' +
  'businesses use trade networks, and communities may rely on it when cash is scarce. It is ' +
  'less common than money because barter needs a double coincidence of wants — each person ' +
  'must want what the other offers.';

const JAVA_LANGUAGE =
  'Java is a high-level, object-oriented programming language designed to run on a virtual ' +
  'machine. You write source code, compile it to bytecode, and a JVM on Windows, macOS or Linux ' +
  'runs that same bytecode. If you meant the Indonesian island, say “Java island” and I’ll answer that instead.';

const WORA =
  'WORA stands for Write Once, Run Anywhere. Java source is compiled to platform-neutral ' +
  'bytecode instead of a Windows or macOS executable. Each platform has its own JVM that reads ' +
  'and runs that bytecode, so the same .class file works wherever a compatible JVM exists. ' +
  'Caveat: platform-specific libraries or native code can still break portability.';

const SCHUR_COMPLEMENT =
  'For a block matrix M = [[A, B], [C, D]] where A is invertible, the Schur complement of A in M ' +
  'is S = D - C A⁻¹ B. It comes from block Gaussian elimination and gives det(M) = det(A) · det(S). ' +
  'If D is invertible instead, the Schur complement of D is A - B D⁻¹ C.';

const CREATE_DATABASE =
  'To create a new database, run `CREATE DATABASE my_app_db;` in MySQL or PostgreSQL. ' +
  'Then select it with `USE my_app_db;` in MySQL or `\\c my_app_db` in PostgreSQL. ' +
  'In SQLite there is no separate CREATE DATABASE step: connecting to a file such as ' +
  '`app.db` creates the database. After that, define tables with `CREATE TABLE ...`.';

const CREATE_COLUMN =
  'Use `ALTER TABLE`. For example, `ALTER TABLE users ADD COLUMN email VARCHAR(255);` adds an ' +
  'email column to the users table. Replace the table name, column name and data type as needed. ' +
  'PostgreSQL supports `ADD COLUMN IF NOT EXISTS`; SQLite supports the basic form, but a NOT NULL ' +
  'column needs a default value.';

const SQLALCHEMY =
  'SQLAlchemy is a Python SQL toolkit and object-relational mapper (ORM). Its Core layer builds ' +
  'SQL expressions and manages connections; its ORM maps Python classes to database tables so you ' +
  "query and update with objects. Install it with `pip install sqlalchemy`, create an engine with " +
  "`create_engine('sqlite:///app.db')`, then use sessions or Core statements to work with data.";

const PYTHON_IF =
  'In Python, `if` is a conditional statement: its indented block runs only when the expression is ' +
  'truthy. Add `elif` for another branch and `else` for the fallback. For example: ' +
  "`x = 5`, then `if x > 0:`, `    print('positive')`, `else:`, `    print('not positive')`. " +
  'Indentation marks the block, and `if` works with any truthy or falsy value, not just comparisons.';

const normalize = (raw: string): string => {
  let value = String(raw || '').trim();
  while (FILLER_PREFIX.test(value)) value = value.replace(FILLER_PREFIX, '').trim();
  for (const [pattern, replacement] of COMMON_TYPOS) value = value.replace(pattern, replacement);
  value = value.replace(/\s+/g, ' ').trim().replace(/^[.!\?]+|[.!\?]+$/g, '');
  while (LEAD_IN.test(value)) value = value.replace(LEAD_IN, '').trim();
  return value.replace(/^[.!\?]+|[.!\?]+$/g, '');
};

/** Return a warm reply for a bare greeting, not for "hi what is X". */
export function greetingAnswer(question: string): string | null {
  const text = correctGreeting(question);
  if (!GREETING.test(text) && !SMALL_TALK.test(text)) return null;
  return /how\s+are\s+you|how'?s\s+it\s+going|how\s+do\s+you\s+do/i.test(text)
    ? HOW_ARE_YOU_ANSWER
    : GREETING_ANSWER;
}

/** Return a direct answer for a known conversational shape, else null. */
export function smartAnswer(question: string): string | null {
  const greeting = greetingAnswer(question);
  if (greeting) return greeting;

  const text = normalize(question);
  if (!text) return null;

  const lower = text.toLowerCase();
  if (/^(?:who\s+are\s+you|what\s+are\s+you|introduce\s+yourself|yourself)$/.test(lower)) {
    return WHO_ARE_YOU;
  }

  if (/^federalism\b/.test(lower)) return FEDERALISM;

  if (/\ba\s*\*\s*(?:star\s*)?(?:algorithm|search)\b|\ba\s+star\s+(?:algorithm|search)\b/i.test(text)) {
    return A_STAR;
  }

  if (/\bbreadth\s*[- ]?first\s+(?:search|traversal)\b|\bBFS\b/i.test(text)) return BREADTH_FIRST;

  if (/\bsectors?\b/i.test(text) && /\bprimary\b|\bsecondary\b|\btertiary\b/i.test(text)) {
    return ECONOMIC_SECTORS;
  }

  if (/\bbrics\b/i.test(text)) return BRICS;

  if (/\bWORA\b|\bwrite\s+once[,\s]+run\s+anywhere\b/i.test(text)) return WORA;

  if (/\bschur\s+complement\b/i.test(text)) return SCHUR_COMPLEMENT;

  if (/\bsql\s*alchemy\b/i.test(text)) return SQLALCHEMY;

  if (
    /\bpython\b/i.test(text) &&
    /\b(?:if|elif|else)\b/i.test(text) &&
    /\b(?:command|commands|statement|statements|keyword|syntax|condition|conditional|use|used|using|work|works)\b/i.test(
      text
    )
  ) {
    return PYTHON_IF;
  }

  if (/\bcolumn\b/i.test(text) && /\b(?:add|create|make|new)\b/i.test(text)) {
    return CREATE_COLUMN;
  }

  if (/\bdatabase\b/i.test(text) && /\b(?:create|make|start|new)\b/i.test(text)) {
    return CREATE_DATABASE;
  }

  if (/\bbarter\b/i.test(text)) return BARTER;

  if (/\bjava\b/i.test(text) && !/\b(?:island|indonesia|sunda|jakarta)\b/i.test(text)) {
    return JAVA_LANGUAGE;
  }

  const capital = text.match(/^(?:the\s+)?capital\s+of\s+([a-z][\w\s.-]+)$/i);
  if (capital) {
    const entry = CAPITALS[capital[1].trim().toLowerCase().replace(/\.$/, '')];
    if (entry) {
      const [country, city] = entry;
      const suffix = /india|bharat/i.test(country) ? " It has been India's capital since 1911." : '';
      return `The capital of ${country} is ${city}.${suffix}`;
    }
  }

  if (lower.includes('nct') && (/(?:which|what)\b.*\bparts?\b|\bparts?\b.*\b(?:in|of|located|within)\b/i.test(text))) {
    return NCT_PARTS;
  }

  if (lower.includes('ncr') && lower.includes('nct') && /\b(?:different|difference|versus|vs|compare|both|same)\b/i.test(text)) {
    return NCR_VS_NCT;
  }

  if (/\bbest\b.*\bcity\b|\bcity\b.*\bbest\b/i.test(text) && /\b(?:explore|visit|travel|world|choose)\b/i.test(text)) {
    return /\b(?:new york|nyc)\b/i.test(text) ? NEW_YORK_BEST : BEST_CITY;
  }

  return null;
}

export default smartAnswer;
