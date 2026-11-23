/**
 * Curated answers for conversational questions that keyword search mishandles.
 *
 * This is the browser-side twin of python/smart_answers.py. It keeps Chat Lab
 * intelligent even when the Python service is unavailable.
 */

const FILLER_PREFIX = /^(?:ok(?:ay)?|so|and|now|well|yeah|yes|but|also|please)\b[\s,.!]*/i;
const LEAD_IN = /^(?:tell me about|what about|what is|what are|what's|who is|describe|explain)\s+/i;
const COMMON_TYPOS: Array<[RegExp, string]> = [
  [/\boart\b/gi, 'part'],
  [/\bthus\b/gi, 'this'],
];

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

const normalize = (raw: string): string => {
  let value = String(raw || '').trim();
  while (FILLER_PREFIX.test(value)) value = value.replace(FILLER_PREFIX, '').trim();
  for (const [pattern, replacement] of COMMON_TYPOS) value = value.replace(pattern, replacement);
  value = value.replace(/\s+/g, ' ').trim().replace(/^[.!\?]+|[.!\?]+$/g, '');
  while (LEAD_IN.test(value)) value = value.replace(LEAD_IN, '').trim();
  return value.replace(/^[.!\?]+|[.!\?]+$/g, '');
};

/** Return a direct answer for a known conversational shape, else null. */
export function smartAnswer(question: string): string | null {
  const text = normalize(question);
  if (!text) return null;

  const capital = text.match(/^(?:the\s+)?capital\s+of\s+([a-z][\w\s.-]+)$/i);
  if (capital) {
    const entry = CAPITALS[capital[1].trim().toLowerCase().replace(/\.$/, '')];
    if (entry) {
      const [country, city] = entry;
      const suffix = /india|bharat/i.test(country) ? " It has been India's capital since 1911." : '';
      return `The capital of ${country} is ${city}.${suffix}`;
    }
  }

  const lower = text.toLowerCase();
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
